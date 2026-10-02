from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import yaml

from mlp_core.config import MORE_SETTINGS
from mlp_core.pipeline_request.schema import PipelineRequest


@dataclass
class FormField:
    """One input of the Pipeline Request form, or the heading of the object it starts."""

    name: str
    kind: Literal["section", "fixed", "choice", "text", "list", "integer", "number", "more"]
    depth: int
    fixed: object = None
    choices: tuple = ()
    value: str = ""
    error: str = ""

    @property
    def label(self) -> str:
        *parents, last = self.name.split(".")
        return f"{parents[-1]} {int(last) + 1}" if last.isdigit() else last


def pipeline_form(
    values: Mapping[str, str] | None = None, errors: Sequence[dict] = ()
) -> tuple[list[FormField], list[str]]:
    """The form's fields holding `values` and their errors, and the errors no field can show."""
    # 1. A field for every value in the published schema, holding what the user entered.
    fields = schema_fields()
    for field in fields:
        field.value = values.get(field.name, "") if values else ""

    # 2. Each error next to its field, or the nearest object's; the rest above the form.
    by_name = {field.name: field for field in fields}
    unplaced = []
    for error in errors:
        field, rest = nearest_field(by_name, [str(part) for part in error["loc"]])
        message = f"{'.'.join(rest)}: {error['msg']}" if rest else error["msg"]
        if field is None:
            unplaced.append(message)
        else:
            field.error = f"{field.error}; {message}" if field.error else message
    return fields, unplaced


def pipeline_request_from_form(values: Mapping[str, str]) -> tuple[dict, list[dict]]:
    """The Pipeline Request the form's values describe, and the errors of unreadable values."""
    tree, errors = {}, []
    for field in schema_fields():
        value = values.get(field.name, "").strip()
        if field.kind == "fixed":
            put(tree, field.name.split("."), field.fixed)
        elif field.kind == "section" or not value:
            continue
        elif field.kind == "more":
            try:
                more = yaml.safe_load(value)
            except yaml.YAMLError:
                more = None
            if not isinstance(more, dict):
                errors.append({"loc": field.name.split("."), "msg": "must be a YAML mapping"})
                continue
            *parents, _ = field.name.split(".")
            for key, setting in more.items():
                put(tree, [*parents, str(key)], setting)
        elif field.kind == "choice":
            choices = {str(choice): choice for choice in field.choices}
            put(tree, field.name.split("."), choices.get(value, value))
        else:
            put(tree, field.name.split("."), read_value(field.kind, value))
    return as_lists(tree), errors


def schema_fields() -> list[FormField]:
    schema = PipelineRequest.model_json_schema()
    return list(fields_of(schema, schema.get("$defs", {}), "", 0))


def fields_of(node: dict, defs: dict, name: str, depth: int) -> Iterator[FormField]:
    """The fields of one schema node; an array of objects gets as many items as it needs."""
    node = resolve_ref(node, defs)
    types = {option.get("type") for option in node.get("anyOf", [node])}
    if "const" in node:
        yield FormField(name, "fixed", depth, fixed=node["const"])
    elif "enum" in node:
        yield FormField(name, "choice", depth, choices=tuple(node["enum"]))
    elif "object" in types:
        if name:
            yield FormField(name, "section", depth)
        for key, child in node["properties"].items():
            yield from fields_of(child, defs, f"{name}.{key}" if name else key, depth + 1)
        if node.get("additionalProperties") is True:
            yield FormField(f"{name}.{MORE_SETTINGS}", "more", depth + 1)
    elif node.get("type") == "array" and resolve_ref(node["items"], defs).get("type") == "object":
        yield FormField(name, "section", depth)
        for index in range(node.get("minItems", 1)):
            yield from fields_of(node["items"], defs, f"{name}.{index}", depth + 1)
    elif "array" in types:
        yield FormField(name, "list", depth)
    elif types & {"integer", "number"}:
        yield FormField(name, "integer" if "integer" in types else "number", depth)
    else:
        yield FormField(name, "text", depth)


def resolve_ref(node: dict, defs: dict) -> dict:
    return defs[node["$ref"].rsplit("/", 1)[-1]] if "$ref" in node else node


def nearest_field(by_name: dict, loc: list[str]) -> tuple[FormField | None, list[str]]:
    """The field an error at `loc` belongs to, and the part of `loc` below it."""
    for length in range(len(loc), 0, -1):
        name = ".".join(loc[:length])
        more = f"{name}.{MORE_SETTINGS}"
        if length < len(loc) and more in by_name:
            return by_name[more], loc[length:]
        if name in by_name:
            return by_name[name], loc[length:]
    return None, loc


def read_value(kind: str, text: str):
    """The typed value; unreadable numbers stay text, so validation reports them inline."""
    try:
        if kind == "integer":
            return int(text)
        if kind == "number":
            return float(text)
    except ValueError:
        return text
    if kind == "list" and "," in text:
        return [part.strip() for part in text.split(",")]
    return text


def put(tree: dict, parts: list[str], value) -> None:
    *parents, last = parts
    for part in parents:
        tree = tree.setdefault(part, {})
    tree[last] = value


def as_lists(node):
    """The tree with every object keyed 0, 1, 2… turned into a list."""
    if not isinstance(node, dict):
        return node
    if node and all(key.isdigit() for key in node):
        return [as_lists(node[key]) for key in sorted(node, key=int)]
    return {key: as_lists(value) for key, value in node.items()}
