# Run with the trainer image's versions: uv run --with trl==X --with peft==Y <this script>
import dataclasses
import json
import typing

import peft
import trl
from pydantic import TypeAdapter

from mlp_core.config import ALGORITHMS, LORA_CONFIG, TRAINER_CONFIGS_DIRECTORY


def inline_definitions(schema: dict) -> dict:
    definitions = schema.pop("$defs", {})

    def inline(node):
        if isinstance(node, dict):
            if "$ref" in node:
                return inline(definitions[node["$ref"].rsplit("/", 1)[1]])
            return {
                key: inline(value)
                for key, value in node.items()
                if key not in ("title", "description")
            }
        if isinstance(node, list):
            return [inline(value) for value in node]
        return node

    return inline(schema)


def field_schema(field_type) -> dict:
    try:
        return inline_definitions(TypeAdapter(field_type).json_schema())
    except Exception:
        # Types without a JSON form (callables, torch types) accept any value; TRL checks them.
        return {}


def default_value(field: dataclasses.Field):
    if field.default_factory is not dataclasses.MISSING:
        value = field.default_factory()
    else:
        value = field.default
    try:
        json.dumps(value)
    except TypeError:
        return dataclasses.MISSING
    return value


def config_schema(config: type) -> dict:
    hints = typing.get_type_hints(config)
    properties = {}
    for field in dataclasses.fields(config):
        if not field.init or field.name.startswith("_"):
            continue
        properties[field.name] = field_schema(hints.get(field.name, typing.Any))
        default = default_value(field)
        if default is not dataclasses.MISSING:
            properties[field.name]["default"] = default
    return {"type": "object", "properties": properties, "additionalProperties": False}


def main() -> None:
    for name in [*ALGORITHMS.values(), LORA_CONFIG]:
        config = getattr(trl, name, None) or getattr(peft, name, None)
        if config is None:
            print(f"{name} is in neither trl nor peft; its settings stay unchecked")
            continue
        schema = json.dumps(config_schema(config), indent=1)
        (TRAINER_CONFIGS_DIRECTORY / f"{name}.json").write_text(schema + "\n")


if __name__ == "__main__":
    main()
