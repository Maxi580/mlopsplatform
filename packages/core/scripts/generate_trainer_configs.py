# Run with the trainer image's versions: uv run --with trl==X --with peft==Y <this script>
import dataclasses
import json
import typing
from pathlib import Path

import peft
import trl
from pydantic import TypeAdapter

from mlp_core.trainers import ALGORITHMS


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
    output = Path(__file__).parents[1] / "src" / "mlp_core" / "trainer_configs"
    configs = [getattr(trl, config) for config in ALGORITHMS.values()]
    for config in [*configs, peft.LoraConfig]:
        schema = config_schema(config)
        (output / f"{config.__name__}.json").write_text(json.dumps(schema, indent=1) + "\n")


if __name__ == "__main__":
    main()
