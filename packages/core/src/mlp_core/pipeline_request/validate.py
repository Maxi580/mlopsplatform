import json
from collections.abc import Iterator

from jsonschema import Draft202012Validator
from pydantic import ValidationError

from mlp_core import config
from mlp_core.pipeline_request.schema import PipelineRequest


def validate_pipeline_request(data: dict) -> tuple[PipelineRequest | None, list[dict]]:
    """The request and every error: missing or malformed values first, then TRL/PEFT checks."""
    try:
        request = PipelineRequest.model_validate(data)
    except ValidationError as error:
        return None, [{"loc": list(e["loc"]), "msg": e["msg"]} for e in error.errors()]
    return request, list(trainer_config_errors(request))


def trainer_config_errors(request: PipelineRequest) -> Iterator[dict]:
    for index, phase in enumerate(request.finetune.phases):
        loc = ["finetune", "phases", index]
        settings_class = config.ALGORITHMS[phase.algorithm]
        yield from config_class_errors([*loc, "settings"], phase.settings, settings_class)
        yield from config_class_errors([*loc, "lora"], phase.lora, config.LORA_CONFIG)


def config_class_errors(loc: list, values, config_class: str) -> Iterator[dict]:
    fields = trainer_config_fields(config_class)
    if fields is None:
        return
    for name, value in values.model_dump().items():
        if name not in fields:
            yield {"loc": [*loc, name], "msg": f"`{name}` is not a {config_class} setting"}
        elif not value_matches(fields[name], value):
            expected = json.dumps({k: v for k, v in fields[name].items() if k != "default"})
            yield {"loc": [*loc, name], "msg": f"`{name}` must match {expected}"}


# This check only adds to the basic one; a missing or broken schema skips it rather than failing.
def trainer_config_fields(config_class: str) -> dict | None:
    try:
        path = config.TRAINER_CONFIGS_DIRECTORY / f"{config_class}.json"
        return json.loads(path.read_text())["properties"]
    except Exception:
        return None


def value_matches(field_schema: dict, value) -> bool:
    try:
        return Draft202012Validator(field_schema).is_valid(value)
    except Exception:
        return True
