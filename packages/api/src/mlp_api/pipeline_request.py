import json
from collections.abc import Iterator

from jsonschema import Draft202012Validator
from pydantic import ValidationError

from mlp_api.config import DEFAULT_HF_REVISION, HF_TOKEN_PATTERN, MIN_SECRET_LENGTH
from mlp_api.hugging_face import HuggingFace
from mlp_core import config
from mlp_core.pipeline_request.references import base_model_reference, split_base_model_reference
from mlp_core.pipeline_request.schema import PipelineRequest, TrainerSettings


def resolve_pipeline_request(
    data: dict, secrets: dict[str, str], hugging_face: HuggingFace
) -> tuple[PipelineRequest | None, list[dict]]:
    """The request with every reference pinned, or None and every error with its path."""
    errors = [*remote_code_errors(data), *secret_value_errors(data, secrets)]
    try:
        request = PipelineRequest.model_validate(data)
    except ValidationError as validation_error:
        schema_errors = [error(list(e["loc"]), e["msg"]) for e in validation_error.errors()]
        return None, schema_errors + errors
    errors = [*trainer_config_errors(request), *errors]
    base_model, base_model_errors = pin_base_model(request, secrets, hugging_face)
    errors += base_model_errors
    if errors:
        return None, errors
    finetune = request.finetune.model_copy(update={"base_model": base_model})
    return request.model_copy(update={"finetune": finetune}), []


def error(loc: list, msg: str) -> dict:
    return {"loc": loc, "msg": msg}


def nodes(node, loc: list) -> Iterator[tuple[list, object]]:
    yield loc, node
    if isinstance(node, dict):
        for key, value in node.items():
            yield from nodes(value, [*loc, key])
    elif isinstance(node, list):
        for index, value in enumerate(node):
            yield from nodes(value, [*loc, index])


def remote_code_errors(data: dict) -> Iterator[dict]:
    for loc, _ in nodes(data, []):
        if loc and loc[-1] == "trust_remote_code":
            yield error(loc, "`trust_remote_code` is never allowed: repository code doesn't run")


def secret_value_errors(data: dict, secrets: dict[str, str]) -> Iterator[dict]:
    values = [value for value in secrets.values() if len(value) >= MIN_SECRET_LENGTH]
    for loc, node in nodes(data, []):
        if isinstance(node, str) and (
            HF_TOKEN_PATTERN.search(node) or any(value in node for value in values)
        ):
            yield error(loc, "contains a Secret value; name the Secret instead")


def trainer_config_errors(request: PipelineRequest) -> Iterator[dict]:
    for index, phase in enumerate(request.finetune.phases):
        for name, value in phase:
            if isinstance(value, TrainerSettings):
                yield from trainer_settings_errors(["finetune", "phases", index, name], value)


def trainer_settings_errors(loc: list, settings: TrainerSettings) -> Iterator[dict]:
    config_class = settings.trainer_config
    fields = trainer_config_fields(config_class)
    if fields is None:
        return
    for name, value in settings.model_dump().items():
        if name not in fields:
            yield error([*loc, name], f"`{name}` is not a {config_class} setting")
        elif not value_matches(fields[name], value):
            expected = json.dumps({k: v for k, v in fields[name].items() if k != "default"})
            yield error([*loc, name], f"`{name}` must match {expected}")


# This check only adds to the schema; a missing or broken trainer config skips it, not fails.
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


def pin_base_model(
    request: PipelineRequest, secrets: dict[str, str], hugging_face: HuggingFace
) -> tuple[str, list[dict]]:
    reference = request.finetune.base_model
    repo, revision = split_base_model_reference(reference)
    model = hugging_face.find_model(repo, revision or DEFAULT_HF_REVISION, secrets.get("hf_token"))
    loc = ["finetune", "base_model"]
    if model is None:
        reason = "is missing, gated for this token, or has no such revision"
        return reference, [error(loc, f"{repo} on Hugging Face {reason}")]
    if model.needs_remote_code:
        return reference, [error(loc, f"{repo} needs remote code, which never runs here")]
    return base_model_reference(repo, model.commit), []
