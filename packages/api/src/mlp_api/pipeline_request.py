import json

from jsonschema import Draft202012Validator
from pydantic import ValidationError
from sqlalchemy import Engine

from mlp_api.datasets import find_dataset_version
from mlp_api.hugging_face import HuggingFace
from mlp_core import config
from mlp_core.pipeline_request.references import (
    base_model_reference,
    dataset_reference,
    split_base_model_reference,
    split_dataset_reference,
)
from mlp_core.pipeline_request.schema import PipelineRequest


def validate_pipeline_request(
    data: dict, secrets: dict[str, str], hugging_face: HuggingFace, engine: Engine
) -> tuple[PipelineRequest | None, list[dict]]:
    """The request with every Reference pinned, or None and every error with its path."""
    # 1. The schema: required values, types, no unknown fields.
    try:
        request = PipelineRequest.model_validate(data)
    except ValidationError as validation_error:
        return None, [error(e["loc"], e["msg"]) for e in validation_error.errors()]

    # 2. The settings TRL/PEFT would receive, and no Secret value anywhere.
    errors = trainer_config_errors(request) + secret_value_errors(request, secrets)

    # 3. The Base Model on Hugging Face.
    repo, revision = split_base_model_reference(request.finetune.base_model)
    model = hugging_face.find_model(
        repo, revision or config.DEFAULT_HF_REVISION, secrets.get("hf_token")
    )
    base_model_loc = ["finetune", "base_model"]
    if model is None:
        reason = "is missing, gated for this token, or has no such revision"
        errors.append(error(base_model_loc, f"{repo} on Hugging Face {reason}"))
    elif model.needs_remote_code:
        errors.append(error(base_model_loc, f"{repo} needs remote code, which never runs here"))

    # 4. The Datasets in the Dataset registry, pinned to a version.
    for index, phase in enumerate(request.finetune.phases):
        name, version = split_dataset_reference(phase.dataset)
        pinned = find_dataset_version(engine, name, version)
        if pinned is not None:
            phase.dataset = dataset_reference(name, pinned)
        elif version is None:
            errors.append(error(["finetune", "phases", index, "dataset"], f"no Dataset `{name}`"))
        else:
            missing = f"`{name}` has no version {version}"
            errors.append(error(["finetune", "phases", index, "dataset"], missing))

    if errors:
        return None, errors
    request.finetune.base_model = base_model_reference(repo, model.commit)
    return request, []


def error(loc, msg: str) -> dict:
    return {"loc": list(loc), "msg": msg}


def trainer_config_errors(request: PipelineRequest) -> list[dict]:
    errors = []
    for index, phase in enumerate(request.finetune.phases):
        for block in ("settings", "lora"):
            settings = getattr(phase, block)
            config_class = settings.trainer_config
            fields = trainer_config_fields(config_class)
            if fields is None:
                continue
            for name, value in settings.model_dump().items():
                loc = ["finetune", "phases", index, block, name]
                if name not in fields:
                    errors.append(error(loc, f"`{name}` is not a {config_class} setting"))
                elif not value_matches(fields[name], value):
                    expected = {k: v for k, v in fields[name].items() if k != "default"}
                    errors.append(error(loc, f"`{name}` must match {json.dumps(expected)}"))
    return errors


# Settings stay unchecked when the generated schema is missing or broken, rather than failing.
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


def secret_value_errors(request: PipelineRequest, secrets: dict[str, str]) -> list[dict]:
    values = [value for value in secrets.values() if len(value) >= config.MIN_SECRET_LENGTH]
    return [
        error(loc, "contains a Secret value; name the Secret instead")
        for loc, text in strings(request.model_dump(), [])
        if config.HF_TOKEN_PATTERN.search(text) or any(value in text for value in values)
    ]


def strings(node, loc: list) -> list[tuple[list, str]]:
    """Every string inside the request, with its path."""
    if isinstance(node, str):
        return [(loc, node)]
    if isinstance(node, dict):
        items = node.items()
    elif isinstance(node, list):
        items = enumerate(node)
    else:
        return []
    return [found for key, value in items for found in strings(value, [*loc, key])]
