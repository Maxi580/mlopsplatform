import json

from jsonschema import Draft202012Validator
from pydantic import ValidationError
from sqlalchemy import Engine

from mlp_api.datasets.registry import find_dataset_version
from mlp_api.models.mlflow import MLflow
from mlp_api.models.registry import pin_full_weights
from mlp_api.pipelines.hugging_face import HuggingFace, pin_base_model
from mlp_core import config
from mlp_core.pipeline_request.references import dataset_reference, split_dataset_reference
from mlp_core.pipeline_request.schema import PipelineRequest


def validate_pipeline_request(
    data: dict,
    secrets: dict[str, str],
    hugging_face: HuggingFace,
    engine: Engine,
    model_registry: MLflow,
) -> tuple[PipelineRequest | None, list[dict]]:
    """The request with every Reference pinned, or None and every error with its path."""
    # 1. The schema: required values, types, no unknown fields.
    try:
        request = PipelineRequest.model_validate(data)
    except ValidationError as validation_error:
        return None, [error(e["loc"], e["msg"]) for e in validation_error.errors()]

    # 2. The settings TRL/PEFT would receive, and no Secret value anywhere.
    errors = trainer_config_errors(request) + secret_value_errors(request, secrets)

    # 3. The starting model: the Base Model on Hugging Face, or a full-weight Model Version.
    finetune = request.finetune
    try:
        if finetune.base_model:
            finetune.base_model = pin_base_model(
                hugging_face, finetune.base_model, secrets.get("hf_token")
            )
        else:
            finetune.from_ = pin_full_weights(model_registry, finetune.from_)
    except ValueError as reason:
        errors.append(
            error(["finetune", "base_model" if finetune.base_model else "from"], str(reason))
        )

    # 4. The Datasets in the Dataset registry, pinned to a version the algorithm trains on.
    for index, phase in enumerate(request.finetune.phases):
        loc = ["finetune", "phases", index, "dataset"]
        name, version = split_dataset_reference(phase.dataset)
        pinned = find_dataset_version(engine, name, version)
        row_formats = config.ALGORITHMS[phase.algorithm]["row_formats"]
        if pinned is None and version is None:
            errors.append(error(loc, f"no Dataset `{name}`"))
        elif pinned is None:
            errors.append(error(loc, f"`{name}` has no version {version}"))
        elif pinned.row_format not in row_formats:
            has = f"`{name}@{pinned.version}` has {pinned.row_format} rows"
            trains_on = f"{phase.algorithm} trains on {', '.join(row_formats)} rows"
            errors.append(error(loc, f"{has}; {trains_on}"))
        else:
            phase.dataset = dataset_reference(name, pinned.version)

    if errors:
        return None, errors
    return request, []


def error(loc, msg: str) -> dict:
    return {"loc": list(loc), "msg": msg}


def trainer_config_errors(request: PipelineRequest) -> list[dict]:
    errors = []
    for index, phase in enumerate(request.finetune.phases):
        algorithm = config.ALGORITHMS[phase.algorithm]
        checks = {
            "settings": (algorithm["config"], algorithm["blocked_settings"]),
            "lora": (config.LORA_CONFIG, config.BLOCKED_LORA_SETTINGS),
        }
        for block, (config_class, blocked) in checks.items():
            settings = getattr(phase, block)
            fields = trainer_config_fields(config_class) or {}
            for name, value in settings.model_dump().items():
                loc = ["finetune", "phases", index, block, name]
                if name in blocked:
                    errors.append(error(loc, f"`{name}` is blocked; see the README for why"))
                elif not fields:
                    continue
                elif name not in fields:
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
