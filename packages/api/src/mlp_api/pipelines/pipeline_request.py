import json

from jsonschema import Draft202012Validator, SchemaError
from pydantic import TypeAdapter, ValidationError
from sqlalchemy import Engine

from mlp_api.datasets.registry import find_dataset_version
from mlp_api.endpoints.endpoint_model import tool_parser_tag
from mlp_api.endpoints.lifecycle import find_endpoint
from mlp_api.models.mlflow import MLflow
from mlp_api.models.registry import find_referenced_model_version, pin_full_weights
from mlp_api.pipelines.hugging_face import HuggingFace, find_base_model, pin_base_model
from mlp_core import config
from mlp_core.endpoint_spec import EndpointName, ServingOptions
from mlp_core.pipeline_request.references import (
    dataset_reference,
    model_reference,
    split_dataset_reference,
    split_endpoint_reference,
)
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

    # 3. What `distill` reads: its prompts, its Teacher and the tools the Teacher may call.
    if request.distill:
        errors += pin_distill(request, secrets, hugging_face, engine, model_registry)

    # 4. The starting model: the Base Model on Hugging Face, or a full-weight Model Version.
    finetune = request.finetune
    try:
        if finetune and finetune.base_model:
            finetune.base_model = pin_base_model(
                hugging_face, finetune.base_model, secrets.get("hf_token")
            )
        elif finetune:
            finetune.from_ = pin_full_weights(model_registry, finetune.from_)
    except ValueError as reason:
        errors.append(
            error(["finetune", "base_model" if finetune.base_model else "from"], str(reason))
        )

    # 5. The Datasets in the Dataset registry, pinned to a version the algorithm trains on.
    for index, phase in enumerate(finetune.phases if finetune else []):
        loc = ["finetune", "phases", index, "dataset"]
        row_formats = config.ALGORITHMS[phase.algorithm]["row_formats"]
        trains_on = f"{phase.algorithm} trains on {', '.join(row_formats)} rows"
        if phase.dataset == config.DISTILL_OUTPUT and request.distill is None:
            errors.append(error(loc, "`distill` isn't enabled, so name a Dataset"))
        elif phase.dataset == config.DISTILL_OUTPUT:
            if config.DISTILLATION_ROW_FORMAT not in row_formats:
                has = f"`distill` makes {config.DISTILLATION_ROW_FORMAT} rows"
                errors.append(error(loc, f"{has}; {trains_on}"))
        else:
            try:
                phase.dataset = pin_dataset(engine, phase.dataset, row_formats, trains_on)
            except ValueError as reason:
                errors.append(error(loc, str(reason)))

    # 6. The model `evaluate` runs on, pinned; `finetune`'s output unless named.
    if request.evaluate:
        try:
            request.evaluate.model = pin_evaluated_model(
                request, secrets.get("hf_token"), hugging_face, engine, model_registry
            )
        except ValueError as reason:
            errors.append(error(["evaluate", "model"], str(reason)))

    # 7. The Endpoint `serve` starts: named after the Pipeline unless named, and free for now.
    if request.serve:
        name = request.serve.name = request.serve.name or request.name
        if not is_endpoint_name(name):
            errors.append(error(["serve", "name"], f"`{name}` can't name an Endpoint; name one"))
        elif find_endpoint(engine, name) is not None:
            msg = f"Endpoint {name} is already running; stop it or name another"
            errors.append(error(["serve", "name"], msg))

    if errors:
        return None, errors
    return request, []


def error(loc, msg: str) -> dict:
    return {"loc": list(loc), "msg": msg}


def trainer_config_errors(request: PipelineRequest) -> list[dict]:
    errors = []
    for index, phase in enumerate(request.finetune.phases if request.finetune else []):
        # Only the first Phase trains a new Adapter; the others continue it (#19).
        loc = ["finetune", "phases", index, "lora"]
        if index == 0 and phase.lora is None:
            errors.append(error(loc, "the first Phase trains a new Adapter; set its `lora`"))
        elif index > 0 and phase.lora is not None:
            msg = "continues the Adapter of Phase 1, keeping its rank and targets; leave out `lora`"
            errors.append(error(loc, msg))
        algorithm = config.ALGORITHMS[phase.algorithm]
        checks = {
            "settings": (algorithm["config"], algorithm["blocked_settings"]),
            "lora": (config.LORA_CONFIG, config.BLOCKED_LORA_SETTINGS),
        }
        for block, (config_class, blocked) in checks.items():
            settings = getattr(phase, block)
            if settings is None:
                continue
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


def pin_distill(
    request: PipelineRequest,
    secrets: dict[str, str],
    hugging_face: HuggingFace,
    engine: Engine,
    model_registry: MLflow,
) -> list[dict]:
    """Every error with its path, after pinning the prompts, the Teacher and its tool parser."""
    # 1. The prompts, under another name than the Distillation Dataset, which the Pipeline names.
    distill = request.distill
    errors = []
    try:
        if split_dataset_reference(distill.dataset)[0] == request.name:
            raise ValueError(f"the Pipeline's output is Dataset `{request.name}`; rename either")
        reads = f"`distill` reads {config.PROMPT_ROW_FORMAT} rows"
        distill.dataset = pin_dataset(engine, distill.dataset, [config.PROMPT_ROW_FORMAT], reads)
    except ValueError as reason:
        errors.append(error(["distill", "dataset"], str(reason)))

    # 2. Each tool's arguments as a JSON Schema, under a name of its own.
    tools = distill.tools or []
    for index, tool in enumerate(tools):
        try:
            Draft202012Validator.check_schema(tool.function.parameters)
        except SchemaError as reason:
            loc = ["distill", "tools", index, "function", "parameters"]
            errors.append(error(loc, f"is no JSON Schema: {reason.message}"))
    names = [tool.function.name for tool in tools]
    if len(set(names)) < len(names):
        errors.append(error(["distill", "tools"], "two tools have the same name"))

    # 3. An API Teacher, reached with the key from the Secrets.
    if distill.api_url:
        if "teacher_api_key" not in secrets:
            msg = "supply the Teacher's key as the `teacher_api_key` Secret"
            errors.append(error(["distill", "api_url"], msg))
        return errors

    # 4. Any other Teacher pinned, with the tool parser its replies are read with (#17).
    token = secrets.get("hf_token")
    try:
        distill.teacher = pin_served_model(
            distill.teacher, token, hugging_face, engine, model_registry
        )
        if tools:
            parser = distill.serving and distill.serving.tool_parser
            parser = parser or tool_parser_of(
                distill.teacher, token, hugging_face, engine, model_registry
            )
            if parser is None:
                msg = f"{distill.teacher} has no tool parser; name one in `serving.tool_parser`"
                errors.append(error(["distill", "tools"], msg))
            elif not distill.teacher.startswith("endpoint:"):
                serving = distill.serving or ServingOptions()
                distill.serving = serving.model_copy(update={"tool_parser": parser})
    except ValueError as reason:
        errors.append(error(["distill", "teacher"], str(reason)))
    return errors


def pin_dataset(engine: Engine, reference: str, row_formats, reader: str) -> str:
    """The Dataset Reference pinned to a version with rows the reader reads; else ValueError."""
    name, version = split_dataset_reference(reference)
    pinned = find_dataset_version(engine, name, version)
    if pinned is None and version is None:
        raise ValueError(f"no Dataset `{name}`")
    if pinned is None:
        raise ValueError(f"`{name}` has no version {version}")
    if pinned.row_format not in row_formats:
        raise ValueError(f"`{name}@{pinned.version}` has {pinned.row_format} rows; {reader}")
    return dataset_reference(name, pinned.version)


def pin_served_model(
    model: str, token: str | None, hugging_face: HuggingFace, engine: Engine, model_registry: MLflow
) -> str:
    """A pinned Base Model or Model Version, or a running Endpoint; else ValueError."""
    if model.startswith("hf:"):
        return pin_base_model(hugging_face, model, token)
    if model.startswith("endpoint:"):
        name = split_endpoint_reference(model)
        found = find_endpoint(engine, name)
        if found is None or found.status != "running":
            raise ValueError(f"no Endpoint {name} is running")
        return model
    found = find_referenced_model_version(model_registry, model)
    return model_reference(found.name, found.version)


def tool_parser_of(
    model: str, token: str | None, hugging_face: HuggingFace, engine: Engine, model_registry: MLflow
) -> str | None:
    """The tool parser vLLM reads the pinned model's or running Endpoint's tool calls with."""
    if model.startswith("endpoint:"):
        spec = find_endpoint(engine, split_endpoint_reference(model)).spec
        return spec.get("tool_parser") or tool_parser_of(
            spec["model"], None, hugging_face, engine, model_registry
        )
    if model.startswith("hf:"):
        return config.TOOL_PARSERS.get(find_base_model(hugging_face, model, token)[1].model_type)
    # An Adapter tagged with no parser of its own takes its base's, as on an Endpoint.
    found = find_referenced_model_version(model_registry, model)
    if found.tags.get("weights") == "adapter" and "tool_parser" not in found.tags:
        base = found.tags["base_model"]
        return tool_parser_of(base, token, hugging_face, engine, model_registry)
    return tool_parser_tag(found.tags)


def pin_evaluated_model(
    request: PipelineRequest,
    token: str | None,
    hugging_face: HuggingFace,
    engine: Engine,
    model_registry: MLflow,
) -> str:
    """`@finetune`, a pinned Base Model or Model Version, or a running Endpoint; else ValueError."""
    model = request.evaluate.model
    if model in (None, config.FINETUNE_OUTPUT):
        if request.finetune is None:
            raise ValueError("`finetune` isn't enabled, so name the model to evaluate")
        return config.FINETUNE_OUTPUT
    if model.startswith("endpoint:"):
        name = split_endpoint_reference(model)
        if request.evaluate.serving:
            raise ValueError(f"Endpoint {name} serves with its own options; leave out `serving`")
        # Other users' requests to a shared Endpoint would skew the measurement.
        if request.evaluate.performance:
            raise ValueError(f"`performance` is measured on a vLLM of its own, not Endpoint {name}")
    return pin_served_model(model, token, hugging_face, engine, model_registry)


def is_endpoint_name(name: str) -> bool:
    try:
        TypeAdapter(EndpointName).validate_python(name)
    except ValidationError:
        return False
    return True
