import ast
import json

from jsonschema import Draft202012Validator, SchemaError
from pydantic import TypeAdapter, ValidationError
from sqlalchemy import Engine

from mlp_api.datasets.registry import find_dataset_version
from mlp_api.endpoints.endpoint_model import tool_parser_tag
from mlp_api.endpoints.lifecycle import find_endpoint
from mlp_api.models.mlflow import MLflow
from mlp_api.models.registry import find_referenced_model_version, pin_full_weights
from mlp_api.models.upload_checks import adapter_problems
from mlp_api.pipelines.chat_template import assistant_mask_problem
from mlp_api.pipelines.hugging_face import HuggingFace, find_base_model, pin_base_model
from mlp_api.storage.object_store import ObjectStore
from mlp_core import config
from mlp_core.endpoint_spec import EndpointName, ServingOptions
from mlp_core.pipeline_request.references import (
    dataset_reference,
    model_reference,
    split_base_model_reference,
    split_dataset_reference,
    split_endpoint_reference,
)
from mlp_core.pipeline_request.schema import Phase, PipelineRequest


def validate_pipeline_request(
    data: dict,
    secrets: dict[str, str],
    hugging_face: HuggingFace,
    engine: Engine,
    model_registry: MLflow,
    object_store: ObjectStore,
) -> tuple[PipelineRequest | None, list[dict]]:
    """The request with every Reference pinned, or None and every error with its path."""
    # 1. The schema: required values, types, no unknown fields.
    try:
        request = PipelineRequest.model_validate(data)
    except ValidationError as validation_error:
        return None, [error(e["loc"], e["msg"]) for e in validation_error.errors()]

    # 2. Phases the backend can train, the settings TRL/PEFT would receive, rewards that can run,
    # and no Secret value anywhere.
    errors = backend_errors(request) + trainer_config_errors(request) + reward_errors(request)
    errors += secret_value_errors(request, secrets)

    # 3. What `distill` reads: its prompts, its Teacher and the tools the Teacher may call.
    if request.distill:
        errors += pin_distill(request, secrets, hugging_face, engine, model_registry)

    # 4. The starting model: the Base Model on Hugging Face, or a full-weight Model Version.
    finetune = request.finetune
    starting_model_found = False
    try:
        if finetune and finetune.base_model:
            finetune.base_model = pin_base_model(
                hugging_face, finetune.base_model, secrets.get("hf_token")
            )
        elif finetune:
            finetune.from_ = pin_full_weights(model_registry, finetune.from_)
        starting_model_found = True
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

    # 6. `assistant_only_loss` on messages rows, which the starting model's template can mask.
    if finetune and starting_model_found:
        errors += assistant_only_loss_errors(
            request, secrets.get("hf_token"), hugging_face, engine, model_registry, object_store
        )

    # 7. Each Phase's Teacher: a pinned Base Model or Model Version.
    for index, phase in enumerate(finetune.phases if finetune else []):
        if phase.teacher:
            try:
                phase.teacher = pin_served_model(
                    phase.teacher, secrets.get("hf_token"), hugging_face, engine, model_registry
                )
            except ValueError as reason:
                errors.append(error(["finetune", "phases", index, "teacher"], str(reason)))

    # 8. The model `evaluate` runs on, pinned; `finetune`'s output unless named. BFCL scores its
    # tool calls as `serve` would parse them, with the parser of the model `finetune` starts from.
    evaluate = request.evaluate
    if evaluate:
        token = secrets.get("hf_token")
        try:
            evaluate.model = pin_evaluated_model(
                request, token, hugging_face, engine, model_registry
            )
            if any(benchmark.startswith("bfcl:") for benchmark in evaluate.benchmarks):
                model = evaluate.model
                if model == config.FINETUNE_OUTPUT:
                    model = finetune.starting_model
                try:
                    evaluate.serving = pin_tool_parser(
                        model, evaluate.serving, token, hugging_face, engine, model_registry
                    )
                except ValueError as reason:
                    errors.append(error(["evaluate", "benchmarks"], str(reason)))
        except ValueError as reason:
            errors.append(error(["evaluate", "model"], str(reason)))

    # 9. The Endpoint `serve` starts: named after the Pipeline unless named, and free for now.
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


def backend_errors(request: PipelineRequest) -> list[dict]:
    """An error for each Phase whose algorithm or method its backend doesn't train."""
    finetune = request.finetune
    if finetune is None:
        return []
    backend, errors = finetune.backend, []
    for index, phase in enumerate(finetune.phases):
        methods = config.BACKENDS[backend].get(phase.algorithm)
        loc = ["finetune", "phases", index]
        if methods is None:
            msg = f"{backend} doesn't train {phase.algorithm} Phases; use backend `hf`"
            errors.append(error([*loc, "algorithm"], msg))
        elif phase.method not in methods:
            supported = ", ".join(methods)
            msg = f"{backend} trains {phase.algorithm} Phases with {supported}, not {phase.method}"
            errors.append(error([*loc, "method"], f"{msg}; use backend `hf`"))
    return errors


def trainer_config_errors(request: PipelineRequest) -> list[dict]:
    errors = []
    phases = request.finetune.phases if request.finetune else []
    for index, phase in enumerate(phases):
        loc = ["finetune", "phases", index, "lora"]
        errors += [error(loc, msg) for msg in phase_lora_errors(phases, index)]
        algorithm = config.ALGORITHMS[phase.algorithm]
        length = algorithm["length_setting"]
        if length not in phase.settings.model_dump():
            loc = ["finetune", "phases", index, "settings", length]
            errors.append(error(loc, "Field required"))
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


def reward_errors(request: PipelineRequest) -> list[dict]:
    """An error for each reward whose source can't define `reward(sample, item)`."""
    errors = []
    phases = request.finetune.phases if request.finetune else []
    for index, phase in enumerate(phases):
        for name, reward in (phase.rewards or {}).items():
            loc = ["finetune", "phases", index, "rewards", name, "source"]
            # Only parsed here, never run: reward code runs in the Sandbox alone.
            try:
                module = ast.parse(reward.source)
            except SyntaxError as reason:
                errors.append(
                    error(loc, f"`source` is no Python: {reason.msg} (line {reason.lineno})")
                )
                continue
            if not any(defines_reward(node) for node in module.body):
                errors.append(error(loc, "`source` defines no `def reward(sample, item)`"))
    return errors


def defines_reward(node: ast.stmt) -> bool:
    if not isinstance(node, ast.FunctionDef) or node.name != "reward":
        return False
    return len(node.args.posonlyargs + node.args.args) == 2


def phase_lora_errors(phases: list[Phase], index: int) -> list[str]:
    """Why the Phase's `lora` doesn't fit its method and what the Phase before it left (#19)."""
    phase = phases[index]
    continues_adapter = index > 0 and phases[index - 1].keeps_adapter
    if phase.method == "full":
        # An Adapter before it is merged into its base first.
        return (
            ["`full` trains every weight, not an Adapter; leave out `lora`"] if phase.lora else []
        )
    if continues_adapter and phase.lora:
        return [
            f"continues the Adapter of Phase {index}, keeping its rank and targets; leave out "
            f"`lora`, or set `output: merged` on Phase {index} to train a new Adapter"
        ]
    if continues_adapter:
        return []
    if phase.lora is None:
        base = f"the full weights of Phase {index}" if index else "the starting model"
        return [f"trains a new Adapter on {base}; set its `lora`"]
    # Only an Adapter kept as one must be servable by vLLM; merged, it is plain weights.
    if phase.output == "merged":
        return []
    return [
        f"{problem}; set `output: merged`" for problem in adapter_problems(phase.lora.model_dump())
    ]


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
    except ValueError as reason:
        return [*errors, error(["distill", "teacher"], str(reason))]
    if tools:
        try:
            distill.serving = pin_tool_parser(
                distill.teacher, distill.serving, token, hugging_face, engine, model_registry
            )
        except ValueError as reason:
            errors.append(error(["distill", "tools"], str(reason)))
    return errors


def assistant_only_loss_errors(
    request: PipelineRequest,
    token: str | None,
    hugging_face: HuggingFace,
    engine: Engine,
    model_registry: MLflow,
    object_store: ObjectStore,
) -> list[dict]:
    """An error for each Phase whose `assistant_only_loss` can't keep to the assistant's turns."""
    finetune = request.finetune
    # Every Phase renders with the starting model's template, so it is read at most once.
    mask_problem = None
    if finetune.backend == "hf" and any(p.settings.assistant_only_loss for p in finetune.phases):
        read = model_file_reader(
            finetune.starting_model, token, hugging_face, model_registry, object_store
        )
        mask_problem = assistant_mask_problem(finetune.starting_model, read)
    errors = []
    for index, phase in enumerate(finetune.phases):
        if not phase.settings.assistant_only_loss:
            continue
        loc = ["finetune", "phases", index, "settings", "assistant_only_loss"]
        # 1. Messages rows; prompt-completion rows keep TRL's completion-only loss (#24).
        if phase.dataset == config.DISTILL_OUTPUT:
            row_format = config.DISTILLATION_ROW_FORMAT
        else:
            pinned = find_dataset_version(engine, *split_dataset_reference(phase.dataset))
            row_format = pinned and pinned.row_format
        if row_format == "prompt_completion":
            msg = "prompt_completion rows train only on the completion already; leave it out"
            errors.append(error(loc, msg))
        elif row_format and row_format != "messages":
            errors.append(error(loc, f"is for messages rows, not {row_format} rows"))
        # 2. On `hf`, a chat template TRL can mask.
        elif mask_problem:
            errors.append(error(loc, mask_problem))
    return errors


def model_file_reader(
    model: str,
    token: str | None,
    hugging_face: HuggingFace,
    model_registry: MLflow,
    object_store: ObjectStore,
):
    """Reads one file of the pinned Base Model or Model Version, or None if it has none."""
    if model.startswith("hf:"):
        repo, commit = split_base_model_reference(model)
        return lambda path: hugging_face.model_file(repo, commit, path, token)
    found = find_referenced_model_version(model_registry, model)
    bucket = model_registry.artifact_bucket
    keys = {item["Key"] for item in object_store.list_objects(bucket, found.artifact_prefix)}
    return lambda path: (
        object_store.read(bucket, found.artifact_prefix + path)
        if found.artifact_prefix + path in keys
        else None
    )


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


def pin_tool_parser(
    model: str,
    serving: ServingOptions | None,
    token: str | None,
    hugging_face: HuggingFace,
    engine: Engine,
    model_registry: MLflow,
) -> ServingOptions | None:
    """The serving options with the model's tool parser pinned; else ValueError."""
    parser = serving and serving.tool_parser
    parser = parser or tool_parser_of(model, token, hugging_face, engine, model_registry)
    if parser is None:
        raise ValueError(f"{model} has no tool parser; name one in `serving.tool_parser`")
    # A running Endpoint serves with its own options.
    if model.startswith("endpoint:"):
        return serving
    return (serving or ServingOptions()).model_copy(update={"tool_parser": parser})


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
