from mlp_api.pipelines.pipeline_request import trainer_config_fields
from mlp_core import config
from mlp_core.pipeline_request.schema import PhaseSettings, PipelineRequest


def published_schema() -> dict:
    """The Pipeline Request's JSON Schema, with what a client needs to build one: by algorithm,
    when its fields apply, its defaults and its settings."""
    # 1. The schema of the request, every field with its description, default and `applies_if`.
    schema = PipelineRequest.model_json_schema()

    # 2. Each algorithm: what its fields depend on, its defaults, and its TRL settings, the
    # common ones to show directly and every other allowed one.
    schema["algorithms"] = {
        name: published_algorithm(algorithm) for name, algorithm in config.ALGORITHMS.items()
    }

    # 3. The methods that train an Adapter, and the LoraConfig settings a form doesn't show
    # directly, without those of an Adapter vLLM can't serve.
    schema["adapter_methods"] = config.ADAPTER_METHODS
    hidden = (*config.BLOCKED_LORA_SETTINGS, *config.HIDDEN_LORA_SETTINGS)
    schema["more_lora_settings"] = other_setting_schemas(
        config.LORA_CONFIG, (*config.SHOWN_LORA_SETTINGS, *hidden)
    )

    # 4. Each Dataset row format: its fields and a one-line JSONL example.
    schema["row_formats"] = {
        name: {"fields": fields, "example": config.ROW_FORMAT_EXAMPLES[name]}
        for name, fields in config.ROW_FORMATS.items()
    }
    return schema


def published_algorithm(algorithm: dict) -> dict:
    """What decides a Phase of the algorithm: when its fields apply, its defaults and settings."""
    # Our own settings, e.g. `assistant_only_loss`, with our explanation.
    ours = PhaseSettings.model_json_schema()["properties"]
    common = list(config.SHOWN_SETTINGS)
    common.insert(common.index("gradient_accumulation_steps") + 1, algorithm["length_setting"])
    shown = [*common, *algorithm["shown_settings"]]
    lora = setting_schemas(
        config.LORA_CONFIG, config.SHOWN_LORA_SETTINGS, algorithm["default_lora"]
    )
    # An Adapter vLLM serves has one of the ranks it takes.
    lora["r"] = {**lora["r"], "enum": config.VLLM_LORA_RANKS}
    return {
        "learns_from_teacher": algorithm["learns_from_teacher"],
        "learns_from_rewards": algorithm["learns_from_rewards"],
        "row_formats": algorithm["row_formats"],
        "length_setting": algorithm["length_setting"],
        "default_settings": algorithm["default_settings"],
        "default_lora": algorithm["default_lora"],
        # A Sweep's, when it names none.
        "objective": {"metric": algorithm["objectives"][0], "goal": algorithm["goal"]},
        "settings": setting_schemas(
            algorithm["config"], shown, algorithm["default_settings"], ours
        ),
        "more_settings": other_setting_schemas(
            algorithm["config"], (*shown, *algorithm["blocked_settings"])
        ),
        "lora_settings": lora,
    }


def setting_schemas(config_class: str, names, defaults: dict, ours=None) -> dict:
    """Each named setting's schema in the TRL/PEFT config, with our default and explanation;
    where we set none, the config's own default is only a `placeholder`, as it applies anyway."""
    fields = trainer_config_fields(config_class) or {}
    schemas = {}
    for name in names:
        schema = {key: value for key, value in fields.get(name, {}).items() if key != "default"}
        if name in (ours or {}):
            schema["description"] = ours[name]["description"]
        if name in defaults:
            schema["default"] = defaults[name]
        elif fields.get(name, {}).get("default") is not None:
            schema["placeholder"] = fields[name]["default"]
        schemas[name] = schema
    return schemas


def other_setting_schemas(config_class: str, left_out) -> dict:
    """The schema of every setting of the config but those left out, TRL's defaults as
    placeholders."""
    fields = trainer_config_fields(config_class) or {}
    others = [name for name in fields if name not in left_out]
    return setting_schemas(config_class, others, {})
