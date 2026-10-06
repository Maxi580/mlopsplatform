from mlp_api.pipelines.pipeline_request import trainer_config_fields
from mlp_core import config
from mlp_core.pipeline_request.schema import PhaseSettings, PipelineRequest


def published_schema() -> dict:
    """The Pipeline Request's JSON Schema, with what a client needs to build one: by algorithm,
    when its fields apply, its defaults and its settings."""
    schema = PipelineRequest.model_json_schema()
    schema["algorithms"] = {
        name: published_algorithm(algorithm) for name, algorithm in config.ALGORITHMS.items()
    }
    schema["adapter_methods"] = config.ADAPTER_METHODS
    return schema


def published_algorithm(algorithm: dict) -> dict:
    """What decides a Phase of the algorithm: when its fields apply, its defaults and settings."""
    # Our own settings, e.g. `assistant_only_loss`, with our explanation, where its config has them.
    ours = PhaseSettings.model_json_schema()["properties"]
    fields = trainer_config_fields(algorithm["config"]) or {}
    shown = [*algorithm["default_settings"], *(name for name in ours if name in fields)]
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
        "lora_settings": setting_schemas(
            config.LORA_CONFIG, list(algorithm["default_lora"]), algorithm["default_lora"]
        ),
    }


def setting_schemas(config_class: str, names: list[str], defaults: dict, ours=None) -> dict:
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
