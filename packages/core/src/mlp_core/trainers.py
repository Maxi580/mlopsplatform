import json
from functools import cache

from mlp_core import config

# Phase algorithm -> its TRL config class.
ALGORITHMS = {"sft": "SFTConfig"}
LORA_CONFIG = "LoraConfig"
# Training backend -> the weight methods it supports.
BACKENDS = {"hf": ("lora",)}
METHODS = sorted({method for methods in BACKENDS.values() for method in methods})


@cache
def trainer_config_schema(config_class: str) -> dict | None:
    """JSON Schema of a TRL/PEFT config class; None if none was generated, so it goes unchecked."""
    path = config.TRAINER_CONFIGS_DIRECTORY / f"{config_class}.json"
    return json.loads(path.read_text()) if path.exists() else None
