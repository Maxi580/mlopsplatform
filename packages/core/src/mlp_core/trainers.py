import json
from fnmatch import fnmatchcase
from functools import cache

from mlp_core import config

# Phase algorithm -> its TRL config class.
ALGORITHMS = {"sft": "SFTConfig"}
LORA_CONFIG = "LoraConfig"
# Training backend -> the weight methods it supports.
BACKENDS = {"hf": ("lora",)}
METHODS = sorted({method for methods in BACKENDS.values() for method in methods})

# Fields the platform owns; `*` matches any suffix.
DENIED_SETTINGS = (
    "output_dir",
    "logging_dir",
    "report_to",
    "push_to_hub",
    "hub_*",
    "save_strategy",
    "save_steps",
    "save_total_limit",
    "resume_from_checkpoint",
    "vllm_mode",
    "vllm_server_*",
)
# Adapter options vLLM can't serve.
DENIED_LORA = ("use_dora", "modules_to_save", "bias")


@cache
def trainer_config_schema(config_class: str) -> dict | None:
    """JSON Schema of a TRL/PEFT config class; None if none was generated, so it goes unchecked."""
    path = config.TRAINER_CONFIGS_DIRECTORY / f"{config_class}.json"
    return json.loads(path.read_text()) if path.exists() else None


def is_denied(name: str, denied: tuple[str, ...]) -> bool:
    return any(fnmatchcase(name, pattern) for pattern in denied)
