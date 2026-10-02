from pathlib import Path

# JSON Schemas of the TRL/PEFT config classes, written by scripts/generate_trainer_configs.py.
TRAINER_CONFIGS_DIRECTORY = Path(__file__).parent / "pipeline_request" / "trainer_configs"
# Phase algorithm -> its TRL config class.
ALGORITHMS = {"sft": "SFTConfig"}
LORA_CONFIG = "LoraConfig"
# Training backend -> the weight methods it supports.
BACKENDS = {"hf": ("lora",)}
