from pathlib import Path

# JSON Schemas of the TRL/PEFT config classes, written by generate_trainer_configs.py.
TRAINER_CONFIGS_DIRECTORY = Path(__file__).parent / "pipeline_request" / "trainer_configs"
# Training backend -> the weight methods it supports.
BACKENDS = {"hf": ("lora",)}
