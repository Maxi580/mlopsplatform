from pathlib import Path

import trl
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

from mlp_core import config
from mlp_core.pipeline_request.schema import Phase


def load_tokenizer(model_directory: Path):
    """The starting model's tokenizer and chat template."""
    return AutoTokenizer.from_pretrained(model_directory)


def build_trainer(model_directory: Path, phase: Phase, tokenizer, dataset, output_directory: str):
    """The catalog row's TRL trainer, training a fresh LoRA Adapter on the starting model."""
    algorithm = config.ALGORITHMS[phase.algorithm]
    # At their saved precision (usually bfloat16).
    model = AutoModelForCausalLM.from_pretrained(model_directory, dtype="auto")
    return getattr(trl, algorithm["trainer"])(
        model=model,
        args=getattr(trl, algorithm["config"])(
            **{**algorithm["defaults"], **phase.settings.model_dump()},
            output_dir=output_directory,
        ),
        train_dataset=dataset,
        processing_class=tokenizer,
        # TRL freezes the Base Model's weights and trains only this Adapter.
        peft_config=LoraConfig(**{**config.LORA_DEFAULTS, **phase.lora.model_dump()}),
    )
