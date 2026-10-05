from pathlib import Path

import trl
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from mlp_core import config
from mlp_core.pipeline_request.schema import Phase


def load_tokenizer(model_directory: Path):
    """The starting model's tokenizer and chat template."""
    return AutoTokenizer.from_pretrained(model_directory)


def build_trainer(
    model_directory: Path,
    phase: Phase,
    tokenizer,
    dataset,
    output_directory: str,
    adapter_directory: Path | None,
):
    """The catalog row's TRL trainer, training a fresh LoRA Adapter or continuing the one given."""
    algorithm = config.ALGORITHMS[phase.algorithm]
    # At their saved precision (usually bfloat16).
    model = AutoModelForCausalLM.from_pretrained(model_directory, dtype="auto")
    peft_config = None
    if adapter_directory:
        # Keeps its rank and targets; DPO and KTO take a frozen copy of it as their reference.
        model = PeftModel.from_pretrained(model, adapter_directory, is_trainable=True)
    else:
        peft_config = LoraConfig(**{**config.LORA_DEFAULTS, **phase.lora.model_dump()})
    return getattr(trl, algorithm["trainer"])(
        model=model,
        args=getattr(trl, algorithm["config"])(
            **{**algorithm["defaults"], **phase.settings.model_dump()},
            output_dir=output_directory,
        ),
        train_dataset=dataset,
        processing_class=tokenizer,
        # TRL freezes the starting model's weights and trains only the Adapter.
        peft_config=peft_config,
    )


def save_adapter(trainer, model_directory: Path) -> None:
    """Saves the trained Adapter with the tokenizer, without the reference copy DPO and KTO add."""
    if "ref" in trainer.model.peft_config:
        trainer.model.delete_adapter("ref")
    trainer.save_model(str(model_directory))
