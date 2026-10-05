from pathlib import Path

import trl
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

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
    teacher: list[Path | None] | None,
):
    """The catalog row's TRL trainer for the Phase's weight method, on the base with the Adapter."""
    algorithm = config.ALGORITHMS[phase.algorithm]
    # Only `distillation` has a Teacher: its base files, and its Adapter files to merge in, if any.
    trainer_kwargs = {"teacher_model": load_model(*teacher, "auto")} if teacher else {}
    peft_config = None
    if phase.method == "full":
        # Trains every weight; an Adapter before it is merged into its base first (#19).
        model = load_model(model_directory, adapter_directory, "auto")
    else:
        # At their saved precision (usually bfloat16), or in 4 bits for `qlora`.
        qlora = phase.method == "qlora"
        quantization = BitsAndBytesConfig(**config.QLORA_QUANTIZATION) if qlora else None
        model = AutoModelForCausalLM.from_pretrained(
            model_directory, dtype="auto", quantization_config=quantization
        )
        if adapter_directory:
            # TRL prepares a 4-bit model only for a new Adapter.
            if qlora:
                model = prepare_model_for_kbit_training(model)
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
        # With an Adapter, TRL freezes the base's weights and trains only the Adapter.
        peft_config=peft_config,
        **trainer_kwargs,
    )


def load_model(model_directory: Path, adapter_directory: Path | None, dtype: str):
    """The model in `dtype`, with the Adapter merged into it if one is given."""
    model = AutoModelForCausalLM.from_pretrained(model_directory, dtype=dtype)
    if adapter_directory is None:
        return model
    return PeftModel.from_pretrained(model, adapter_directory).merge_and_unload()


def save_model_version(
    trainer, phase: Phase, tokenizer, model_directory: Path, base_directory: Path
) -> None:
    """Saves the Phase's output: full weights, the Adapter, or the Adapter merged into its base."""
    # 1. Without the reference copy DPO and KTO add to a continued Adapter.
    if "ref" in getattr(trainer.model, "peft_config", {}):
        trainer.model.delete_adapter("ref")
    if not phase.merges_adapter:
        trainer.save_model(str(model_directory))
        return

    # 2. Merged into the base reloaded in bfloat16, not the 4 bits `qlora` trained on.
    adapter_directory = model_directory.with_name("trained-adapter")
    trainer.save_model(str(adapter_directory))
    load_model(base_directory, adapter_directory, "bfloat16").save_pretrained(model_directory)
    tokenizer.save_pretrained(model_directory)
