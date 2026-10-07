# Patches transformers, TRL and PEFT, so it is imported before them.
from unsloth import FastLanguageModel
from unsloth.chat_templates import train_on_responses_only

# isort: split
from pathlib import Path

import trl
from peft import PeftConfig, PeftModel

from mlp_core import config
from mlp_core.pipeline_request.schema import Phase
from mlp_stages.finetune.backends import hf

load_tokenizer = hf.load_tokenizer


def build_trainer(
    model_directory: Path,
    phase: Phase,
    tokenizer,
    dataset,
    output_directory: str,
    adapter_directory: Path | None,
    teacher: list[Path | None] | None,
    eval_dataset=None,
):
    """The catalog row's TRL trainer, patched by Unsloth, for the Phase's weight method."""
    # 1. The settings; Unsloth masks the user's turns itself, so TRL never looks for markers.
    algorithm = config.ALGORITHMS[phase.algorithm]
    settings, trainer_kwargs = hf.trainer_settings(phase)
    assistant_only_loss = settings.pop("assistant_only_loss", False)

    # 2. Unsloth loads files, so an Adapter before a `full` Phase is merged on disk first (#19).
    full = phase.method == "full"
    if full and adapter_directory:
        merged = Path(output_directory) / "merged-start"
        hf.load_model(model_directory, adapter_directory, "auto").save_pretrained(merged)
        tokenizer.save_pretrained(merged)
        model_directory, adapter_directory = merged, None

    # 3. The model: exactly the pinned files, in 4 bits only for `qlora`; online RL generates on
    # Unsloth's vLLM, which shares the Adapter's weights and needs its rank up front.
    rl = algorithm["learns_from_rewards"]
    vllm = {}
    if rl:
        rank = phase.lora.r if phase.lora else PeftConfig.from_pretrained(adapter_directory).r
        vllm = {"fast_inference": True, "max_lora_rank": rank}
        if "vllm_gpu_memory_utilization" in settings:
            vllm["gpu_memory_utilization"] = settings["vllm_gpu_memory_utilization"]
    model, _ = FastLanguageModel.from_pretrained(
        str(model_directory),
        load_in_4bit=phase.method == "qlora",
        full_finetuning=full,
        use_exact_model_name=True,
        **vllm,
    )

    # 4. The Adapter: the one before it continued as it is, or a new one.
    if adapter_directory:
        model = PeftModel.from_pretrained(model, adapter_directory, is_trainable=True)
        model = FastLanguageModel.patch_peft_model(model)
    elif not full:
        lora = phase.lora.model_dump()
        # Unsloth reads a string as a list of names; its default targets are the linear layers.
        if lora["target_modules"] == "all-linear":
            del lora["target_modules"]
        model = FastLanguageModel.get_peft_model(model, **lora)

    # 5. The trainer, with only the assistant's turns in the loss if asked.
    trainer = getattr(trl, algorithm["trainer"])(
        model=model,
        args=getattr(trl, algorithm["config"])(**settings, output_dir=output_directory),
        train_dataset=rendered_conversations(dataset, tokenizer),
        # A Sweep's Trials are measured on held-out rows.
        eval_dataset=rendered_conversations(eval_dataset, tokenizer),
        processing_class=tokenizer,
        **trainer_kwargs,
    )
    if assistant_only_loss:
        trainer = train_on_responses_only(trainer)
    return trainer


# Unsloth's SFT tokenizes a `text` column but, unlike TRL, can't render `messages` itself.
def rendered_conversations(dataset, tokenizer):
    if dataset is None or "messages" not in dataset.column_names:
        return dataset
    return dataset.map(
        lambda row: {"text": tokenizer.apply_chat_template(row["messages"], tokenize=False)},
        remove_columns=["messages"],
    )


def save_model_version(
    trainer, phase: Phase, tokenizer, model_directory: Path, base_directory: Path
) -> None:
    """Saves the Phase's output: full weights, the Adapter, or the Adapter merged into its base."""
    # 1. Merged in 16 bits, not the 4 bits `qlora` trained on.
    if phase.merges_adapter:
        trainer.model.save_pretrained_merged(
            str(model_directory), tokenizer, save_method="merged_16bit"
        )
        return

    # 2. Unsloth moves embedding targets into modules_to_save, which vLLM can't serve.
    saved_modules = phase.keeps_adapter and trainer.model.peft_config["default"].modules_to_save
    if saved_modules:
        raise SystemExit(
            f"Unsloth trained {', '.join(saved_modules)} as full weights, which vLLM can't serve "
            "in an Adapter; leave embed_tokens and lm_head out of `target_modules`, or set "
            "`output: merged`"
        )
    trainer.save_model(str(model_directory))
