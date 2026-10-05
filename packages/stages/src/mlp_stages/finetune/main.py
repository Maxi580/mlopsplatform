import importlib
import os
import tempfile
from pathlib import Path

import mlflow
import torch
from datasets import Dataset
from huggingface_hub import snapshot_download

from mlp_core import config
from mlp_core.pipeline_request.references import (
    checkpoint_prefix,
    split_base_model_reference,
    split_model_reference,
)
from mlp_core.pipeline_request.schema import PipelineRequest
from mlp_stages.dataset_versions import download_dataset_version
from mlp_stages.finetune.model_version import register_model_version


def finetune(
    pipeline_id: str,
    phase_index: str,
    request: str,
    distilled_dataset: str,
    previous_model_version: str,
    checkpoint_minutes: str,
    resume_checkpoint: str,
    stop_after_checkpoint: str,
    model_version: str,
) -> None:
    """Trains one Phase and registers its output as a Model Version, written to `model_version`."""
    # 1. The Phase to train, from the resolved request the compiler passed in.
    resolved = PipelineRequest.model_validate_json(request)
    index = int(phase_index)
    phase = resolved.finetune.phases[index]
    # `@distill` is the Dataset Version the `distill` step registered, which KFP hands over.
    if phase.dataset == config.DISTILL_OUTPUT:
        phase.dataset = distilled_dataset
    starting_model = resolved.finetune.starting_model
    # Read now, as MLflow removes it from the environment once the trainer resumes the Run.
    run_id = os.environ["MLFLOW_RUN_ID"]

    # 2. The backend, imported by name before any training library: Unsloth patches them on import.
    # Only backends import transformers, TRL and PEFT; each has load_tokenizer, build_trainer and
    # save_model_version.
    backend = importlib.import_module(f"mlp_stages.finetune.backends.{resolved.finetune.backend}")
    # Imports transformers, so only after the backend.
    from mlp_stages.finetune import checkpoints

    with tempfile.TemporaryDirectory() as output_directory:
        # 3. What the Phase starts from (#19), and the Teacher a `distillation` Phase loads too.
        scratch = Path(output_directory)
        base, base_directory, adapter_directory = model_with_adapter(
            previous_model_version or starting_model, scratch / "start"
        )
        teacher = None
        if phase.teacher:
            _, *teacher = model_with_adapter(phase.teacher, scratch / "teacher")

        # 4. The Dataset; conversations only ever use the starting model's own chat template (#16),
        # which every Model Version carries.
        tokenizer = backend.load_tokenizer(adapter_directory or base_directory)
        path = download_dataset_version(phase.dataset, scratch)
        dataset = Dataset.from_json(str(path), keep_in_memory=True)
        if is_conversation(dataset[0]) and tokenizer.chat_template is None:
            raise SystemExit(
                f"{starting_model} has no chat template, so it can't train on the conversations "
                f"in {phase.dataset}; pick a model with one, or a text Dataset"
            )

        # 5. The trainer for the Phase's weight method, with its settings over defaults.
        trainer = backend.build_trainer(
            base_directory, phase, tokenizer, dataset, output_directory, adapter_directory, teacher
        )

        # 6. Training, logged into the Run, from the Checkpoint a resume hands over; a Checkpoint is
        # uploaded every checkpoint_minutes. With no size limits, out of memory must read plainly.
        checkpoint = None
        if resume_checkpoint:
            checkpoint = checkpoints.download_checkpoint(resume_checkpoint, scratch / "resume")
        own_checkpoints = checkpoint_prefix(pipeline_id, index)
        trainer.add_callback(
            checkpoints.CheckpointUploads(
                own_checkpoints, int(checkpoint_minutes), bool(stop_after_checkpoint)
            )
        )
        try:
            trainer.train(resume_from_checkpoint=checkpoint and str(checkpoint))
        except torch.OutOfMemoryError as error:
            raise SystemExit(
                f"Out of GPU memory: {error}\nTry a smaller per_device_train_batch_size or "
                "length setting, more gradient_accumulation_steps, `qlora`, a smaller Base Model "
                "or a smaller Teacher; for grpo and rloo, fewer num_generations or another "
                "vllm_gpu_memory_utilization."
            ) from None
        # The Smoke Test's interrupted step leaves its Checkpoint and registers nothing.
        if stop_after_checkpoint:
            Path(model_version).write_text("")
            return

        # 7. The output with the tokenizer and chat template, registered as the next Model Version,
        # which the next Phase starts from.
        model_directory = scratch / "model"
        backend.save_model_version(trainer, phase, tokenizer, model_directory, base_directory)
        model_type = trainer.model.config.model_type
        parent = previous_model_version or starting_model
        registered = register_model_version(
            model_directory,
            run_id,
            resolved,
            pipeline_id,
            index,
            parent,
            base,
            tokenizer,
            model_type,
        )
        Path(model_version).write_text(registered)

        # 8. The Phase succeeded, so its Checkpoints are no longer needed.
        checkpoints.delete_checkpoints(own_checkpoints)


def model_with_adapter(reference: str, scratch: Path) -> tuple[str, Path, Path | None]:
    """The model's base, the base's files, and the files of the model's Adapter, if it is one."""
    files = model_files(reference, scratch / "model")
    if reference.startswith("hf:"):
        return reference, files, None
    name, version = split_model_reference(reference)
    tags = mlflow.MlflowClient().get_model_version(name, str(version)).tags
    if tags["weights"] != "adapter":
        return reference, files, None
    base = tags["base_model"]
    return base, model_files(base, scratch / "base"), files


def model_files(reference: str, destination: Path) -> Path:
    """Where the `hf:` or `model:` Reference's files lie: the Model Cache, or a download."""
    if reference.startswith("hf:"):
        repo, commit = split_base_model_reference(reference)
        # fetch put it in the Model Cache; offline, this only finds it there.
        return Path(snapshot_download(repo, revision=commit))
    name, version = split_model_reference(reference)
    uri = f"models:/{name}/{version}"
    return Path(mlflow.artifacts.download_artifacts(uri, dst_path=str(destination)))


# Rows of messages, which TRL renders with the chat template; TRL may only load after the backend.
def is_conversation(row: dict) -> bool:
    return any(
        isinstance(value, list) and value and isinstance(value[0], dict) and "role" in value[0]
        for value in row.values()
    )
