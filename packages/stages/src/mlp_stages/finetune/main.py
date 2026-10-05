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

    with tempfile.TemporaryDirectory() as output_directory:
        # 3. What the Phase starts from: full weights, or an Adapter to continue or merge (#19).
        scratch = Path(output_directory)
        base, base_directory, adapter_directory = phase_start(
            starting_model, previous_model_version, scratch
        )

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
            base_directory, phase, tokenizer, dataset, output_directory, adapter_directory
        )

        # 6. Training, logged into the Run; with no size limits, out of memory must read plainly.
        try:
            trainer.train()
        except torch.OutOfMemoryError as error:
            raise SystemExit(
                f"Out of GPU memory: {error}\nTry a smaller per_device_train_batch_size or "
                "max_length, more gradient_accumulation_steps, `qlora`, or a smaller Base Model."
            ) from None

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


def phase_start(
    starting_model: str, previous_model_version: str, scratch: Path
) -> tuple[str, Path, Path | None]:
    """The base the Phase trains on, its files, and the files of the Adapter on it, if any."""
    if not previous_model_version:
        return starting_model, model_files(starting_model, scratch / "base"), None
    previous_files = model_files(previous_model_version, scratch / "previous-phase")
    name, version = split_model_reference(previous_model_version)
    tags = mlflow.MlflowClient().get_model_version(name, str(version)).tags
    if tags["weights"] != "adapter":
        return previous_model_version, previous_files, None
    base = tags["base_model"]
    return base, model_files(base, scratch / "base"), previous_files


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
