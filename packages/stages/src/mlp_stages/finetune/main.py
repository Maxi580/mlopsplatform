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
from mlp_core.pipeline_request.schema import Finetune, PipelineRequest
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
    """Trains one Phase and registers its Adapter as a Model Version, written to `model_version`."""
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
    # Only backends import transformers, TRL and PEFT; each has load_tokenizer and build_trainer.
    backend = importlib.import_module(f"mlp_stages.finetune.backends.{resolved.finetune.backend}")

    with tempfile.TemporaryDirectory() as output_directory:
        # 3. The Dataset; conversations only ever use the starting model's own chat template (#16).
        starting_directory = starting_model_directory(resolved.finetune, Path(output_directory))
        tokenizer = backend.load_tokenizer(starting_directory)
        path = download_dataset_version(phase.dataset, Path(output_directory))
        dataset = Dataset.from_json(str(path), keep_in_memory=True)
        if is_conversation(dataset[0]) and tokenizer.chat_template is None:
            raise SystemExit(
                f"{starting_model} has no chat template, so it can't train on the conversations "
                f"in {phase.dataset}; pick a model with one, or a text Dataset"
            )

        # 4. The trainer: the starting model with a fresh Adapter, or with the previous Phase's to
        # continue (#19), and the Phase's settings over defaults.
        adapter_directory = None
        if previous_model_version:
            scratch = Path(output_directory) / "previous-phase"
            adapter_directory = model_version_directory(previous_model_version, scratch)
        trainer = backend.build_trainer(
            starting_directory, phase, tokenizer, dataset, output_directory, adapter_directory
        )

        # 5. Training, logged into the Run; with no size limits, out of memory must read plainly.
        try:
            trainer.train()
        except torch.OutOfMemoryError as error:
            raise SystemExit(
                f"Out of GPU memory: {error}\nTry a smaller per_device_train_batch_size or "
                "max_length, more gradient_accumulation_steps, or a smaller Base Model."
            ) from None

        # 6. The Adapter with the tokenizer and chat template, registered as the next Model Version,
        # which the next Phase starts from.
        model_directory = Path(output_directory) / "model"
        backend.save_adapter(trainer, model_directory)
        model_type = trainer.model.config.model_type
        parent = previous_model_version or starting_model
        registered = register_model_version(
            model_directory, run_id, resolved, pipeline_id, index, parent, tokenizer, model_type
        )
        Path(model_version).write_text(registered)


def starting_model_directory(finetune: Finetune, scratch: Path) -> Path:
    """Where the starting model's files lie: the Model Cache, or a download of the Model Version."""
    if finetune.base_model:
        repo, commit = split_base_model_reference(finetune.base_model)
        # fetch put it in the Model Cache; offline, this only finds it there.
        return Path(snapshot_download(repo, revision=commit))
    return model_version_directory(finetune.from_, scratch / "starting-model")


def model_version_directory(reference: str, destination: Path) -> Path:
    """The `model:` Reference's files, downloaded from the Model Registry."""
    name, version = split_model_reference(reference)
    uri = f"models:/{name}/{version}"
    return Path(mlflow.artifacts.download_artifacts(uri, dst_path=str(destination)))


# Rows of messages, which TRL renders with the chat template; TRL may only load after the backend.
def is_conversation(row: dict) -> bool:
    return any(
        isinstance(value, list) and value and isinstance(value[0], dict) and "role" in value[0]
        for value in row.values()
    )
