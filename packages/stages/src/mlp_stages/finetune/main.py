import importlib
import os
import tempfile
from pathlib import Path

import boto3
import mlflow
import torch
from datasets import Dataset
from huggingface_hub import snapshot_download

from mlp_core.pipeline_request.references import (
    dataset_key,
    split_base_model_reference,
    split_dataset_reference,
    split_model_reference,
)
from mlp_core.pipeline_request.schema import Finetune, PipelineRequest
from mlp_stages.finetune.model_version import register_model_version


def finetune(pipeline_id: str, phase_index: str, request: str) -> None:
    """Trains one Phase of the resolved request and registers its Adapter as a Model Version."""
    # 1. The Phase to train, from the resolved request the compiler passed in.
    resolved = PipelineRequest.model_validate_json(request)
    index = int(phase_index)
    phase = resolved.finetune.phases[index]
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
        dataset = load_dataset_version(phase.dataset)
        if is_conversation(dataset[0]) and tokenizer.chat_template is None:
            raise SystemExit(
                f"{starting_model} has no chat template, so it can't train on the conversations "
                f"in {phase.dataset}; pick a model with one, or a text Dataset"
            )

        # 4. The trainer: the starting model, a fresh Adapter, the Phase's settings over defaults.
        trainer = backend.build_trainer(
            starting_directory, phase, tokenizer, dataset, output_directory
        )

        # 5. Training, logged into the Run; with no size limits, out of memory must read plainly.
        try:
            trainer.train()
        except torch.OutOfMemoryError as error:
            raise SystemExit(
                f"Out of GPU memory: {error}\nTry a smaller per_device_train_batch_size or "
                "max_length, more gradient_accumulation_steps, or a smaller Base Model."
            ) from None

        # 6. The Adapter with the tokenizer and chat template, registered as the next Model Version.
        model_directory = Path(output_directory) / "model"
        trainer.save_model(str(model_directory))
        model_type = trainer.model.config.model_type
        register_model_version(
            model_directory, run_id, resolved, pipeline_id, index, tokenizer, model_type
        )


def starting_model_directory(finetune: Finetune, scratch: Path) -> Path:
    """Where the starting model's files lie: the Model Cache, or a download of the Model Version."""
    if finetune.base_model:
        repo, commit = split_base_model_reference(finetune.base_model)
        # fetch put it in the Model Cache; offline, this only finds it there.
        return Path(snapshot_download(repo, revision=commit))
    name, version = split_model_reference(finetune.from_)
    uri = f"models:/{name}/{version}"
    return Path(mlflow.artifacts.download_artifacts(uri, dst_path=str(scratch / "starting-model")))


def load_dataset_version(reference: str) -> Dataset:
    """The Dataset Version's rows, from the platform bucket."""
    name, version = split_dataset_reference(reference)
    object_store = boto3.client("s3", endpoint_url=os.environ["S3_ENDPOINT_URL"])
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "data.jsonl")
        object_store.download_file(os.environ["S3_BUCKET"], dataset_key(name, version), path)
        return Dataset.from_json(path, keep_in_memory=True)


# Rows of messages, which TRL renders with the chat template; TRL may only load after the backend.
def is_conversation(row: dict) -> bool:
    return any(
        isinstance(value, list) and value and isinstance(value[0], dict) and "role" in value[0]
        for value in row.values()
    )
