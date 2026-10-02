import importlib
import os
import tempfile
from pathlib import Path

import boto3
import torch
from datasets import Dataset

from mlp_core.pipeline_request.references import dataset_key, split_dataset_reference
from mlp_core.pipeline_request.schema import PipelineRequest
from mlp_stages.finetune.model_version import register_model_version


def finetune(pipeline_id: str, phase_index: str, request: str) -> None:
    """Trains one Phase of the resolved request and registers its Adapter as a Model Version."""
    # 1. The Phase to train, from the resolved request the compiler passed in.
    resolved = PipelineRequest.model_validate_json(request)
    index = int(phase_index)
    phase = resolved.finetune.phases[index]
    base_model = resolved.finetune.base_model
    # Read now, as MLflow removes it from the environment once the trainer resumes the Run.
    run_id = os.environ["MLFLOW_RUN_ID"]

    # 2. The backend, imported by name before any training library: Unsloth patches them on import.
    # Only backends import transformers, TRL and PEFT; each has load_tokenizer and build_trainer.
    backend = importlib.import_module(f"mlp_stages.finetune.backends.{resolved.finetune.backend}")

    # 3. The Dataset; conversations only ever use the Base Model's own chat template (#16).
    tokenizer = backend.load_tokenizer(base_model)
    dataset = load_dataset_version(phase.dataset)
    if is_conversation(dataset[0]) and tokenizer.chat_template is None:
        raise SystemExit(
            f"{base_model} has no chat template, so it can't train on the conversations in "
            f"{phase.dataset}; pick a Base Model with one, or a text Dataset"
        )

    with tempfile.TemporaryDirectory() as output_directory:
        # 4. The trainer: the Base Model with a fresh Adapter, the Phase's settings over defaults.
        trainer = backend.build_trainer(base_model, phase, tokenizer, dataset, output_directory)

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
