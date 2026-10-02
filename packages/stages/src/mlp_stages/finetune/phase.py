import os
import tempfile
from pathlib import Path

import boto3
import torch
import trl
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl.data_utils import is_conversational

from mlp_core import config
from mlp_core.pipeline_request.references import (
    dataset_key,
    split_base_model_reference,
    split_dataset_reference,
)
from mlp_core.pipeline_request.schema import PipelineRequest
from mlp_stages.finetune.model_version import register_model_version


def finetune(pipeline_id: str, phase_index: str, request: str) -> None:
    """Trains one Phase of the resolved request and registers its output as a Model Version."""
    # 1. The Phase, its algorithm's catalog row and the Base Model's tokenizer, offline.
    # KFP's MLflow plugin made the step's Run; MLflow removes its ID once the trainer resumes it.
    run_id = os.environ["MLFLOW_RUN_ID"]
    resolved = PipelineRequest.model_validate_json(request)
    phase = resolved.finetune.phases[int(phase_index)]
    algorithm = config.ALGORITHMS[phase.algorithm]
    repo, commit = split_base_model_reference(resolved.finetune.base_model)
    tokenizer = AutoTokenizer.from_pretrained(repo, revision=commit)

    # 2. The Dataset; conversations need the Base Model's own chat template, never another.
    dataset = load_dataset_version(phase.dataset)
    if is_conversational(dataset[0]) and tokenizer.chat_template is None:
        raise SystemExit(
            f"{repo} has no chat template, so it can't train on the conversations in "
            f"{phase.dataset}; pick a Base Model with one, or a text Dataset"
        )

    with tempfile.TemporaryDirectory() as output_directory:
        # 3. The trainer from the catalog row, with the user's settings over the platform's.
        trainer = getattr(trl, algorithm["trainer"])(
            model=AutoModelForCausalLM.from_pretrained(repo, revision=commit, dtype="auto"),
            args=getattr(trl, algorithm["config"])(
                **{**algorithm["defaults"], **phase.settings.model_dump()},
                output_dir=output_directory,
            ),
            train_dataset=dataset,
            processing_class=tokenizer,
            peft_config=LoraConfig(**{**config.LORA_DEFAULTS, **phase.lora.model_dump()}),
        )

        # 4. Training; running out of memory is the user's to fix, so it reads plainly.
        try:
            trainer.train()
        except torch.OutOfMemoryError as error:
            raise SystemExit(
                f"Out of GPU memory: {error}\nTry a smaller per_device_train_batch_size or "
                "max_length, more gradient_accumulation_steps, or a smaller Base Model."
            ) from None

        # 5. The Adapter with the Base Model's tokenizer and chat template, registered.
        model_directory = Path(output_directory) / "model"
        trainer.save_model(str(model_directory))
        model_type = trainer.model.config.model_type
        register_model_version(
            model_directory, run_id, resolved, pipeline_id, int(phase_index), tokenizer, model_type
        )


def load_dataset_version(reference: str) -> Dataset:
    """The Dataset Version's rows, from the platform bucket."""
    name, version = split_dataset_reference(reference)
    object_store = boto3.client("s3", endpoint_url=os.environ["S3_ENDPOINT_URL"])
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "data.jsonl")
        object_store.download_file(os.environ["S3_BUCKET"], dataset_key(name, version), path)
        return Dataset.from_json(path, keep_in_memory=True)
