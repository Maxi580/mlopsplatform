import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import mlflow

from mlp_core import config
from mlp_core.endpoint_spec import ServingOptions, VllmModel, vllm_args
from mlp_core.pipeline_request.schema import PipelineRequest, Speculate
from mlp_stages.dataset_versions import download_dataset_version
from mlp_stages.model_versions import model_files, register_model_version
from mlp_stages.served_model import running_vllm


def speculate(
    pipeline_id: str,
    request: str,
    finetuned: str,
    quantized: str,
    distilled_dataset: str,
    gpus: str,
    dataloader_workers: str,
    model_version: str,
) -> None:
    """Trains a Speculator for the verifier and registers it, its Reference in `model_version`."""
    # 1. The block of the resolved request; `@…` are the outputs KFP handed over.
    resolved = PipelineRequest.model_validate_json(request)
    block = resolved.speculate
    handed = {
        config.FINETUNE_OUTPUT: finetuned,
        config.QUANTIZE_OUTPUT: quantized,
        config.DISTILL_OUTPUT: distilled_dataset,
    }
    verifier = handed.get(block.model, block.model)
    run_id = os.environ["MLFLOW_RUN_ID"]
    params = {"speculator": block.speculator, "verifier": verifier, **block.settings.model_dump()}
    for key, value in params.items():
        mlflow.MlflowClient().log_param(run_id, key, value)
    os.environ.update(config.SPECULATE_ENVIRONMENT)

    with tempfile.TemporaryDirectory() as directory:
        # 2. The verifier's files, and the conversations as `speculators` reads them.
        scratch = Path(directory)
        source = model_files(verifier, scratch / "verifier")
        conversations = conversations_file(handed.get(block.dataset, block.dataset), scratch)

        # 3. The verifier's hidden states for every conversation, from vLLM serving them instead
        # of answering; under its path, which the tools name in every request.
        serving = ServingOptions(prefix_caching=False)
        args = vllm_args(serving, VllmModel(str(source)), str(source), int(gpus))
        extracted = scratch / "extracted"
        extracted.mkdir()
        with running_vllm(args + extraction_args(source, extracted)):
            prepare_data(block, source, conversations, scratch)
            generate_offline_data(block, source, scratch)

        # 4. The drafter, trained on the GPU vLLM freed.
        train(block, source, int(dataloader_workers), scratch)

        # 5. The best epoch, without the optimizer state no one resumes from, naming its verifier
        # as the platform does, which an Endpoint matches it on.
        best = best_checkpoint(scratch / "speculator")
        for state in best.glob("*.pt"):
            state.unlink()
        name_verifier(best / "config.json", verifier)

        # 6. The next version of the Pipeline's Speculators.
        tags = {
            "weights": "speculator",
            "speculator": block.speculator,
            "verifier": verifier,
            "pipeline": pipeline_id,
        }
        name = config.SPECULATOR_MODEL_NAME.format(pipeline=resolved.name)
        Path(model_version).write_text(register_model_version(best, run_id, resolved, tags, name))


def conversations_file(dataset: str, scratch: Path) -> Path:
    """The Dataset's rows as `speculators` reads them: whole conversations, as `conversations`."""
    path = scratch / "conversations.jsonl"
    with path.open("w") as file:
        for line in download_dataset_version(dataset, scratch).read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            # `distill` registers each prompt and reply as a prompt-completion row of messages.
            turns = row.get("messages") or row["prompt"] + row["completion"]
            file.write(json.dumps({"conversations": turns}) + "\n")
    return path


def extraction_args(source: Path, extracted: Path) -> list[str]:
    """vLLM flags that write the prompt's hidden states to `extracted` instead of answering."""
    # An early, a middle and a late layer, and the final hidden state, by the verifier's depth.
    depth = json.loads((source / "config.json").read_text())["num_hidden_layers"]
    layers = [3, depth // 2, depth - 3, depth]
    speculative = {
        "method": "extract_hidden_states",
        "num_speculative_tokens": 1,
        "draft_model_config": {"hf_config": {"eagle_aux_hidden_state_layer_ids": layers}},
    }
    kv_transfer = {
        "kv_connector": "ExampleHiddenStatesConnector",
        "kv_role": "kv_producer",
        "kv_connector_extra_config": {"shared_storage_path": str(extracted)},
    }
    return [
        "--speculative-config",
        json.dumps(speculative),
        "--kv-transfer-config",
        json.dumps(kv_transfer),
    ]


def prepare_data(block: Speculate, source: Path, conversations: Path, scratch: Path) -> None:
    """Renders the conversations through the verifier's chat template and tokenizes them."""
    run_speculators(
        scratch,
        "speculators",
        "prepare-data",
        *("--model", source, "--data", conversations, "--output", scratch / "data"),
        *("--render-endpoint", config.LOCAL_VLLM_URL),
        *("--seq-length", block.settings.seq_length, "--max-samples", block.settings.samples),
        "--overwrite",
    )


# A sample that fails after its retries ends the step, rather than leaving a hole training skips.
def generate_offline_data(block: Speculate, source: Path, scratch: Path) -> None:
    """Collects every prepared conversation's hidden states from vLLM onto the disk."""
    run_speculators(
        scratch,
        "speculators",
        "generate-offline-data",
        *("--model", source, "--endpoint", f"{config.LOCAL_VLLM_URL}/v1"),
        *("--preprocessed-data", scratch / "data", "--output", scratch / "hidden_states"),
        *("--max-samples", block.settings.samples),
        "--fail-on-error",
    )


# vLLM is gone by now, so a sample without hidden states is an error, not one to fetch.
def train(block: Speculate, source: Path, dataloader_workers: int, scratch: Path) -> None:
    """Trains the drafter on the hidden states, saving each epoch."""
    settings = block.settings
    run_speculators(
        scratch,
        sys.executable,
        *("-m", "speculators.train", "--verifier-name-or-path", source),
        *("--data-path", scratch / "data", "--hidden-states-path", scratch / "hidden_states"),
        *("--save-path", scratch / "speculator", "--speculator-type", block.speculator),
        *("--epochs", settings.epochs, "--lr", settings.learning_rate),
        *("--total-seq-len", settings.seq_length, "--draft-vocab-size", settings.draft_vocab_size),
        *("--num-workers", dataloader_workers, "--on-missing", "raise"),
    )


def run_speculators(scratch: Path, *command) -> None:
    subprocess.run([str(part) for part in command], check=True, cwd=scratch)


def best_checkpoint(output: Path) -> Path:
    """The epoch the trainer kept as its best, else the last one it wrote."""
    best = output / config.SPECULATE_BEST_CHECKPOINT
    if best.exists():
        return best.resolve()
    epochs = sorted((path for path in output.iterdir() if path.is_dir()), key=os.path.getmtime)
    if not epochs:
        raise SystemExit(f"The training wrote no checkpoint under {output}")
    return epochs[-1]


def name_verifier(config_file: Path, verifier: str) -> None:
    """Names the verifier by its Reference, not by the path this step read it from."""
    speculator_config = json.loads(config_file.read_text())
    speculators = speculator_config.setdefault("speculators_config", {})
    speculators.setdefault("verifier", {})["name_or_path"] = verifier
    config_file.write_text(json.dumps(speculator_config, indent=2))
