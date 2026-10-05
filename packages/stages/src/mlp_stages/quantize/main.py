import json
import os
import tempfile
from pathlib import Path

from mlp_core import config
from mlp_core.pipeline_request.schema import Calibration, PipelineRequest, Quantize
from mlp_stages.dataset_versions import download_dataset_version
from mlp_stages.model_versions import (
    model_with_adapter,
    register_model_version,
    renders_tools,
    tool_parser_of,
)


def quantize(pipeline_id: str, request: str, finetuned: str, model_version: str) -> None:
    """Quantizes the model and registers it as a Model Version, written to `model_version`."""
    # 1. The block of the resolved request; `@finetune` is the Model Version KFP handed over.
    resolved = PipelineRequest.model_validate_json(request)
    block = resolved.quantize
    model = finetuned if block.model == config.FINETUNE_OUTPUT else block.model
    run_id = os.environ["MLFLOW_RUN_ID"]
    from llmcompressor import oneshot
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    with tempfile.TemporaryDirectory() as directory:
        # 2. Full weights at their saved precision; an Adapter is merged into its base first, and
        # only the quantized result is registered.
        scratch = Path(directory)
        _, base_directory, adapter_directory = model_with_adapter(model, scratch / "start")
        loaded = AutoModelForCausalLM.from_pretrained(base_directory, dtype="auto")
        if adapter_directory:
            loaded = PeftModel.from_pretrained(loaded, adapter_directory).merge_and_unload()
        tokenizer = AutoTokenizer.from_pretrained(adapter_directory or base_directory)

        # 3. The scheme's one-shot run, measuring activations on the calibration rows if it needs.
        calibration, calibrating = block.calibration, {}
        if calibration:
            calibrating = {
                "dataset": calibration_rows(calibration, tokenizer, scratch),
                "num_calibration_samples": calibration.samples,
                "max_seq_length": calibration.max_length,
            }
        oneshot(model=loaded, tokenizer=tokenizer, recipe=recipe(block), **calibrating)

        # 4. In the compressed-tensors format vLLM serves, with the tokenizer and chat template.
        output = scratch / "model"
        loaded.save_pretrained(output, save_compressed=True)
        tokenizer.save_pretrained(output)

        # 5. The next full-weight Model Version of the Pipeline's Registered Model.
        tags = {
            "weights": "full",
            "quantization": block.scheme,
            "parent": model,
            "pipeline": pipeline_id,
            "tool_parser": tool_parser_of(model, loaded.config.model_type),
            "tools_rendered": str(renders_tools(tokenizer)).lower(),
        }
        Path(model_version).write_text(register_model_version(output, run_id, resolved, tags))


def recipe(block: Quantize) -> str:
    """The scheme's llm-compressor recipe, whose quantizing modifiers skip the `ignore` layers."""
    modifiers = {
        name: {**arguments, "ignore": block.ignore} if "scheme" in arguments else arguments
        for name, arguments in config.QUANTIZATION_SCHEMES[block.scheme].items()
    }
    return json.dumps({"quantize_stage": {"quantize_modifiers": modifiers}})


def calibration_rows(calibration: Calibration, tokenizer, scratch: Path):
    """`samples` random rows of the calibration Dataset as tokens, each cut at `max_length`."""
    from datasets import Dataset

    path = download_dataset_version(calibration.dataset, scratch)
    rows = Dataset.from_json(str(path), keep_in_memory=True)
    rows = rows.shuffle(seed=config.CALIBRATION_SHUFFLE_SEED)
    rows = rows.select(range(min(calibration.samples, len(rows))))
    return rows.map(
        lambda row: tokenizer(
            text_of(row, tokenizer),
            max_length=calibration.max_length,
            truncation=True,
            add_special_tokens=False,
        ),
        remove_columns=rows.column_names,
    )


def text_of(row: dict, tokenizer) -> str:
    """The row as the model reads it: its conversation in the chat template, else its text."""
    if row.get("messages") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(row["messages"], tokenize=False)
    if row.get("text") is not None:
        return row["text"]
    raise SystemExit(
        "The model has no chat template, so it can't calibrate on messages rows; name a Dataset "
        "of text rows in `calibration.dataset`"
    )
