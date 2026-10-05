import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

mlflow = pytest.importorskip("mlflow", reason="needs the evaluate extra: uv sync --extra evaluate")
pytest.importorskip("datasets", reason="needs the evaluate extra: uv sync --extra evaluate")

from mlp_stages import model_versions  # noqa: E402
from mlp_stages.main import main  # noqa: E402
from mlp_stages.quantize import main as quantize_main  # noqa: E402

BASE_MODEL = "hf:Qwen/Qwen2.5-0.5B-Instruct@7ae557604adf67be50417f59c2c2f167def9a775"
# Where fetch put the Base Model in the Model Cache.
BASE_MODEL_FILES = Path("/model-cache/hub/qwen")
ROWS = [
    {"messages": [{"role": "user", "content": f"question {number} " * 20}]} for number in range(5)
]


class FakeModel:
    def __init__(self, path, merged_adapter=None):
        self.path, self.merged_adapter = Path(path), merged_adapter
        self.config = SimpleNamespace(model_type="qwen2")

    def save_pretrained(self, directory, save_compressed):
        Path(directory).mkdir(parents=True)
        (Path(directory) / "config.json").write_text(json.dumps({"compressed": save_compressed}))


class FakeTokenizer:
    chat_template = "{{ messages }}"

    def apply_chat_template(self, conversation, tools=None, tokenize=False):
        return " ".join(message["content"] for message in conversation) + str(tools or "")

    def __call__(self, text, max_length, truncation, add_special_tokens):
        return {"input_ids": list(range(len(text.split())))[:max_length]}

    def save_pretrained(self, directory):
        (Path(directory) / "tokenizer.json").write_text("{}")


class FakeMlflow:
    """The registry, whose Model Versions download where asked, and the step's Run."""

    def __init__(self):
        self.versions = {}
        self.registered = []
        # `models:/<name>/<version>` -> where its files were downloaded.
        self.downloads = {}

    def __call__(self):
        return self

    def get_model_version(self, name, version):
        return SimpleNamespace(tags=self.versions[(name, int(version))])

    def download_artifacts(self, uri, dst_path):
        self.downloads[uri] = Path(dst_path)
        Path(dst_path).mkdir(parents=True)
        return dst_path

    def log_artifacts(self, run_id, directory, path):
        self.logged = sorted(file.name for file in Path(directory).iterdir())

    def create_registered_model(self, name):
        pass

    def get_run(self, run_id):
        return SimpleNamespace(info=SimpleNamespace(artifact_uri=f"mlflow-artifacts:/{run_id}"))

    def create_model_version(self, name, source, run_id, tags):
        self.registered.append((name, tags))
        return SimpleNamespace(version="3")


@pytest.fixture
def step(monkeypatch, tmp_path):
    """Runs `mlp-stage quantize`, with fakes for transformers, PEFT, llm-compressor and MLflow."""
    fakes = SimpleNamespace(mlflow=FakeMlflow(), oneshot={})
    libraries = {
        "transformers": {
            "AutoModelForCausalLM": SimpleNamespace(
                from_pretrained=lambda path, dtype: FakeModel(path)
            ),
            "AutoTokenizer": SimpleNamespace(from_pretrained=lambda path: FakeTokenizer()),
            # `datasets` looks for it when hashing a map function.
            "PreTrainedTokenizerBase": FakeTokenizer,
        },
        "peft": {
            "PeftModel": SimpleNamespace(
                from_pretrained=lambda model, adapter: SimpleNamespace(
                    merge_and_unload=lambda: FakeModel(model.path, merged_adapter=Path(adapter))
                )
            )
        },
        "llmcompressor": {"oneshot": lambda **arguments: fakes.oneshot.update(arguments)},
    }
    for name, attributes in libraries.items():
        module = ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(mlflow, "MlflowClient", fakes.mlflow)
    monkeypatch.setattr(mlflow.artifacts, "download_artifacts", fakes.mlflow.download_artifacts)
    monkeypatch.setattr(
        model_versions, "snapshot_download", lambda repo, revision: BASE_MODEL_FILES
    )

    def download_dataset_version(reference, directory):
        path = directory / "calibration.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in ROWS))
        return path

    monkeypatch.setattr(quantize_main, "download_dataset_version", download_dataset_version)
    monkeypatch.setenv("MLFLOW_RUN_ID", "run-1")
    # main() wraps stdout and stderr; monkeypatch puts the originals back.
    monkeypatch.setattr(sys, "stdout", sys.stdout)
    monkeypatch.setattr(sys, "stderr", sys.stderr)
    output = tmp_path / "model_version"

    def run(model, finetuned="", **quantize):
        quantize = {"model": model, "scheme": "fp8-dynamic", **quantize}
        request = {"name": "qwen-sft", "quantize": quantize}
        arguments = ["7", json.dumps(request), finetuned, str(output)]
        monkeypatch.setattr(sys, "argv", ["mlp-stage", "quantize", *arguments])
        main()
        return output.read_text()

    fakes.run = run
    return fakes


def modifiers(step) -> dict:
    return json.loads(step.oneshot["recipe"])["quantize_stage"]["quantize_modifiers"]


def test_an_adapter_is_merged_into_its_base_then_quantized_and_registered_as_full_weights(step):
    adapter = {"weights": "adapter", "base_model": BASE_MODEL, "tool_parser": "hermes"}
    step.mlflow.versions[("qwen-sft", 1)] = adapter

    registered = step.run("model:qwen-sft@1")

    assert registered == "model:qwen-sft@3"
    merged = step.oneshot["model"]
    assert merged.path == BASE_MODEL_FILES
    assert merged.merged_adapter == step.mlflow.downloads["models:/qwen-sft/1"]
    assert "dataset" not in step.oneshot
    assert modifiers(step) == {
        "QuantizationModifier": {
            "targets": ["Linear"],
            "scheme": "FP8_DYNAMIC",
            "ignore": ["lm_head"],
        }
    }
    [(name, tags)] = step.mlflow.registered
    assert name == "qwen-sft"
    assert tags == {
        "weights": "full",
        "quantization": "fp8-dynamic",
        "parent": "model:qwen-sft@1",
        "pipeline": "7",
        "tool_parser": "hermes",
        "tools_rendered": "true",
    }
    assert step.mlflow.logged == ["config.json", "pipeline_request.json", "tokenizer.json"]


def test_a_calibrated_scheme_measures_random_rows_in_the_chat_template_cut_at_max_length(step):
    calibration = {"dataset": "dataset:chat@1", "samples": 3, "max_length": 16}

    step.run(BASE_MODEL, scheme="w8a8-int8", calibration=calibration, ignore=["lm_head", "gate"])

    rows = step.oneshot["dataset"]
    assert rows.column_names == ["input_ids"]
    assert [len(row["input_ids"]) for row in rows] == [16, 16, 16]
    assert (step.oneshot["num_calibration_samples"], step.oneshot["max_seq_length"]) == (3, 16)
    assert modifiers(step) == {
        "SmoothQuantModifier": {"smoothing_strength": 0.8},
        "GPTQModifier": {"targets": ["Linear"], "scheme": "W8A8", "ignore": ["lm_head", "gate"]},
    }
    [(_, tags)] = step.mlflow.registered
    assert (tags["parent"], tags["tool_parser"]) == (BASE_MODEL, "hermes")


def test_the_output_of_finetune_is_the_model_version_kfp_hands_over(step):
    step.mlflow.versions[("qwen-sft", 2)] = {"weights": "full", "tool_parser": "none"}

    step.run("@finetune", finetuned="model:qwen-sft@2")

    assert step.oneshot["model"].path == step.mlflow.downloads["models:/qwen-sft/2"]
    [(_, tags)] = step.mlflow.registered
    assert (tags["parent"], tags["tool_parser"]) == ("model:qwen-sft@2", "none")
