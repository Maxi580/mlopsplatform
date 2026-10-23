import json
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest


# Unsloth must be imported before these, so only the backend chosen by the request imports them.
@pytest.mark.parametrize("step", ["mlp_stages.finetune.main", "mlp_stages.sweep.main"])
def test_a_training_step_imports_no_training_library_before_its_backend(step):
    pytest.importorskip("torch", reason="needs a trainer extra, e.g. uv sync --extra hf")
    libraries = ["transformers", "trl", "peft", "unsloth"]
    check = f"import sys, {step}; print([m for m in {libraries} if m in sys.modules])"

    imported = subprocess.run([sys.executable, "-c", check], capture_output=True, text=True)

    assert imported.stdout.strip() == "[]", imported.stderr


@pytest.fixture
def phase_process(monkeypatch, tmp_path):
    """Runs `mlp-stage finetune` as one of `accelerate launch`'s processes, with fakes around it."""
    pytest.importorskip("torch", reason="needs a trainer extra, e.g. uv sync --extra hf")
    import accelerate

    from mlp_stages.finetune import checkpoints
    from mlp_stages.finetune import main as finetune_main
    from mlp_stages.main import main

    fakes = SimpleNamespace(registered=[], deleted=[], output=tmp_path / "model_version")
    backend = ModuleType("mlp_stages.finetune.backends.hf")
    backend.load_tokenizer = lambda path: SimpleNamespace(chat_template="{{ messages }}")
    backend.build_trainer = lambda *arguments: FakeTrainer()
    backend.save_model_version = lambda trainer, phase, tokenizer, directory, base: None
    monkeypatch.setitem(sys.modules, backend.__name__, backend)
    monkeypatch.setattr(
        finetune_main, "model_with_adapter", lambda reference, scratch: (BASE_MODEL, tmp_path, None)
    )

    def download_dataset_version(reference, directory):
        path = directory / "rows.jsonl"
        path.write_text(json.dumps({"text": "Hello"}) + "\n")
        return path

    monkeypatch.setattr(finetune_main, "download_dataset_version", download_dataset_version)
    monkeypatch.setattr(finetune_main, "phase_tags", lambda *arguments: {})

    def register_model_version(directory, run_id, request, tags):
        fakes.registered.append(run_id)
        return "model:qwen-sft@1"

    monkeypatch.setattr(finetune_main, "register_model_version", register_model_version)
    monkeypatch.setattr(checkpoints, "delete_checkpoints", fakes.deleted.append)
    monkeypatch.setenv("MLFLOW_RUN_ID", "run-1")
    monkeypatch.setattr(sys, "stdout", sys.stdout)
    monkeypatch.setattr(sys, "stderr", sys.stderr)

    def run(main_process: bool, stop_after_checkpoint=""):
        state = SimpleNamespace(is_main_process=main_process)
        monkeypatch.setattr(accelerate, "PartialState", lambda: state)
        arguments = ["7", "0", json.dumps(REQUEST), "", "", "", "30", "", stop_after_checkpoint]
        monkeypatch.setattr(sys, "argv", ["mlp-stage", "finetune", *arguments, str(fakes.output)])
        main()

    fakes.run = run
    return fakes


class FakeTrainer:
    def __init__(self):
        self.model = SimpleNamespace(config=SimpleNamespace(model_type="qwen2"))

    def add_callback(self, callback):
        pass

    def train(self, resume_from_checkpoint):
        pass


BASE_MODEL = "hf:Qwen/Qwen2.5-0.5B-Instruct@7ae557604adf67be50417f59c2c2f167def9a775"
REQUEST = {
    "name": "qwen-sft",
    "finetune": {
        "base_model": BASE_MODEL,
        "backend": "hf",
        "phases": [
            {
                "algorithm": "sft",
                "dataset": "dataset:chat@1",
                "settings": {
                    "learning_rate": 1e-4,
                    "num_train_epochs": 1,
                    "per_device_train_batch_size": 2,
                    "gradient_accumulation_steps": 1,
                    "max_length": 64,
                },
            }
        ],
    },
}


def test_the_main_process_registers_the_model_version_and_writes_the_output(phase_process):
    phase_process.run(main_process=True)

    assert phase_process.registered == ["run-1"]
    assert phase_process.output.read_text() == "model:qwen-sft@1"
    assert phase_process.deleted == ["checkpoints/7/0/"]


def test_the_other_processes_register_nothing_and_write_no_output(phase_process):
    phase_process.run(main_process=False)

    assert phase_process.registered == []
    assert not phase_process.output.exists()
    assert phase_process.deleted == []


def test_an_interrupted_phase_has_only_the_main_process_write_its_empty_output(phase_process):
    phase_process.run(main_process=False, stop_after_checkpoint="true")
    assert not phase_process.output.exists()

    phase_process.run(main_process=True, stop_after_checkpoint="true")
    assert phase_process.output.read_text() == ""
    assert phase_process.registered == []


@pytest.mark.parametrize(("world_process_zero", "uploads"), [(True, 1), (False, 0)])
def test_only_the_main_process_uploads_a_checkpoint(monkeypatch, world_process_zero, uploads):
    pytest.importorskip("torch", reason="needs a trainer extra, e.g. uv sync --extra hf")
    from mlp_stages.finetune import checkpoints

    uploaded = []
    monkeypatch.setattr(checkpoints, "upload_checkpoint", lambda *arguments: uploaded.append(1))
    callback = checkpoints.CheckpointUploads("checkpoints/7/0/", 0, False)
    args = SimpleNamespace(output_dir="out")
    state = SimpleNamespace(global_step=3, is_world_process_zero=world_process_zero)
    control = SimpleNamespace(should_save=False, should_training_stop=False)

    callback.on_step_end(args, state, control)
    callback.on_save(args, state, control)

    assert control.should_save
    assert len(uploaded) == uploads
