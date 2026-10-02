import subprocess
import sys

import pytest


# Unsloth must be imported before these, so only the backend chosen by the request imports them.
def test_finetune_imports_no_training_library_before_its_backend():
    pytest.importorskip("torch", reason="needs a trainer extra, e.g. uv sync --extra hf")
    libraries = ["transformers", "trl", "peft", "unsloth"]
    check = (
        f"import sys, mlp_stages.finetune.main; print([m for m in {libraries} if m in sys.modules])"
    )

    imported = subprocess.run([sys.executable, "-c", check], capture_output=True, text=True)

    assert imported.stdout.strip() == "[]", imported.stderr
