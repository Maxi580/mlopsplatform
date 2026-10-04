import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from mlp_core import config
from mlp_stages.evaluate import harness
from mlp_stages.main import main
from mlp_stages.operations import fetch

HF_TOKEN = "hf_" + "s3cr3tT0ken" * 4
REPO = "Qwen/Qwen2.5-0.5B-Instruct"
COMMIT = "7ae557604adf67be50417f59c2c2f167def9a775"
BASE_MODEL = f"hf:{REPO}@{COMMIT}"
# Repo -> the sizes of its files on Hugging Face.
HUB_FILES = {REPO: [400, 100]}


class FakeHfApi:
    def __init__(self, token):
        pass

    def model_info(self, repo, revision, files_metadata):
        return SimpleNamespace(siblings=[SimpleNamespace(size=size) for size in HUB_FILES[repo]])


@pytest.fixture
def model_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "MODEL_CACHE_PATH", str(tmp_path))
    monkeypatch.setattr(fetch, "HF_HUB_CACHE", str(tmp_path / "hub"))
    monkeypatch.setattr(fetch, "HfApi", FakeHfApi)
    return tmp_path / "hub"


class FakeLmEval:
    """Records each process lm-eval runs in; a download writes its datasets where told to."""

    def __init__(self):
        self.runs = []
        self.failing = set()

    def __call__(self, command, env):
        assert command[command.index("--model") + 1] == "dummy"
        task = command[command.index("--tasks") + 1]
        self.runs.append((task, env))
        if task in self.failing:
            return SimpleNamespace(returncode=1)
        datasets = Path(env["HF_DATASETS_CACHE"])
        datasets.mkdir(parents=True)
        (datasets / "test.arrow").write_bytes(b"x" * 10)
        return SimpleNamespace(returncode=0)


@pytest.fixture
def lm_eval(monkeypatch):
    fake = FakeLmEval()
    monkeypatch.setattr(harness.subprocess, "run", fake)
    return fake


@pytest.fixture
def downloads(monkeypatch, model_cache):
    downloads = []
    monkeypatch.setattr(
        fetch, "snapshot_download", lambda repo, **options: downloads.append((repo, options))
    )
    # main() wraps stdout and stderr; monkeypatch puts the originals back.
    monkeypatch.setattr(sys, "stdout", sys.stdout)
    monkeypatch.setattr(sys, "stderr", sys.stderr)
    return downloads


def cache_base_model(model_cache, repo, commit, size_bytes):
    snapshot = model_cache / f"models--{repo.replace('/', '--')}" / "snapshots" / commit
    snapshot.mkdir(parents=True)
    (snapshot / "model.safetensors").write_bytes(b"x" * size_bytes)


def mlp_stage(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["mlp-stage", *args])
    main()


def test_fetch_downloads_the_base_model_at_its_pinned_commit(monkeypatch, downloads):
    monkeypatch.setenv("HF_TOKEN", HF_TOKEN)

    mlp_stage(monkeypatch, "fetch", "100Gi", BASE_MODEL)

    assert downloads == [
        (
            "Qwen/Qwen2.5-0.5B-Instruct",
            {"revision": "7ae557604adf67be50417f59c2c2f167def9a775", "token": HF_TOKEN},
        )
    ]


def test_fetch_downloads_anonymously_without_a_token(monkeypatch, downloads):
    monkeypatch.delenv("HF_TOKEN", raising=False)

    mlp_stage(monkeypatch, "fetch", "100Gi", BASE_MODEL)

    assert downloads[0][1]["token"] is None


def test_secret_values_are_redacted_from_the_step_output(monkeypatch, downloads, capsys):
    monkeypatch.setenv("HF_TOKEN", HF_TOKEN)

    def leak(repo, **options):
        print(f"token={options['token']}")
        logging.getLogger("huggingface_hub").warning("header Bearer %s", options["token"])
        raise RuntimeError(f"401 for token {options['token']}")

    monkeypatch.setattr(fetch, "snapshot_download", leak)

    with pytest.raises(RuntimeError):
        mlp_stage(monkeypatch, "fetch", "100Gi", BASE_MODEL)

    output = capsys.readouterr()
    assert HF_TOKEN not in output.out + output.err
    assert "token=***" in output.out
    assert "Bearer ***" in output.err


def test_cleanup_runs(monkeypatch, downloads):
    mlp_stage(monkeypatch, "cleanup")


def test_fetch_fails_clearly_when_the_download_does_not_fit(monkeypatch, downloads, model_cache):
    cache_base_model(model_cache, "org/other", "1" * 40, 600)

    with pytest.raises(RuntimeError, match="needs 500 bytes but the Model Cache has 400 of 1000"):
        mlp_stage(monkeypatch, "fetch", "1000", BASE_MODEL)

    assert downloads == []


def test_files_already_cached_need_no_room(monkeypatch, downloads, model_cache):
    cache_base_model(model_cache, "org/other", "1" * 40, 400)
    cache_base_model(model_cache, REPO, COMMIT, 400)

    mlp_stage(monkeypatch, "fetch", "1000", BASE_MODEL)

    assert len(downloads) == 1


def test_fetch_downloads_benchmark_datasets_into_the_model_cache(
    monkeypatch, downloads, model_cache, lm_eval
):
    mlp_stage(monkeypatch, "fetch", "100Gi", f"{BASE_MODEL},lm_eval:gsm8k")

    [(task, env)] = lm_eval.runs
    assert task == "gsm8k"
    benchmark = model_cache.parent / "benchmarks" / "lm_eval" / "gsm8k"
    assert env["HF_DATASETS_CACHE"] == str(benchmark / "datasets")
    assert env["HF_HUB_CACHE"] == str(benchmark / "hub")
    assert (benchmark / "fetched").exists()
    assert len(downloads) == 1


def test_a_fetched_benchmark_is_not_downloaded_again(monkeypatch, downloads, model_cache, lm_eval):
    mlp_stage(monkeypatch, "fetch", "100Gi", "lm_eval:gsm8k")
    mlp_stage(monkeypatch, "fetch", "100Gi", "lm_eval:gsm8k")

    assert len(lm_eval.runs) == 1
    assert downloads == []


def test_a_benchmark_that_fails_to_download_is_left_unmarked_and_fetch_goes_on(
    monkeypatch, downloads, model_cache, lm_eval
):
    lm_eval.failing.add("gsm8k")

    mlp_stage(monkeypatch, "fetch", "100Gi", "lm_eval:gsm8k,lm_eval:ifeval")

    benchmarks = model_cache.parent / "benchmarks" / "lm_eval"
    assert not (benchmarks / "gsm8k" / "fetched").exists()
    assert (benchmarks / "ifeval" / "fetched").exists()


def test_benchmark_sizes_count_towards_the_room_fetch_needs(
    monkeypatch, downloads, model_cache, lm_eval
):
    size = config.BENCHMARKS["lm_eval:gsm8k"]["size_bytes"]

    with pytest.raises(RuntimeError, match=f"needs {500 + size} bytes"):
        mlp_stage(monkeypatch, "fetch", str(size), f"{BASE_MODEL},lm_eval:gsm8k")

    assert (downloads, lm_eval.runs) == ([], [])
