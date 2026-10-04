from urllib.parse import urlencode

from typer.testing import CliRunner

from mlp_cli.main import app
from mlp_core import api_paths

from .conftest import TOKEN

REFERENCE = "hf:Qwen/Qwen3-0.6B@abc123"
FREE_PATH = f"{api_paths.MODEL_CACHE}?{urlencode({'reference': REFERENCE})}"


def mlp_cache(*args):
    return CliRunner().invoke(app, ["cache", *args])


def test_cache_lists_base_models_and_benchmarks_with_sizes_and_last_use(logged_in, fake_api):
    used = "2026-10-01T08:30:00+00:00"
    entries = [
        {"kind": "base_model", "reference": REFERENCE, "size_bytes": 1234, "last_used": used},
        {"kind": "benchmark", "reference": "lm_eval:gsm8k", "size_bytes": 66, "last_used": used},
    ]
    fake_api.answers[api_paths.MODEL_CACHE] = (200, {"entries": entries, "capacity_bytes": 10_000})

    result = mlp_cache()

    assert result.exit_code == 0, result.output
    assert "1,300 of 10,000 bytes used" in result.output
    base_model, benchmark = result.output.splitlines()[2:4]
    for text in ("Base Model", REFERENCE, "1,234 bytes", "2026-10-01 08:30"):
        assert text in base_model
    for text in ("Benchmark", "lm_eval:gsm8k", "66 bytes"):
        assert text in benchmark


def test_free_deletes_the_cached_base_model(logged_in, fake_api):
    fake_api.answers[FREE_PATH] = (204, b"")

    result = mlp_cache("free", REFERENCE)

    assert result.exit_code == 0, result.output
    assert fake_api.received == [(FREE_PATH, f"Bearer {TOKEN}")]


def test_a_refused_free_names_the_pipeline_using_it(logged_in, fake_api):
    detail = f"{REFERENCE} is used by Pipeline qwen-sft (#7)"
    fake_api.answers[FREE_PATH] = (409, {"detail": detail})

    result = mlp_cache("free", REFERENCE)

    assert result.exit_code == 1
    assert detail in result.output
