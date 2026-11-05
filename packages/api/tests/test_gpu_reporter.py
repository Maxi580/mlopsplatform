import sys
import threading
from http.server import HTTPServer

import httpx2 as httpx
import pytest

from mlp_api.gpu_usage.reporter import GpuReadingsHandler, gpu_readings
from mlp_core import config

MEMINFO = """\
MemTotal:       134217728 kB
MemFree:         1048576 kB
MemAvailable:   100663296 kB
HugePages_Total:       0
"""


@pytest.fixture
def reporter_url():
    server = HTTPServer(("127.0.0.1", 0), GpuReadingsHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()
    server.server_close()


def test_a_reporter_answers_the_readings_as_json(reporter_url, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "NVIDIA_SMI_COMMAND", (sys.executable, "-c", "print('2048')"))
    (tmp_path / "meminfo").write_text(MEMINFO)
    monkeypatch.setattr(config, "MEMINFO_FILE", str(tmp_path / "meminfo"))

    assert httpx.get(reporter_url).json() == [
        {"memory_total_bytes": 2**31, "shared_with_system": False}
    ]


def test_a_reporter_without_a_working_nvidia_smi_answers_an_error(reporter_url, monkeypatch):
    monkeypatch.setattr(config, "NVIDIA_SMI_COMMAND", ("no-such-nvidia-smi",))

    assert httpx.get(reporter_url).status_code == 500


def test_a_reporter_with_a_gpu_in_an_error_state_answers_an_error(reporter_url, monkeypatch):
    print_gpu = "print('[Unknown Error]')"
    monkeypatch.setattr(config, "NVIDIA_SMI_COMMAND", (sys.executable, "-c", print_gpu))
    monkeypatch.setattr(config, "MEMINFO_FILE", __file__)

    assert httpx.get(reporter_url).status_code == 500


def test_a_normal_gpu_reports_its_memory():
    assert gpu_readings("24576\n", MEMINFO) == [
        {"memory_total_bytes": 24 * 2**30, "shared_with_system": False}
    ]


@pytest.mark.parametrize("not_reported", ["[Not Supported]", "[N/A]"])
def test_memory_not_reported_comes_from_meminfo_as_shared_with_the_system(not_reported):
    assert gpu_readings(f"{not_reported}\n", MEMINFO) == [
        {"memory_total_bytes": 128 * 2**30, "shared_with_system": True}
    ]


def test_every_gpu_of_a_machine_gets_a_reading():
    assert len(gpu_readings("2048\n2048\n", MEMINFO)) == 2
