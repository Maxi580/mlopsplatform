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
    print_gpu = "print('0, 2048, 1024, 1024, 50, Card')"
    monkeypatch.setattr(config, "NVIDIA_SMI_COMMAND", (sys.executable, "-c", print_gpu))
    (tmp_path / "meminfo").write_text(MEMINFO)
    monkeypatch.setattr(config, "MEMINFO_FILE", str(tmp_path / "meminfo"))

    [reading] = httpx.get(reporter_url).json()

    assert (reading["name"], reading["memory_used_bytes"]) == ("Card", 2**30)


def test_a_reporter_without_a_working_nvidia_smi_answers_an_error(reporter_url, monkeypatch):
    monkeypatch.setattr(config, "NVIDIA_SMI_COMMAND", ("no-such-nvidia-smi",))

    assert httpx.get(reporter_url).status_code == 500


def test_a_reporter_with_a_gpu_in_an_error_state_answers_an_error(reporter_url, monkeypatch):
    print_gpu = "print('0, [Unknown Error], [Unknown Error], [Unknown Error], 0, Card')"
    monkeypatch.setattr(config, "NVIDIA_SMI_COMMAND", (sys.executable, "-c", print_gpu))
    monkeypatch.setattr(config, "MEMINFO_FILE", __file__)

    assert httpx.get(reporter_url).status_code == 500


def test_a_normal_gpu_reports_its_memory_and_utilisation():
    nvidia_smi = "0, 24576, 1024, 23552, 37, NVIDIA GeForce RTX 4090\n"

    assert gpu_readings(nvidia_smi, MEMINFO) == [
        {
            "index": 0,
            "name": "NVIDIA GeForce RTX 4090",
            "memory_total_bytes": 24 * 2**30,
            "memory_used_bytes": 2**30,
            "memory_free_bytes": 23 * 2**30,
            "shared_with_system": False,
            "utilization_percent": 37,
        }
    ]


def test_memory_reported_as_not_supported_comes_from_meminfo_as_shared_with_the_system():
    nvidia_smi = "0, [Not Supported], [Not Supported], [Not Supported], 12, NVIDIA GB10\n"

    [reading] = gpu_readings(nvidia_smi, MEMINFO)

    assert reading["memory_total_bytes"] == 128 * 2**30
    assert reading["memory_used_bytes"] == 32 * 2**30
    assert reading["memory_free_bytes"] == 96 * 2**30
    assert reading["shared_with_system"] is True
    assert reading["utilization_percent"] == 12


def test_utilisation_a_gpu_does_not_report_is_left_out():
    nvidia_smi = "0, [N/A], [N/A], [N/A], [N/A], NVIDIA GB10\n"

    [reading] = gpu_readings(nvidia_smi, MEMINFO)

    assert reading["utilization_percent"] is None


def test_every_gpu_of_a_machine_gets_a_reading():
    nvidia_smi = "0, 2048, 0, 2048, 0, Card, v2\n1, 2048, 1024, 1024, 50, Card, v2\n"

    readings = gpu_readings(nvidia_smi, MEMINFO)

    assert [(r["index"], r["name"], r["memory_used_bytes"]) for r in readings] == [
        (0, "Card, v2", 0),
        (1, "Card, v2", 2**30),
    ]
