import json
import os
import subprocess
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from mlp_core import config


def serve_gpu_readings() -> None:
    """Answers every GET with this machine's GPU readings as JSON, until stopped."""
    address = ("", int(os.environ["GPU_REPORTER_PORT"]))
    HTTPServer(address, GpuReadingsHandler).serve_forever()


class GpuReadingsHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        try:
            body = json.dumps(read_gpus()).encode()
        # The API skips a machine that can't say, e.g. with a stuck driver or a failing GPU.
        except (OSError, subprocess.SubprocessError, ValueError) as error:
            self.send_error(500, explain=str(error))
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def read_gpus() -> list[dict]:
    nvidia_smi = subprocess.run(
        config.NVIDIA_SMI_COMMAND,
        capture_output=True,
        text=True,
        check=True,
        timeout=config.GPU_REPORTER_TIMEOUT.total_seconds(),
    )
    return gpu_readings(nvidia_smi.stdout, Path(config.MEMINFO_FILE).read_text())


def gpu_readings(nvidia_smi_csv: str, meminfo: str) -> list[dict]:
    """A reading per line of nvidia-smi's CSV; memory it doesn't report is the machine's."""
    readings = []
    for total_mib in nvidia_smi_csv.splitlines():
        shared = total_mib in config.NVIDIA_SMI_NOT_REPORTED
        memory = system_memory_bytes(meminfo) if shared else int(total_mib) * 2**20
        readings.append({"memory_total_bytes": memory, "shared_with_system": shared})
    return readings


def system_memory_bytes(meminfo: str) -> int:
    """What a GPU sharing the machine's memory, as the GB10 does, has of it."""
    kibibytes = {
        key: int(value.split()[0])
        for key, value in (line.split(":") for line in meminfo.splitlines())
    }
    return kibibytes["MemTotal"] * 2**10
