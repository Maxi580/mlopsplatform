import contextlib
import os
import signal
import subprocess
import sys
import tempfile
from typing import Literal

from mlp_core import config
from mlp_core.pipeline_request.schema import Strict
from mlp_core.settings import Settings


class Snippet(Strict):
    """Python source, run as a script with `input` on its stdin."""

    code: str
    input: str = ""


class SnippetResult(Strict):
    status: Literal["ok", "error", "timeout", "memory_limit"]
    stdout: str
    stderr: str


def run_snippet(snippet: Snippet, settings: Settings) -> SnippetResult:
    """The snippet's result from a fresh Python process, killed at the Sandbox's limits."""
    # 1. A fresh process in an empty directory, without the Sandbox's environment; prlimit caps
    # its memory and the size of every file it writes, its output included. Unbuffered, so a killed
    # snippet still shows what it printed.
    limits = [
        f"--as={settings.sandbox_memory_mb * 2**20}",
        f"--fsize={config.SANDBOX_OUTPUT_LIMIT_BYTES}",
    ]
    with (
        tempfile.TemporaryDirectory() as directory,
        tempfile.TemporaryFile() as stdout,
        tempfile.TemporaryFile() as stderr,
    ):
        process = subprocess.Popen(
            ["prlimit", *limits, "--", sys.executable, "-I", "-u", "-c", snippet.code],
            stdin=subprocess.PIPE,
            stdout=stdout,
            stderr=stderr,
            cwd=directory,
            env={"PATH": os.defpath},
            process_group=0,
        )

        # 2. Run until it exits or the timeout; then its whole process group is killed.
        try:
            process.communicate(snippet.input.encode(), timeout=settings.sandbox_timeout_seconds)
            status = "ok" if process.returncode == 0 else "error"
        except subprocess.TimeoutExpired:
            status = "timeout"
        # Its group outlives it while a process it started still runs.
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        process.wait()

        # 3. Its output; a MemoryError, or a library's subclass of it, is how Python stops at the
        # memory limit.
        printed, errors = read_output(stdout), read_output(stderr)
    raised = errors.strip().rsplit("\n", 1)[-1].split(":", 1)[0]
    if status == "error" and raised.endswith("MemoryError"):
        status = "memory_limit"
    return SnippetResult(status=status, stdout=printed, stderr=errors)


def read_output(file) -> str:
    file.seek(0)
    return file.read().decode(errors="replace")
