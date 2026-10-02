import sys
from typing import TextIO


def redact_output(secrets: list[str]) -> None:
    """Replaces every Secret value written to stdout or stderr with ***."""
    if secrets:
        sys.stdout = RedactingStream(sys.stdout, secrets)
        sys.stderr = RedactingStream(sys.stderr, secrets)


class RedactingStream:
    def __init__(self, stream: TextIO, secrets: list[str]):
        self.stream = stream
        self.secrets = secrets

    def write(self, text: str) -> int:
        for secret in self.secrets:
            text = text.replace(secret, "***")
        return self.stream.write(text)

    def __getattr__(self, name: str):
        return getattr(self.stream, name)
