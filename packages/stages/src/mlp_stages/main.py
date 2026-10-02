import logging
import os
import sys

from mlp_core import config
from mlp_stages.cleanup import cleanup
from mlp_stages.fetch import fetch
from mlp_stages.redaction import redact_output

STEPS = {"fetch": fetch, "cleanup": cleanup}


def main() -> None:
    """`mlp-stage <step> [args...]`: runs one Pipeline step with its Secret values redacted."""
    # 1. Redaction first, so nothing the step prints, logs or raises shows a Secret value.
    secrets = [os.environ.get(variable) for variable in config.SECRET_ENV_VARS.values()]
    redact_output([secret for secret in secrets if secret])
    logging.basicConfig(level=logging.INFO, force=True)

    # 2. The step.
    step, *args = sys.argv[1:]
    STEPS[step](*args)
