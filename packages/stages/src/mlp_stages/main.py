import importlib
import logging
import os
import sys

from mlp_core import config
from mlp_stages.redaction import redact_output

# Step -> its function, imported on use: the trainer libraries exist only in the trainer images.
STEPS = {
    "fetch": "mlp_stages.operations.fetch:fetch",
    "cleanup": "mlp_stages.operations.cleanup:cleanup",
    # An Endpoint's init container: the Model Version files vLLM loads.
    "download": "mlp_stages.operations.download:download",
    "finetune": "mlp_stages.finetune.main:finetune",
    "evaluate": "mlp_stages.evaluate.main:evaluate",
    "serve": "mlp_stages.operations.serve:serve",
    # The Smoke Test's sandbox case.
    "check-sandbox": "mlp_stages.operations.check_sandbox:check_sandbox",
}


def main() -> None:
    """`mlp-stage <step> [args...]`: runs one Pipeline step with its Secret values redacted."""
    # 1. Redaction first, so nothing the step prints, logs or raises shows a Secret value.
    secrets = [os.environ.get(variable) for variable in config.SECRET_ENV_VARS.values()]
    redact_output([secret for secret in secrets if secret])
    logging.basicConfig(level=logging.INFO, force=True)

    # 2. The step.
    step, *args = sys.argv[1:]
    module, function = STEPS[step].split(":")
    getattr(importlib.import_module(module), function)(*args)
