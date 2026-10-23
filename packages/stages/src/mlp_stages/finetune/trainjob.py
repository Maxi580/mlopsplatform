import json
import os
import signal
import sys
from collections import deque
from pathlib import Path

from mlp_core import config
from mlp_core.pipeline_request.references import model_reference


def launch_phase(
    trainjob: str,
    pipeline_id: str,
    phase_index: str,
    request: str,
    distilled_dataset: str,
    previous_model_version: str,
    swept_parameters: str,
    checkpoint_minutes: str,
    resume_checkpoint: str,
    stop_after_checkpoint: str,
    model_version: str,
) -> None:
    """Trains the Phase as its TrainJob, then writes the Model Version it registered."""
    # 1. The compiler's TrainJob, running `finetune` with this step's arguments in this step's
    # MLflow Run; the Phase's own output file goes nowhere, as this step writes the step's.
    spec = json.loads(trainjob)
    trainer = spec["spec"]["trainer"]
    trainer["args"] = [
        pipeline_id,
        phase_index,
        request,
        distilled_dataset,
        previous_model_version,
        swept_parameters,
        checkpoint_minutes,
        resume_checkpoint,
        stop_after_checkpoint,
        os.devnull,
    ]
    mlflow_run = {name: value for name, value in os.environ.items() if name.startswith("MLFLOW_")}
    trainer["env"] = [{"name": name, "value": value} for name, value in mlflow_run.items()]
    jobs = TrainJobs()
    name = jobs.create(spec)

    # 2. KFP stops the step on a cancel, which stops the TrainJob too.
    def delete_trainjob(signum, frame):
        jobs.delete(name)
        sys.exit(f"Stopped, and deleted the Phase's TrainJob {name}")

    signal.signal(signal.SIGTERM, delete_trainjob)

    # 3. The main process's log in the step's, until the TrainJob ends.
    last_lines = deque(maxlen=config.TRAINJOB_FAILURE_LINES)
    for line in jobs.logs(name):
        print(line, flush=True)
        last_lines.append(line)
    if not jobs.succeeded(name):
        lines = "\n".join(last_lines)
        raise SystemExit(f"The Phase's TrainJob {name} failed; its last lines:\n{lines}")

    # 4. The Model Version its main process registered under the Run; an interrupted Phase
    # registers none.
    registered = ""
    if not stop_after_checkpoint:
        registered = registered_model_version(os.environ["MLFLOW_RUN_ID"])
    Path(model_version).write_text(registered)


def registered_model_version(run_id: str) -> str:
    # Only the launcher image has MLflow and the Kubeflow SDK.
    import mlflow

    [found] = mlflow.MlflowClient().search_model_versions(filter_string=f"run_id = '{run_id}'")
    return model_reference(found.name, int(found.version))


class TrainJobs:
    """The step namespace's TrainJobs: created with Kubernetes, followed with the Kubeflow SDK."""

    def __init__(self):
        from kubeflow.trainer import KubernetesBackendConfig, TrainerClient
        from kubernetes import client

        self.namespace = Path(config.SERVICE_ACCOUNT_NAMESPACE_FILE).read_text()
        self.trainer = TrainerClient(KubernetesBackendConfig(namespace=self.namespace))
        # The SDK can't set our command, Secret env or labels; it loaded the in-cluster config.
        self.custom_objects = client.CustomObjectsApi()

    def create(self, spec: dict) -> str:
        resource = config.TRAINJOB_RESOURCE
        body = {"apiVersion": f"{resource['group']}/{resource['version']}", "kind": "TrainJob"}
        created = self.custom_objects.create_namespaced_custom_object(
            **resource, namespace=self.namespace, body={**body, **spec}
        )
        return created["metadata"]["name"]

    def logs(self, name: str):
        """The main process's log lines, once Kueue admitted the TrainJob, until it ends."""
        self.wait(name, {"Running", "Complete", "Failed"})
        yield from self.trainer.get_job_logs(name, follow=True)

    def succeeded(self, name: str) -> bool:
        return self.wait(name, {"Complete", "Failed"}).status == "Complete"

    def wait(self, name: str, statuses: set[str]):
        return self.trainer.wait_for_job_status(
            name,
            statuses,
            timeout=int(config.TRAINJOB_TIMEOUT.total_seconds()),
            polling_interval=int(config.TRAINJOB_POLL_INTERVAL.total_seconds()),
        )

    def delete(self, name: str) -> None:
        self.trainer.delete_job(name)
