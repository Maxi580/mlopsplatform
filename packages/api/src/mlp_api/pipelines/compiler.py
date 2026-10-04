import os
from dataclasses import dataclass

from google.protobuf import json_format
from kfp import dsl, kubernetes

from mlp_core import config
from mlp_core.pipeline_request.schema import PipelineRequest
from mlp_core.settings import Settings


@dataclass(frozen=True)
class StepEnvironment:
    """The images and storage of the steps, from the API's environment; tests make their own."""

    stages_image: str
    # Training backend -> its trainer image.
    trainer_images: dict[str, str]
    model_cache_pvc: str
    object_store_url: str
    object_store_bucket: str
    # How the `serve` step reaches the API inside the cluster.
    api_url: str
    sandbox_url: str

    @classmethod
    def from_environment(cls) -> "StepEnvironment":
        return cls(
            stages_image=os.environ["STAGES_IMAGE"],
            trainer_images={"hf": os.environ["TRAINER_HF_IMAGE"]},
            model_cache_pvc=os.environ["MODEL_CACHE_PVC"],
            object_store_url=os.environ["S3_ENDPOINT_URL"],
            object_store_bucket=os.environ["S3_BUCKET"],
            api_url=os.environ["API_URL"],
            sandbox_url=os.environ["SANDBOX_URL"],
        )


def compile_pipeline(
    pipeline_id: int, request: PipelineRequest, steps: StepEnvironment, settings: Settings
) -> dict:
    """The Kubeflow pipeline spec for a resolved request, in the form KFP's run API takes."""

    @dsl.container_component
    def cleanup():
        return dsl.ContainerSpec(image=steps.stages_image, command=["mlp-stage", "cleanup"])

    @dsl.pipeline(name=request.name)
    def pipeline():
        cleanup_task = cleanup().set_caching_options(False)
        with dsl.ExitHandler(cleanup_task):
            finetune_task = finetune_step(pipeline_id, request, steps, settings.gpus_per_stage)
            # A Model Version to start from lies in the object store; only a Base Model is fetched.
            if request.finetune.base_model:
                fetch_task = fetch_step(request.finetune.base_model, steps, settings)
                # Only fetch gets the token, from the Secret; a pipeline parameter would be logged.
                kubernetes.use_secret_as_env(
                    fetch_task,
                    config.PIPELINE_SECRET_NAME.format(id=pipeline_id),
                    {"hf_token": config.SECRET_ENV_VARS["hf_token"]},
                    optional=True,
                )
                finetune_task.after(fetch_task)
            if request.serve:
                serve_step(pipeline_id, steps).after(finetune_task)

    return pipeline_run_spec(pipeline)


def compile_smoke_test(
    pipeline_id: int,
    name: str,
    base_model: str,
    sandbox: bool,
    finetune_cases: dict[str, PipelineRequest],
    steps: StepEnvironment,
    settings: Settings,
) -> dict:
    """The Smoke Test's Kubeflow pipeline spec: one node per case, named after it, in a chain."""

    @dsl.pipeline(name=name)
    def pipeline():
        previous = fetch_step(base_model, steps, settings).set_display_name("fetch")
        # Each later case runs once the one before it ended, even if that failed; no data passes
        # between them.
        if sandbox:
            previous = (
                sandbox_step(steps, settings)
                .set_display_name(config.SMOKE_TEST_SANDBOX_CASE)
                .after(previous)
                .ignore_upstream_failure()
            )
        for case, request in finetune_cases.items():
            previous = (
                finetune_step(pipeline_id, request, steps, settings.gpus_per_stage)
                .set_display_name(case)
                .after(previous)
                .ignore_upstream_failure()
            )
            # The `serve` Stage case passes or fails with its serve step, run once training passed.
            if request.serve:
                previous.set_display_name(f"{case}-finetune")
                serve_task = serve_step(pipeline_id, steps).after(previous)
                previous = serve_task.set_display_name(case)

    return pipeline_run_spec(pipeline)


def fetch_step(base_model: str, steps: StepEnvironment, settings: Settings):
    @dsl.container_component
    def fetch(base_model: str, model_cache_size: str):
        return dsl.ContainerSpec(
            image=steps.stages_image,
            command=["mlp-stage", "fetch"],
            args=[base_model, model_cache_size],
        )

    task = fetch(base_model=base_model, model_cache_size=settings.model_cache_size)
    use_model_cache(task, steps)
    return task


def finetune_step(
    pipeline_id: int, request: PipelineRequest, steps: StepEnvironment, gpus_per_stage: int
):
    trainer_image = steps.trainer_images[request.finetune.backend]

    @dsl.container_component
    def finetune(pipeline_id: str, phase_index: str, request: str):
        return dsl.ContainerSpec(
            image=trainer_image,
            command=["mlp-stage", "finetune"],
            args=[pipeline_id, phase_index, request],
        )

    # The request holds no Secret value, so it can be a parameter.
    task = finetune(
        pipeline_id=str(pipeline_id), phase_index="0", request=request.model_dump_json()
    )
    use_model_cache(task, steps)
    task.set_env_variable("HF_HUB_OFFLINE", "1")
    task.set_accelerator_type(config.GPU_RESOURCE)
    task.set_accelerator_limit(gpus_per_stage)
    use_object_store(task, steps)
    return task


def serve_step(pipeline_id: int, steps: StepEnvironment):
    @dsl.container_component
    def serve(pipeline_id: str):
        return dsl.ContainerSpec(
            image=steps.stages_image, command=["mlp-stage", "serve"], args=[pipeline_id]
        )

    # The API starts the Endpoint from the request it stored; the step only asks, with the serve
    # token from the Secret.
    task = serve(pipeline_id=str(pipeline_id))
    task.set_caching_options(False)
    task.set_env_variable("API_URL", steps.api_url)
    kubernetes.use_secret_as_env(
        task,
        config.PIPELINE_SECRET_NAME.format(id=pipeline_id),
        {"serve_token": config.SECRET_ENV_VARS["serve_token"]},
    )
    return task


# Runs snippets in the Sandbox, which only Pipeline steps can reach.
def sandbox_step(steps: StepEnvironment, settings: Settings):
    @dsl.container_component
    def sandbox(sandbox_memory_mb: str):
        return dsl.ContainerSpec(
            image=steps.stages_image,
            command=["mlp-stage", "check-sandbox"],
            args=[sandbox_memory_mb],
        )

    task = sandbox(sandbox_memory_mb=str(settings.sandbox_memory_mb))
    task.set_caching_options(False)
    task.set_env_variable("SANDBOX_URL", steps.sandbox_url)
    return task


def pipeline_run_spec(pipeline) -> dict:
    return {
        "pipeline_spec": json_format.MessageToDict(pipeline.pipeline_spec),
        "platform_spec": json_format.MessageToDict(pipeline.platform_spec),
    }


def use_model_cache(task, steps: StepEnvironment) -> None:
    task.set_caching_options(False)
    task.set_env_variable("HF_HOME", config.MODEL_CACHE_PATH)
    kubernetes.mount_pvc(task, steps.model_cache_pvc, config.MODEL_CACHE_PATH)


# Reads Dataset Versions with the object store keys KFP's own steps use.
def use_object_store(task, steps: StepEnvironment) -> None:
    task.set_env_variable("S3_ENDPOINT_URL", steps.object_store_url)
    task.set_env_variable("S3_BUCKET", steps.object_store_bucket)
    kubernetes.use_secret_as_env(
        task, config.OBJECT_STORE_SECRET, config.OBJECT_STORE_SECRET_ENV_VARS
    )
