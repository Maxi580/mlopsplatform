import os
from dataclasses import dataclass

from google.protobuf import json_format
from kfp import dsl, kubernetes

from mlp_core import config
from mlp_core.pipeline_request.schema import PipelineRequest


@dataclass(frozen=True)
class StepEnvironment:
    """The images and storage of the steps, from the API's environment; tests make their own."""

    stages_image: str
    # Training backend -> its trainer image.
    trainer_images: dict[str, str]
    model_cache_pvc: str
    object_store_url: str
    object_store_bucket: str

    @classmethod
    def from_environment(cls) -> "StepEnvironment":
        return cls(
            stages_image=os.environ["STAGES_IMAGE"],
            trainer_images={"hf": os.environ["TRAINER_HF_IMAGE"]},
            model_cache_pvc=os.environ["MODEL_CACHE_PVC"],
            object_store_url=os.environ["S3_ENDPOINT_URL"],
            object_store_bucket=os.environ["S3_BUCKET"],
        )


def compile_pipeline(
    pipeline_id: int, request: PipelineRequest, steps: StepEnvironment, gpus_per_stage: int
) -> dict:
    """The Kubeflow pipeline spec for a resolved request, in the form KFP's run API takes."""
    trainer_image = steps.trainer_images[request.finetune.backend]

    @dsl.container_component
    def fetch(base_model: str):
        return dsl.ContainerSpec(
            image=steps.stages_image, command=["mlp-stage", "fetch"], args=[base_model]
        )

    @dsl.container_component
    def finetune(pipeline_id: str, phase_index: str, request: str):
        return dsl.ContainerSpec(
            image=trainer_image,
            command=["mlp-stage", "finetune"],
            args=[pipeline_id, phase_index, request],
        )

    @dsl.container_component
    def cleanup():
        return dsl.ContainerSpec(image=steps.stages_image, command=["mlp-stage", "cleanup"])

    @dsl.pipeline(name=request.name)
    def pipeline():
        cleanup_task = cleanup().set_caching_options(False)
        with dsl.ExitHandler(cleanup_task):
            fetch_task = fetch(base_model=request.finetune.base_model)
            use_model_cache(fetch_task, steps)
            # Only fetch gets the token, from the Secret; a pipeline parameter would be logged.
            kubernetes.use_secret_as_env(
                fetch_task,
                config.PIPELINE_SECRET_NAME.format(id=pipeline_id),
                {"hf_token": config.SECRET_ENV_VARS["hf_token"]},
                optional=True,
            )

            # The request holds no Secret value, so it can be a parameter.
            finetune_task = finetune(
                pipeline_id=str(pipeline_id), phase_index="0", request=request.model_dump_json()
            ).after(fetch_task)
            use_model_cache(finetune_task, steps)
            finetune_task.set_env_variable("HF_HUB_OFFLINE", "1")
            finetune_task.set_accelerator_type(config.GPU_RESOURCE)
            finetune_task.set_accelerator_limit(gpus_per_stage)
            use_object_store(finetune_task, steps)

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
