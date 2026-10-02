from google.protobuf import json_format
from kfp import dsl, kubernetes

from mlp_core import config
from mlp_core.pipeline_request.schema import PipelineRequest


def compile_pipeline(
    pipeline_id: int, request: PipelineRequest, stages_image: str, model_cache_pvc: str
) -> dict:
    """The Kubeflow pipeline spec for a resolved request, in the form KFP's run API takes."""

    @dsl.container_component
    def fetch(base_model: str):
        return dsl.ContainerSpec(
            image=stages_image, command=["mlp-stage", "fetch"], args=[base_model]
        )

    @dsl.container_component
    def cleanup():
        return dsl.ContainerSpec(image=stages_image, command=["mlp-stage", "cleanup"])

    @dsl.pipeline(name=request.name)
    def pipeline():
        cleanup_task = cleanup().set_caching_options(False)
        with dsl.ExitHandler(cleanup_task):
            fetch_task = fetch(base_model=request.finetune.base_model)
            fetch_task.set_caching_options(False)
            fetch_task.set_env_variable("HF_HOME", config.MODEL_CACHE_PATH)
            kubernetes.mount_pvc(fetch_task, model_cache_pvc, config.MODEL_CACHE_PATH)
            # Only fetch gets the token, from the Secret; a pipeline parameter would be logged.
            kubernetes.use_secret_as_env(
                fetch_task,
                config.PIPELINE_SECRET_NAME.format(id=pipeline_id),
                {"hf_token": config.SECRET_ENV_VARS["hf_token"]},
                optional=True,
            )

    return {
        "pipeline_spec": json_format.MessageToDict(pipeline.pipeline_spec),
        "platform_spec": json_format.MessageToDict(pipeline.platform_spec),
    }
