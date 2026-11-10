import os
from dataclasses import dataclass


@dataclass(frozen=True)
class EndpointEnvironment:
    """Where Endpoints run and what with, from the API's environment; tests make their own."""

    namespace: str
    domain: str
    vllm_image: str
    stages_image: str
    model_cache_host_path: str
    object_store_url: str
    mlflow_url: str
    # Where the API reaches the Endpoint Proxy inside the cluster, as the Smoke Test does.
    proxy_url: str
    # The Kueue LocalQueue an Endpoint waits in for its GPUs, and the one a Smoke Test's wait in.
    gpu_queue: str
    smoke_test_gpu_queue: str

    @classmethod
    def from_environment(cls) -> "EndpointEnvironment":
        return cls(
            namespace=os.environ["PLATFORM_NAMESPACE"],
            domain=os.environ["DOMAIN"],
            vllm_image=os.environ["VLLM_IMAGE"],
            stages_image=os.environ["STAGES_IMAGE"],
            model_cache_host_path=os.environ["MODEL_CACHE_HOST_PATH"],
            object_store_url=os.environ["S3_ENDPOINT_URL"],
            mlflow_url=os.environ["MLFLOW_URL"],
            proxy_url=os.environ["PROXY_URL"],
            gpu_queue=os.environ["GPU_QUEUE"],
            smoke_test_gpu_queue=os.environ["SMOKE_TEST_GPU_QUEUE"],
        )
