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

    @classmethod
    def from_environment(cls) -> "EndpointEnvironment":
        return cls(
            namespace=os.environ["PLATFORM_NAMESPACE"],
            domain=os.environ["DOMAIN"],
            vllm_image=os.environ["VLLM_IMAGE"],
            stages_image=os.environ["STAGES_IMAGE"],
            model_cache_host_path=os.environ["MODEL_CACHE_HOST_PATH"],
            object_store_url=os.environ["S3_ENDPOINT_URL"],
        )
