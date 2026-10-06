from mlp_api.endpoints.endpoint_model import EndpointModel
from mlp_api.endpoints.environment import EndpointEnvironment
from mlp_core import config
from mlp_core.endpoint_spec import EndpointSpec, vllm_args


def endpoint_manifests(
    name: str,
    endpoint_uuid: str,
    spec: EndpointSpec,
    model: EndpointModel,
    gpus: int,
    model_cache_size: str,
    environment: EndpointEnvironment,
) -> dict[str, dict]:
    """The Endpoint's Deployment, Service and Traefik route, by kind."""
    object_name = config.ENDPOINT_OBJECT_NAME.format(name=name)
    labels = {config.ENDPOINT_LABEL: name}
    metadata = {"name": object_name, "namespace": environment.namespace, "labels": labels}
    model_cache = {"name": "model-cache", "mountPath": config.MODEL_CACHE_PATH}
    models = {"name": "models", "mountPath": config.ENDPOINT_MODEL_DIRECTORY}

    # 1. Before vLLM starts: the Base Models into the Model Cache, and the Model Version files.
    init_containers = []
    if model.base_models:
        references = ",".join(model.base_models)
        init_containers.append(
            {
                "name": "fetch",
                "image": environment.stages_image,
                "command": ["mlp-stage", "fetch", model_cache_size, references],
                "env": [{"name": "HF_HOME", "value": config.MODEL_CACHE_PATH}],
                "volumeMounts": [model_cache],
            }
        )
    if model.downloads:
        files = [f"{source}={directory}" for source, directory in model.downloads.items()]
        keys = config.PLATFORM_OBJECT_STORE_SECRET_ENV_VARS.items()
        init_containers.append(
            {
                "name": "download",
                "image": environment.stages_image,
                "command": ["mlp-stage", "download", *files],
                "env": [
                    {"name": "S3_ENDPOINT_URL", "value": environment.object_store_url},
                    *(secret_env(variable, key) for key, variable in keys),
                ],
                "volumeMounts": [models],
            }
        )

    # 2. vLLM, offline, on whole GPUs; it reports ready once the model is loaded.
    vllm = {
        "name": "vllm",
        "image": environment.vllm_image,
        "command": ["vllm", "serve"],
        "args": vllm_args(spec, model.vllm, name, gpus, spec.speculative),
        "env": [
            {"name": "HF_HOME", "value": config.MODEL_CACHE_PATH},
            {"name": "HF_HUB_OFFLINE", "value": "1"},
        ],
        "ports": [{"containerPort": config.VLLM_PORT}],
        "resources": {"limits": {config.GPU_RESOURCE: gpus}},
        "readinessProbe": {"httpGet": {"path": "/health", "port": config.VLLM_PORT}},
        "volumeMounts": [model_cache, models, {"name": "shm", "mountPath": "/dev/shm"}],
    }
    deployment = {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": metadata,
        "spec": {
            "replicas": 1,
            # A new pod would wait for the GPUs the old one holds.
            "strategy": {"type": "Recreate"},
            "selector": {"matchLabels": labels},
            "template": {
                "metadata": {"labels": labels},
                "spec": {
                    "initContainers": init_containers,
                    "containers": [vllm],
                    "volumes": [
                        {
                            "name": "model-cache",
                            "hostPath": {
                                "path": environment.model_cache_host_path,
                                "type": "DirectoryOrCreate",
                            },
                        },
                        {"name": "models", "emptyDir": {}},
                        # vLLM's workers share tensors through it.
                        {"name": "shm", "emptyDir": {"medium": "Memory"}},
                    ],
                },
            },
        },
    }

    # 3. The Service, and the route to it behind the Endpoint Key, with the URL prefix stripped.
    service = {
        "apiVersion": "v1",
        "kind": "Service",
        "metadata": metadata,
        "spec": {"selector": labels, "ports": [{"port": config.VLLM_PORT}]},
    }
    path = config.ENDPOINT_PATH_PREFIX.format(uuid=endpoint_uuid)
    route = {
        "apiVersion": "traefik.io/v1alpha1",
        "kind": "IngressRoute",
        "metadata": metadata,
        "spec": {
            "entryPoints": ["websecure"],
            "tls": {},
            "routes": [
                {
                    "kind": "Rule",
                    "match": f"Host(`{environment.domain}`) && PathPrefix(`{path}`)",
                    "middlewares": [{"name": m} for m in config.ENDPOINT_ROUTE_MIDDLEWARES],
                    "services": [{"name": object_name, "port": config.VLLM_PORT}],
                }
            ],
        },
    }
    return {"deployment": deployment, "service": service, "route": route}


def secret_env(variable: str, key: str) -> dict:
    secret = {"name": config.PLATFORM_CREDENTIALS_SECRET, "key": key}
    return {"name": variable, "valueFrom": {"secretKeyRef": secret}}
