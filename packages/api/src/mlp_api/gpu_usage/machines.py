from mlp_api.pipelines.cluster import Cluster
from mlp_core.settings import Settings


def gpu_machines(cluster: Cluster, settings: Settings) -> list[dict]:
    """Each machine that answers, with its GPUs' readings and how much of them jobs reserved."""
    slices = cluster.machine_gpu_slices()
    return [
        {
            "name": machine,
            "reserved_percent": round(
                100 * settings.gpus_of_request(slices.get(machine, 0)) / len(gpus)
            ),
            "gpus": gpus,
        }
        for machine, gpus in sorted(cluster.machine_gpu_readings().items())
        if gpus
    ]
