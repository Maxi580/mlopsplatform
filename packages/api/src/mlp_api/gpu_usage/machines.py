import logging

from mlp_api.pipelines.cluster import Cluster
from mlp_core.gpu_share import Gpu
from mlp_core.settings import Settings

logger = logging.getLogger(__name__)


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


def find_gpu(cluster: Cluster) -> Gpu | None:
    """The size of a GPU, as machines are identical; None if no reporter says."""
    try:
        readings = cluster.machine_gpu_readings().values()
        reading = next((gpus[0] for gpus in readings if gpus), None)
        return reading and Gpu(reading["memory_total_bytes"], reading["shared_with_system"])
    # Starting work matters more than its calculated share.
    except Exception:
        logger.warning("No GPU size to calculate GPU Shares with", exc_info=True)
        return None
