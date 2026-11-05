import logging

from mlp_api.pipelines.cluster import Cluster
from mlp_core.gpu_share import Gpu
from mlp_core.settings import Settings

logger = logging.getLogger(__name__)


def fullest_machine_reserved_percent(cluster: Cluster, settings: Settings) -> int | None:
    """How much of its GPUs jobs reserved on the fullest machine; None if no machine says."""
    try:
        slices = cluster.machine_gpu_slices()
        return max(
            (
                round(100 * settings.gpus_of_request(slices.get(machine, 0)) / len(gpus))
                for machine, gpus in cluster.machine_gpu_readings().items()
                if gpus
            ),
            default=None,
        )
    # A form works without the line that says it.
    except Exception:
        logger.warning("Can't read the GPU Shares reserved", exc_info=True)
        return None


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
