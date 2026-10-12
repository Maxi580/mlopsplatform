import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime

from prometheus_client.parser import text_string_to_metric_families
from sqlalchemy import Engine, select

from mlp_api.endpoints.stats import Histogram, histogram, metric_summary, quantile
from mlp_api.endpoints.table import endpoint
from mlp_api.models.mlflow import MLflow
from mlp_core import config

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Reading:
    at: datetime
    # vLLM metric -> its value, or its histogram.
    values: dict[str, float | Histogram]
    # Whether nothing ran in the interval up to it, so a further idle sample is skipped.
    idle: bool


def sample_endpoint_stats(state, now: datetime) -> None:
    """Logs each running Endpoint's Server Sample since its previous reading to MLflow."""
    # 1. A stopped Endpoint's next reading is a baseline again.
    names = running_endpoint_names(state.engine)
    for name in set(state.server_readings) - set(names):
        del state.server_readings[name]
    # 2. Every running one, read at once.
    with ThreadPoolExecutor(max(len(names), 1)) as pool:
        list(pool.map(lambda name: sample_endpoint(state, name, now), names))


def running_endpoint_names(engine: Engine) -> list[str]:
    with engine.connect() as connection:
        query = select(endpoint.c.name).where(endpoint.c.status == "running")
        return list(connection.scalars(query))


def sample_endpoint(state, name: str, now: datetime) -> None:
    """Logs the Server Sample unless a baseline or a further idle one; failures are skipped."""
    try:
        # 1. The reading, kept for the next interval.
        values = server_reading(state.cluster.endpoint_metrics(name))
        previous = state.server_readings.get(name)
        current = Reading(now, values, previous is None or is_idle(previous.values, values))
        state.server_readings[name] = current
        if previous is None or (current.idle and previous.idle):
            return

        # 2. The sample, in the Run of the UTC day.
        sample = server_sample(previous, current)
        run_id = day_run_id(state, name, now)
        state.model_registry.log_metrics(run_id, sample, int(now.timestamp()), round_ms(now))
    except Exception:
        # The cached day Run may be gone, so the next sample looks it up again.
        state.endpoint_day_runs.pop(name, None)
        logger.exception("Sampling Endpoint %s's stats failed", name)


def server_reading(text: str) -> dict[str, float | Histogram]:
    """Each curated vLLM metric's value, or its histogram."""
    wanted = {name for metric in curated_metrics().values() for name in as_tuple(metric)}
    return {
        family.name: histogram(family)
        if family.type == "histogram"
        else metric_summary(family)["value"]
        for family in text_string_to_metric_families(text)
        if family.name in wanted
    }


def curated_metrics() -> dict[str, str | tuple[str, str]]:
    """Each curated value's vLLM metric, or pair of metrics, by its key."""
    return {
        key: value[0]
        for values in config.ENDPOINT_STATS_SECTIONS.values()
        for key, value in values.items()
    }


def as_tuple(metric: str | tuple[str, str]) -> tuple[str, ...]:
    return metric if isinstance(metric, tuple) else (metric,)


def is_idle(previous: dict, current: dict) -> bool:
    """Nothing runs or waits, and no counter changed."""
    metrics = curated_metrics()
    if current.get(metrics["running"]) or current.get(metrics["waiting"]):
        return False
    return all(
        counted(value) == counted(previous.get(name))
        for name, value in current.items()
        if name not in gauge_metrics()
    )


def gauge_metrics() -> set[str]:
    metrics = curated_metrics()
    return {metrics[key] for key in config.ENDPOINT_SAMPLE_GAUGES}


def counted(value: float | Histogram | None) -> float | None:
    return value.count if isinstance(value, Histogram) else value


def server_sample(previous: Reading, current: Reading) -> dict[str, float]:
    """The Server Sample of the interval between two readings."""
    metrics = curated_metrics()
    seconds = (current.at - previous.at).total_seconds()
    # A counter that went down restarted with its pod, so the interval counts from 0.
    earlier = {} if restarted(previous.values, current.values) else previous.values

    def change(metric: str) -> float:
        return current.values[metric] - earlier.get(metric, 0)

    # 1. Gauges as they are.
    sample = {
        key: current.values[metrics[key]]
        for key in config.ENDPOINT_SAMPLE_GAUGES
        if metrics[key] in current.values
    }
    # 2. Counters as rates per second.
    for key, counter in config.ENDPOINT_SAMPLE_RATES.items():
        if metrics[counter] in current.values:
            sample[key] = change(metrics[counter]) / seconds
    # 3. Pairs as the change of one over the other's, while that changes.
    for key, metric in metrics.items():
        if isinstance(metric, tuple) and all(name in current.values for name in metric):
            part, whole = (change(name) for name in metric)
            if whole:
                sample[key] = part / whole
    # 4. Latencies as the quantiles and mean of the requests the interval observed.
    for key, keys in latency_keys().items():
        if metrics[key] not in current.values:
            continue
        observed = interval_histogram(earlier.get(metrics[key]), current.values[metrics[key]])
        if observed.count:
            for name, q in config.ENDPOINT_STATS_QUANTILES.items():
                sample[keys[name]] = quantile(q, observed.cumulative)
            sample[keys["mean"]] = observed.total / observed.count
    return sample


def restarted(previous: dict, current: dict) -> bool:
    """Whether any counter or histogram count went down."""
    return any(
        name in previous and counted(value) < counted(previous[name])
        for name, value in current.items()
        if name not in gauge_metrics()
    )


def latency_keys() -> dict[str, dict[str, str]]:
    """Each Latency value's Server Sample keys, by its quantiles and mean."""
    statistics = [*config.ENDPOINT_STATS_QUANTILES, "mean"]
    return {
        key: {statistic: f"{key}.{statistic}" for statistic in statistics}
        for key in config.ENDPOINT_STATS_SECTIONS["Latency"]
    }


def interval_histogram(previous: Histogram | None, current: Histogram) -> Histogram:
    if previous is None:
        return current
    earlier = dict(previous.cumulative)
    return Histogram(
        cumulative=[(bound, count - earlier.get(bound, 0)) for bound, count in current.cumulative],
        total=current.total - previous.total,
        count=current.count - previous.count,
    )


def day_run_id(state, name: str, now: datetime) -> str:
    """The ID of the Endpoint name's Run of the UTC day, created if missing; cached per day."""
    # 1. The cached ID, while the day lasts.
    day = now.strftime(config.ENDPOINT_HISTORY_RUN_NAME)
    cached_day, cached_run_id = state.endpoint_day_runs.get(name, (None, None))
    if cached_day == day:
        return cached_run_id
    # 2. Else the day's Run in the Endpoint name's experiment, found or created.
    registry: MLflow = state.model_registry
    experiment_id = registry.experiment_id(config.ENDPOINT_HISTORY_EXPERIMENT.format(name=name))
    run_id = registry.run_ids(experiment_id).get(day)
    if run_id is None:
        run_id = registry.create_run(experiment_id, day, round_ms(now))
    state.endpoint_day_runs[name] = (day, run_id)
    return run_id


def round_ms(at: datetime) -> int:
    return round(at.timestamp() * 1000)


def endpoint_history(model_registry: MLflow, name: str, span: str, now: datetime) -> dict:
    """The span's start, end, and [unix seconds, value] points per key, latencies by chart."""
    # 1. The span's steps, and the day Runs it touches.
    start = now - config.ENDPOINT_HISTORY_SPANS[span]
    steps = (int(start.timestamp()), int(now.timestamp()))
    experiment = config.ENDPOINT_HISTORY_EXPERIMENT.format(name=name)
    experiment_id = model_registry.experiment_id(experiment, create=False)
    runs = model_registry.run_ids(experiment_id) if experiment_id else {}
    days = (
        start.strftime(config.ENDPOINT_HISTORY_RUN_NAME),
        now.strftime(config.ENDPOINT_HISTORY_RUN_NAME),
    )
    run_ids = [run_id for day, run_id in runs.items() if days[0] <= day <= days[1]]

    # 2. Every key's points, read at once.
    max_results = max(config.ENDPOINT_HISTORY_POINTS // max(len(run_ids), 1), 1)

    def points(key: str) -> list[list[float]]:
        if not run_ids:
            return []
        return [
            list(point) for point in model_registry.metric_history(run_ids, key, steps, max_results)
        ]

    keys = sample_keys()
    with ThreadPoolExecutor(len(keys)) as pool:
        values = dict(zip(keys, pool.map(points, keys), strict=True))

    # 3. The latencies under their chart.
    latencies = {
        key: {statistic: values.pop(found) for statistic, found in keys.items()}
        for key, keys in latency_keys().items()
    }
    return {"start": steps[0], "end": steps[1], "values": values, "latencies": latencies}


def sample_keys() -> list[str]:
    """Every key a Server Sample can hold."""
    pairs = [key for key, metric in curated_metrics().items() if isinstance(metric, tuple)]
    latencies = [found for keys in latency_keys().values() for found in keys.values()]
    return [*config.ENDPOINT_SAMPLE_GAUGES, *config.ENDPOINT_SAMPLE_RATES, *pairs, *latencies]
