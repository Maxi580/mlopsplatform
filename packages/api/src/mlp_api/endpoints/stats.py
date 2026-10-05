import logging
import math
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime

from prometheus_client.metrics_core import Metric
from prometheus_client.parser import text_string_to_metric_families

from mlp_api.pipelines.cluster import Cluster
from mlp_core import config

logger = logging.getLogger(__name__)


def fetch_endpoint_stats(cluster: Cluster, names: list[str]) -> dict[str, dict | None]:
    """Each running Endpoint's full stats, read from all of them at once; None if unreachable."""
    with ThreadPoolExecutor(max(len(names), 1)) as pool:
        read = pool.map(lambda name: read_stats(cluster, name), names)
        return dict(zip(names, read, strict=True))


def read_stats(cluster: Cluster, name: str) -> dict | None:
    try:
        text = cluster.endpoint_metrics(name)
    except Exception as error:
        logger.info("Endpoint %s has no metrics: %s", name, error)
        return None
    return endpoint_stats(text, datetime.now(UTC))


def endpoint_stats(text: str, read_at: datetime) -> dict:
    """The curated sections, histograms as quantiles, and every metric with vLLM's HELP text."""
    # 1. Every metric, summed over its series, e.g. one per engine or finish reason; a `_created`
    # family only says when another metric started.
    metrics = {
        family.name: metric_summary(family)
        for family in text_string_to_metric_families(text)
        if not family.name.endswith("_created")
    }

    # 2. The curated values vLLM exports, each with its label and explanation.
    sections = []
    for title, values in config.ENDPOINT_STATS_SECTIONS.items():
        curated = []
        for key, (metric, label, explanation) in values.items():
            value = curated_value(metrics, metric)
            if value is not None:
                curated.append({"key": key, "label": label, "explanation": explanation, **value})
        if curated:
            sections.append({"title": title, "values": curated})
    return {"read_at": read_at.isoformat(), "sections": sections, "metrics": list(metrics.values())}


def metric_summary(family: Metric) -> dict:
    """The metric's value, or for a histogram its quantiles, mean, sum and count; a summary has
    no buckets, so no quantiles."""
    summary = {"name": family.name, "type": family.type, "help": family.documentation}
    if family.type not in ("histogram", "summary"):
        names = (family.name, f"{family.name}_total")
        summary["value"] = sum(s.value for s in family.samples if s.name in names)
        return summary
    buckets: dict[float, float] = {}
    for sample in family.samples:
        if sample.name.endswith("_bucket"):
            bound = float(sample.labels["le"])
            buckets[bound] = buckets.get(bound, 0) + sample.value
    total = sum(s.value for s in family.samples if s.name.endswith("_sum"))
    count = sum(s.value for s in family.samples if s.name.endswith("_count"))
    cumulative = sorted(buckets.items())
    summary["histogram"] = {
        **{key: quantile(q, cumulative) for key, q in config.ENDPOINT_STATS_QUANTILES.items()},
        "mean": total / count if count else None,
        "sum": total,
        "count": count,
    }
    return summary


def quantile(q: float, cumulative: list[tuple[float, float]]) -> float | None:
    """Prometheus' `histogram_quantile`: linear inside the bucket the quantile falls into."""
    if not cumulative or cumulative[-1][1] == 0:
        return None
    rank = q * cumulative[-1][1]
    lower_bound, lower_count = 0.0, 0.0
    for bound, count in cumulative:
        if count >= rank:
            # Past the last finite bound, the best estimate is that bound.
            if math.isinf(bound):
                return lower_bound
            share = (rank - lower_count) / (count - lower_count)
            return lower_bound + (bound - lower_bound) * share
        lower_bound, lower_count = bound, count
    return lower_bound


def curated_value(metrics: dict[str, dict], metric: str | tuple[str, str]) -> dict | None:
    """The metric's value or histogram, or a rate of two counters; None if vLLM lacks one."""
    if isinstance(metric, tuple):
        part, whole = (metrics.get(name) for name in metric)
        if part is None or whole is None:
            return None
        return {"value": part["value"] / whole["value"] if whole["value"] else None}
    found = metrics.get(metric)
    if found is None:
        return None
    return {key: found[key] for key in ("value", "histogram") if key in found}


def endpoint_stats_summary(stats: dict | None) -> dict | None:
    """What the Endpoint list shows: its load, generated tokens for tokens/s, and TTFT p50."""
    if stats is None:
        return None
    values = curated_values(stats)
    ttft = values.get("time_to_first_token", {}).get("histogram", {})
    return {
        "running": values.get("running", {}).get("value"),
        "waiting": values.get("waiting", {}).get("value"),
        "generation_tokens": values.get("generation_tokens", {}).get("value"),
        "read_at": stats["read_at"],
        "time_to_first_token_p50": ttft.get("p50"),
    }


def curated_values(stats: dict) -> dict[str, dict]:
    """The curated values of every section, by key."""
    return {value["key"]: value for section in stats["sections"] for value in section["values"]}
