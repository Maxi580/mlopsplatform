import { ArrowLeft, CircleAlert, LoaderCircle } from "lucide-react";
import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { useApi } from "../api";
import { endpointStats } from "../apiPaths";
import {
  ENDPOINT_STATS_HISTORY_MS,
  ENDPOINT_STATS_REFRESH_MS,
  KV_CACHE_DANGER_PERCENT,
  KV_CACHE_WARNING_PERCENT,
  MS_PER_MINUTE,
  PERCENT_STATS,
} from "../config";
import {
  type EndpointStats,
  formatSeconds,
  type Histogram,
  type StatsValue,
  statsValues,
} from "./endpoint";
import LineChart, { type ChartSeries } from "./LineChart";

// What the charts need of one reading: its time, gauges, and the counters they take rates of.
type Reading = { at: number; values: Record<string, number | null> };
type Chart = {
  title: string;
  series: (ChartSeries & { point: (earlier: Reading, later: Reading) => number | null })[];
  format: (value: number | null) => string;
};

// Dot decimals, as everywhere else in the Web UI.
const count = (value: number | null) =>
  value == null ? "—" : value.toLocaleString("en", { maximumFractionDigits: 2 });
const perSecond = (value: number | null) => (value == null ? "—" : `${value.toFixed(1)}/s`);
const percent = (value: number | null) => (value == null ? "—" : `${(value * 100).toFixed(1)} %`);
const gauge = (key: string) => (_: Reading, later: Reading) => later.values[key];
// The change in `part` over the change in `whole` between two readings, e.g. tokens per second.
const rate = (part: string, whole?: string) => (earlier: Reading, later: Reading) => {
  const change = (key: string) => (later.values[key] ?? 0) - (earlier.values[key] ?? 0);
  const divisor = whole ? change(whole) : (later.at - earlier.at) / 1000;
  return divisor > 0 ? change(part) / divisor : null;
};
const CHARTS: Chart[] = [
  {
    title: "Requests",
    series: [
      { label: "Running", color: "--accent", point: gauge("running") },
      { label: "Waiting", color: "--warning", point: gauge("waiting") },
    ],
    format: count,
  },
  {
    title: "Tokens/s",
    series: [
      { label: "Generated", color: "--accent", point: rate("generation_tokens") },
      { label: "Prompt", color: "--info", point: rate("prompt_tokens") },
    ],
    format: perSecond,
  },
  {
    title: "Latency (mean per interval)",
    series: [
      {
        label: "Time to first token",
        color: "--accent",
        point: rate("time_to_first_token.sum", "time_to_first_token.count"),
      },
      {
        label: "Time per output token",
        color: "--info",
        point: rate("time_per_output_token.sum", "time_per_output_token.count"),
      },
    ],
    format: formatSeconds,
  },
  {
    title: "Prefix-cache hit rate",
    series: [
      {
        label: "Hit rate",
        color: "--success",
        point: rate("prefix_cache_hits", "prefix_cache_queries"),
      },
    ],
    format: percent,
  },
];

export default function EndpointStatsPage() {
  const { name = "" } = useParams();
  const stats = useApi<EndpointStats>(endpointStats(name), ENDPOINT_STATS_REFRESH_MS);
  const readings = useReadings(stats.data);
  const values = stats.data && statsValues(stats.data);

  return (
    <>
      <header className="page-header">
        <div>
          <Link to="/serving" className="muted">
            <ArrowLeft size={14} /> Serving
          </Link>
          <h1>{name}</h1>
          <p className="muted">
            Live from the Endpoint's vLLM, counting every request it serves. Charts fill while this
            page stays open, for up to {ENDPOINT_STATS_HISTORY_MS / MS_PER_MINUTE} minutes.
          </p>
        </div>
      </header>

      {stats.error && (
        <p className="banner danger">
          <CircleAlert size={16} /> {stats.error.message}
        </p>
      )}
      {!stats.data && !stats.error && <LoaderCircle className="spin" size={16} />}

      {values?.kv_cache_usage && <KvCacheBar usage={values.kv_cache_usage} />}

      {stats.data && (
        <>
          <div className="charts">
            {CHARTS.map((chart) => (
              <ChartOfReadings key={chart.title} chart={chart} readings={readings} />
            ))}
          </div>

          <div className="stats-sections">
            {stats.data.sections.map((section) => (
              <section key={section.title} className="card stats-section">
                <h2>{section.title}</h2>
                <dl>
                  {section.values.map((value) => (
                    <StatsValueView key={value.key} value={value} />
                  ))}
                </dl>
              </section>
            ))}
          </div>

          <section className="storage-section">
            <h2>All metrics</h2>
            <div className="card table-card">
              <table>
                <thead>
                  <tr>
                    <th>Metric</th>
                    <th>Value</th>
                    <th>What vLLM says it is</th>
                  </tr>
                </thead>
                <tbody>
                  {stats.data.metrics.map((metric) => (
                    <tr key={metric.name}>
                      <td className="mono">{metric.name}</td>
                      <td>
                        {metric.histogram
                          ? histogramText(
                              metric.histogram,
                              metric.name.endsWith("_seconds") ? formatSeconds : count,
                            )
                          : count(metric.value ?? null)}
                      </td>
                      <td className="muted">{metric.help}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>
        </>
      )}
    </>
  );
}

/** The readings of the last ENDPOINT_STATS_HISTORY_MS, one per answer. */
function useReadings(stats?: EndpointStats): Reading[] {
  const [readings, setReadings] = useState<Reading[]>([]);
  useEffect(() => {
    if (!stats) return;
    const values: Reading["values"] = {};
    for (const value of Object.values(statsValues(stats))) {
      values[value.key] = value.value ?? null;
      if (value.histogram) {
        values[`${value.key}.sum`] = value.histogram.sum;
        values[`${value.key}.count`] = value.histogram.count;
      }
    }
    const at = Date.parse(stats.read_at);
    setReadings((earlier) => [
      ...earlier.filter(
        (reading) => reading.at > at - ENDPOINT_STATS_HISTORY_MS && reading.at < at,
      ),
      { at, values },
    ]);
  }, [stats]);
  return readings;
}

function ChartOfReadings({ chart, readings }: { chart: Chart; readings: Reading[] }) {
  // Each point is taken between a reading and the one before it.
  const pairs = readings.slice(1).map((later, index) => [readings[index], later] as const);
  return (
    <LineChart
      title={chart.title}
      series={chart.series}
      times={pairs.map(([, later]) => later.at / 1000)}
      values={chart.series.map((line) =>
        pairs.map(([earlier, later]) => line.point(earlier, later)),
      )}
      format={chart.format}
    />
  );
}

function KvCacheBar({ usage }: { usage: StatsValue }) {
  const used = (usage.value ?? 0) * 100;
  return (
    <section className="card usage">
      <div className="usage-heading">
        <h2>KV cache usage</h2>
        <strong>{used.toFixed(1)} %</strong>
      </div>
      {/* The browser colours it green below `low`, amber up to `high` and red past it. */}
      <meter
        aria-label="KV cache usage"
        min={0}
        max={100}
        low={KV_CACHE_WARNING_PERCENT}
        high={KV_CACHE_DANGER_PERCENT}
        optimum={0}
        value={used}
      />
      <p className="muted">{usage.explanation}</p>
    </section>
  );
}

function StatsValueView({ value }: { value: StatsValue }) {
  // The curated histograms are latencies, in seconds.
  const shown = value.histogram
    ? histogramText(value.histogram, formatSeconds)
    : PERCENT_STATS.includes(value.key)
      ? percent(value.value ?? null)
      : count(value.value ?? null);
  return (
    <div className="stats-value">
      <dt>{value.label}</dt>
      <dd>
        <strong>{shown}</strong>
        <span className="muted">{value.explanation}</span>
      </dd>
    </div>
  );
}

function histogramText(histogram: Histogram, format: (value: number | null) => string): string {
  if (!histogram.count) return "no requests yet";
  const { p50, p95, mean } = histogram;
  return `p50 ${format(p50)} · p95 ${format(p95)} · mean ${format(mean)}`;
}
