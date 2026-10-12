import { ArrowLeft, CircleAlert, LoaderCircle } from "lucide-react";
import { useState } from "react";
import { Link, useParams } from "react-router-dom";
import { useApi } from "../api";
import { ENDPOINTS, endpointHistory, endpointStats } from "../apiPaths";
import {
  ENDPOINT_HISTORY_REFRESH_MS,
  ENDPOINT_HISTORY_RETENTION_DAYS,
  ENDPOINT_HISTORY_SPANS,
  ENDPOINT_STATS_REFRESH_MS,
  KV_CACHE_DANGER_PERCENT,
  KV_CACHE_WARNING_PERCENT,
  PERCENT_STATS,
} from "../config";
import CopyValue from "./CopyValue";
import {
  type Endpoint,
  type EndpointHistory,
  type EndpointStats,
  formatSeconds,
  type Histogram,
  type Point,
  type StatsValue,
  statsValues,
} from "./endpoint";
import LineChart, { type ChartSeries } from "./LineChart";

type Span = keyof typeof ENDPOINT_HISTORY_SPANS;
type Chart = {
  title: string;
  series: (ChartSeries & { points: (history: EndpointHistory) => Point[] })[];
  format: (value: number | null) => string;
  // Shown only once it has points, as speculative decoding may be off.
  optional?: boolean;
};

// Dot decimals, as everywhere else in the Web UI.
const count = (value: number | null) =>
  value == null ? "—" : value.toLocaleString("en", { maximumFractionDigits: 2 });
const perSecond = (value: number | null) => (value == null ? "—" : `${value.toFixed(1)}/s`);
const percent = (value: number | null) => (value == null ? "—" : `${(value * 100).toFixed(1)} %`);
const value = (key: string) => (history: EndpointHistory) => history.values[key] ?? [];
const latency = (key: string, statistic: string) => (history: EndpointHistory) =>
  history.latencies[key]?.[statistic] ?? [];
const latencyChart = (title: string, key: string): Chart => ({
  title,
  series: [
    { label: "p50", color: "--accent", points: latency(key, "p50") },
    { label: "p95", color: "--warning", points: latency(key, "p95") },
    { label: "Mean", color: "--info", points: latency(key, "mean") },
  ],
  format: formatSeconds,
});
// Stats per request.
const REQUEST_CHARTS: Chart[] = [
  latencyChart("Time to first token", "time_to_first_token"),
  latencyChart("Time per output token", "time_per_output_token"),
  latencyChart("End-to-end latency", "end_to_end"),
  latencyChart("Queue time", "queue_time"),
  {
    title: "Prefill and decode time",
    series: [
      { label: "Prefill p50", color: "--accent", points: latency("prefill_time", "p50") },
      { label: "Prefill p95", color: "--info", points: latency("prefill_time", "p95") },
      { label: "Prefill mean", color: "--text-muted", points: latency("prefill_time", "mean") },
      { label: "Decode p50", color: "--success", points: latency("decode_time", "p50") },
      { label: "Decode p95", color: "--warning", points: latency("decode_time", "p95") },
      { label: "Decode mean", color: "--danger", points: latency("decode_time", "mean") },
    ],
    format: formatSeconds,
  },
  {
    title: "Tokens/s",
    series: [
      { label: "Generated", color: "--accent", points: value("generation_tokens_per_second") },
      { label: "Prompt", color: "--info", points: value("prompt_tokens_per_second") },
    ],
    format: perSecond,
  },
  {
    title: "Finished requests/s",
    series: [{ label: "Finished", color: "--success", points: value("finished_per_second") }],
    format: perSecond,
  },
];
// The state of the vLLM server.
const SERVER_CHARTS: Chart[] = [
  {
    title: "Running/waiting",
    series: [
      { label: "Running", color: "--accent", points: value("running") },
      { label: "Waiting", color: "--warning", points: value("waiting") },
    ],
    format: count,
  },
  {
    title: "KV-cache usage",
    series: [{ label: "Usage", color: "--accent", points: value("kv_cache_usage") }],
    format: percent,
  },
  {
    title: "Preemptions/s",
    series: [{ label: "Preemptions", color: "--danger", points: value("preemptions_per_second") }],
    format: perSecond,
  },
  {
    title: "Prefix-cache hit rate",
    series: [{ label: "Hit rate", color: "--success", points: value("prefix_cache_hit_rate") }],
    format: percent,
  },
  {
    title: "Speculative acceptance rate",
    series: [{ label: "Accepted", color: "--success", points: value("acceptance_rate") }],
    format: percent,
    optional: true,
  },
];

export default function EndpointStatsPage() {
  const { name = "" } = useParams();
  const [span, setSpan] = useState<Span>("live");
  const history = useApi<EndpointHistory>(
    endpointHistory(name, span),
    span === "live" ? ENDPOINT_STATS_REFRESH_MS : ENDPOINT_HISTORY_REFRESH_MS,
  );
  const stats = useApi<EndpointStats>(endpointStats(name), ENDPOINT_STATS_REFRESH_MS);
  // A stopped Endpoint keeps its row, and its name may be running again.
  const endpoint = useApi<Endpoint[]>(ENDPOINTS).data?.find(
    (found) => found.name === name && found.status !== "stopped",
  );
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
            The charts show the stored history of this Endpoint name, kept{" "}
            {ENDPOINT_HISTORY_RETENTION_DAYS} days; the values below them are live from its vLLM.
          </p>
          {endpoint && (
            <dl className="endpoint-access">
              <dt>URL</dt>
              <dd>
                <CopyValue label="URL" value={endpoint.url} />
              </dd>
              <dt>Endpoint Key</dt>
              <dd>
                <CopyValue label="Endpoint Key" value={endpoint.key} />
              </dd>
            </dl>
          )}
        </div>
      </header>

      <fieldset className="span-picker" aria-label="Timespan">
        {Object.entries(ENDPOINT_HISTORY_SPANS).map(([key, label]) => (
          <button
            key={key}
            type="button"
            className={`button small ${key === span ? "primary" : "ghost"}`}
            aria-pressed={key === span}
            onClick={() => setSpan(key as Span)}
          >
            {label}
          </button>
        ))}
      </fieldset>

      {history.error && (
        <p className="banner danger">
          <CircleAlert size={16} /> {history.error.message}
        </p>
      )}
      {!history.data && !history.error && <LoaderCircle className="spin" size={16} />}
      {history.data && (
        <>
          <ChartArea title="Requests" charts={REQUEST_CHARTS} history={history.data} />
          <ChartArea title="Server" charts={SERVER_CHARTS} history={history.data} />
        </>
      )}

      <section className="storage-section">
        <h2>Live values</h2>
        {/* The API answers 409 while the Endpoint isn't running. */}
        {stats.error?.status === 409 && (
          <p className="muted">Live values appear while the Endpoint runs.</p>
        )}
        {stats.error && stats.error.status !== 409 && (
          <p className="banner danger">
            <CircleAlert size={16} /> {stats.error.message}
          </p>
        )}
        {values?.kv_cache_usage && <KvCacheBar usage={values.kv_cache_usage} />}
        {stats.data && <LiveValues stats={stats.data} />}
      </section>
    </>
  );
}

function ChartArea({
  title,
  charts,
  history,
}: {
  title: string;
  charts: Chart[];
  history: EndpointHistory;
}) {
  return (
    <section className="storage-section">
      <h2>{title}</h2>
      <div className="charts">
        {charts.map((chart) => (
          <ChartOfHistory key={chart.title} chart={chart} history={history} />
        ))}
      </div>
    </section>
  );
}

function ChartOfHistory({ chart, history }: { chart: Chart; history: EndpointHistory }) {
  // One time per Server Sample, which all log `running`; a series without a point there has a gap.
  const points = chart.series.map((line) => new Map(line.points(history)));
  const logged = [history.values.running ?? [], ...points.map((found) => [...found])];
  const times = [...new Set(logged.flat().map(([time]) => time))].sort((a, b) => a - b);
  if (chart.optional && !times.length) return null;
  return (
    <LineChart
      title={chart.title}
      series={chart.series}
      times={times}
      values={points.map((found) => times.map((time) => found.get(time) ?? null))}
      span={[history.start, history.end]}
      format={chart.format}
    />
  );
}

function LiveValues({ stats }: { stats: EndpointStats }) {
  return (
    <>
      <div className="stats-sections">
        {stats.sections.map((section) => (
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
              {stats.metrics.map((metric) => (
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
