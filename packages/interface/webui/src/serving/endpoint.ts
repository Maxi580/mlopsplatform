import type { AccelerationBenchmark } from "../fields/AccelerationBenchmark";
import { type FormSection, pipelineForm } from "../pipelines/pipelineForm";

// What `GET /endpoints` answers for each Endpoint.
export type Endpoint = {
  name: string;
  owner: string;
  model: string;
  status: string;
  // Why it crashed, until a restart: a plain message or the last log lines.
  failure?: string | null;
  url: string;
  // The Endpoint Key: the bearer token its URL takes.
  key: string;
  created_at: string;
  // The Judge, which the API runs while judging is enabled; users don't start or stop it.
  judge: boolean;
  // Whether the Judge scores every `every`th answer, at most `max_per_hour`.
  judging: Judging;
  // Null while it is waiting for a GPU, pending, loading or unreachable.
  stats: EndpointStatsSummary | null;
};

export type Judging = { judged: boolean; every: number; max_per_hour: number };

export type EndpointStatsSummary = {
  running: number;
  waiting: number;
  generation_tokens: number;
  read_at: string;
  time_to_first_token_p50: number | null;
};

// What `GET /endpoints/<name>/stats` answers.
export type EndpointStats = {
  read_at: string;
  sections: { title: string; values: StatsValue[] }[];
  metrics: Metric[];
};

export type StatsValue = {
  key: string;
  label: string;
  explanation: string;
  value?: number | null;
  histogram?: Histogram;
};

export type Histogram = {
  p50: number | null;
  p95: number | null;
  mean: number | null;
  sum: number;
  count: number;
};

// What `GET /endpoints/<name>/history` answers: per Server Sample key its [unix seconds, value]
// points, the latencies' p50, p95 and mean by their chart, and the Feedback votes summed per
// bucket of the span, and the Judge Scores averaged per bucket.
export type Point = [number, number];
export type EndpointHistory = {
  start: number;
  end: number;
  values: Record<string, Point[]>;
  latencies: Record<string, Record<string, Point[]>>;
  feedback: { up: Point[]; down: Point[] };
  judge_score: Point[];
};

// A row of `GET /endpoints/<name>/requests`: one Request Trace, times in seconds, the sampling
// settings null where the request left them unset, and its tags.
export type RequestTrace = {
  trace_id: string;
  arrived_at: string;
  // From its arrival at the Endpoint Proxy to the end of its answer.
  latency: number;
  end_to_end: number | null;
  time_to_first_token: number | null;
  queue_time: number | null;
  prefill_time: number | null;
  decode_time: number | null;
  prompt_tokens: number | null;
  completion_tokens: number | null;
  temperature: number | null;
  top_p: number | null;
  max_tokens: number | null;
  source: string | null;
  status: string | null;
  rating: "up" | "down" | null;
  judge_score: number | null;
  prompt_preview: string | null;
  mlflow_url: string;
};
export type EndpointRequests = { requests: RequestTrace[]; next_page_token: string | null };

// What `GET /endpoints/<name>/requests/<trace_id>` answers: the row with the full content.
export type Message = { role: string; content: unknown; tool_calls?: unknown[] };
export type RequestDetail = RequestTrace & {
  messages: Message[];
  tools: unknown[] | null;
  reasoning: string | null;
  answer: string | null;
  tool_calls: unknown[] | null;
  error: string | null;
  judge_reason: string | null;
};

export type Metric = {
  name: string;
  type: string;
  help: string;
  value?: number;
  histogram?: Histogram;
};

/** Per second between two readings of a counter; null without an earlier reading. */
export function ratePerSecond(
  earlier: { value: number; at: string } | undefined,
  later: { value: number; at: string },
): number | null {
  if (!earlier) return null;
  const seconds = (Date.parse(later.at) - Date.parse(earlier.at)) / 1000;
  return seconds > 0 ? Math.max(later.value - earlier.value, 0) / seconds : null;
}

/** Each curated value by its key. */
export function statsValues(stats: EndpointStats): Record<string, StatsValue> {
  return Object.fromEntries(
    stats.sections.flatMap((section) => section.values.map((value) => [value.key, value])),
  );
}

export type PublishedSchema = {
  $defs: Record<string, Record<string, unknown>>;
  acceleration_benchmark?: AccelerationBenchmark;
};

// The serving options of `evaluate` and Teachers are an Endpoint's, so the schema publishes them.
export function servingOptionsForm(schema: PublishedSchema): FormSection {
  const { $defs, acceleration_benchmark } = schema;
  // Each option keeps its results; the benchmark's intro and terms are for the Quantize section.
  const { benchmark: _, ...form } = pipelineForm({
    ...$defs.ServingOptions,
    $defs,
    acceleration_benchmark,
  });
  return form;
}

export function formatSeconds(seconds: number | null | undefined): string {
  if (seconds == null) return "—";
  return seconds < 1 ? `${Math.round(seconds * 1000)} ms` : `${seconds.toFixed(2)} s`;
}
