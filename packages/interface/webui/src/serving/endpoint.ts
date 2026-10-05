// What `GET /endpoints` answers for each Endpoint.
export type Endpoint = {
  name: string;
  owner: string;
  model: string;
  status: string;
  url: string;
  created_at: string;
  // Null while it is pending, loading or unreachable.
  stats: EndpointStatsSummary | null;
};

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

export function formatSeconds(seconds: number | null | undefined): string {
  if (seconds == null) return "—";
  return seconds < 1 ? `${Math.round(seconds * 1000)} ms` : `${seconds.toFixed(2)} s`;
}
