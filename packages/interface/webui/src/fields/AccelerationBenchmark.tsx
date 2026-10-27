// What the API publishes as `acceleration_benchmark`: by form field, the options it measured.
export type BenchmarkRow = { label: string; small: string; medium: string };
export type BenchmarkResult = { note?: string; rows: BenchmarkRow[] };
export type AccelerationBenchmark = {
  intro: string;
  terms: string;
  results: Record<string, Record<string, BenchmarkResult>>;
};

/** What the benchmark compared, and how to read its results. */
export function BenchmarkIntro({ benchmark }: { benchmark: AccelerationBenchmark }) {
  return (
    <div className="benchmark-intro">
      <p className="muted">{benchmark.intro}</p>
      <p className="muted">{benchmark.terms}</p>
    </div>
  );
}

/** The option's results by model size, or that it wasn't measured; nothing for an unmeasured
 * default, e.g. `auto`, which is no choice to judge. */
export function BenchmarkResults({
  results,
  option,
  fallback,
}: {
  results: Record<string, BenchmarkResult>;
  option: string;
  fallback?: string;
}) {
  const result = results[option];
  if (!result)
    return option && option !== fallback ? (
      <p className="muted">Not measured in the benchmark</p>
    ) : null;
  return (
    <table className="benchmark-results" aria-label={`Benchmark results of ${option}`}>
      <thead>
        <tr>
          <th>{result.note}</th>
          <th>Small</th>
          <th>Medium</th>
        </tr>
      </thead>
      <tbody>
        {result.rows.map((row) => (
          <tr key={row.label}>
            <th scope="row">{row.label}</th>
            {row.small === row.medium ? (
              <td colSpan={2}>{row.small}</td>
            ) : (
              <>
                <td>{row.small}</td>
                <td>{row.medium}</td>
              </>
            )}
          </tr>
        ))}
      </tbody>
    </table>
  );
}
