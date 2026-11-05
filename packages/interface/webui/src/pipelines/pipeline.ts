import { formatShare } from "../fields/numbers";

export type GpuShares = Record<string, number>;

/** What each Stage holds, e.g. `GPU Share: finetune 30%, evaluate 25%`; empty if none does. */
export function gpuSharesSummary(shares: GpuShares = {}): string {
  const held = Object.entries(shares).map(([stage, share]) => `${stage} ${formatShare(share)}`);
  return held.length ? `GPU Share: ${held.join(", ")}` : "";
}

// A Pipeline as `GET /pipelines` lists it.
export type Pipeline = {
  id: number;
  name: string;
  owner: string;
  status: string;
  // Why it failed: a plain message or the last log lines.
  failure?: string | null;
  stages: string[];
  // Each Stage -> the GPU Share its steps hold.
  gpu_shares: GpuShares;
  created_at: string;
  kubeflow_run_url: string | null;
  mlflow_run_url: string | null;
  // A Smoke Test's results; it can't be resumed.
  cases?: unknown;
  // The best parameters its `sweep` Stage found, once it reported them.
  sweep?: SweepOutput | null;
};

export type SweepOutput = {
  parameters: Record<string, Record<string, unknown>>;
  objective: number;
};

// A benchmark of the catalog, as `GET /benchmarks` lists it.
export type Benchmark = {
  name: string;
  category: string;
  description: string;
  licence: string;
  size_bytes: number;
};
