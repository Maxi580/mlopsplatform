// A Pipeline as `GET /pipelines` lists it.
export type Pipeline = {
  id: number;
  name: string;
  owner: string;
  status: string;
  stages: string[];
  created_at: string;
  kubeflow_run_url: string | null;
  mlflow_run_url: string | null;
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
