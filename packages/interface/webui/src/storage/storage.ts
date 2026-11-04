// What `GET /datasets`, `GET /models`, `GET /storage`, `GET /storage/gpus`, `GET /cache`,
// `GET /checkpoints` and the model upload routes answer.
export type Dataset = {
  name: string;
  versions: { version: number; size_bytes: number; row_format: string }[];
};

export type RegisteredModel = {
  name: string;
  versions: {
    version: number;
    size_bytes: number;
    tags: Record<string, string>;
    // Null for an uploaded version, or once its Pipeline is deleted or has no MLflow Run yet.
    pipeline_mlflow_run_url: string | null;
  }[];
};

export type Storage = {
  buckets: { name: string; size_bytes: number }[];
  capacity_bytes: number;
};

export type GpuReading = {
  index: number;
  name: string;
  memory_total_bytes: number;
  memory_used_bytes: number;
  memory_free_bytes: number;
  // The GPU has no memory of its own to report, so these are the machine's.
  shared_with_system: boolean;
  // Null if the GPU doesn't report it.
  utilization_percent: number | null;
};

export type GpuMachines = {
  machines: { name: string; reserved_percent: number; gpus: GpuReading[] }[];
};

// A Phase's newest Checkpoint, kept after its Pipeline failed or was cancelled.
export type Checkpoint = {
  pipeline_id: number;
  // Unknown once its Pipeline's row is gone.
  pipeline_name: string | null;
  phase_index: number;
  size_bytes: number;
};

export type ModelCache = {
  entries: {
    kind: "base_model" | "benchmark";
    reference: string;
    size_bytes: number;
    last_used: string;
  }[];
  capacity_bytes: number;
};

export type ModelVersionFile = { path: string; size_bytes: number; url: string };

export type StartedModelUpload = {
  id: string;
  part_size_bytes: number;
  files: { path: string; part_urls: string[] }[];
};

export type NewModelVersion = { name: string; version: number };
