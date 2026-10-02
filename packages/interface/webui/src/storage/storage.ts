// What `GET /datasets`, `GET /models`, `GET /storage`, `GET /cache/base-models` and the model
// upload routes answer.
export type Dataset = {
  name: string;
  versions: { version: number; size_bytes: number; row_format: string }[];
};

export type RegisteredModel = {
  name: string;
  versions: { version: number; size_bytes: number; tags: Record<string, string> }[];
};

export type Storage = {
  buckets: { name: string; size_bytes: number }[];
  capacity_bytes: number;
};

export type CachedBaseModels = {
  base_models: { reference: string; size_bytes: number; last_used: string }[];
  capacity_bytes: number;
};

export type ModelVersionFile = { path: string; size_bytes: number; url: string };

export type StartedModelUpload = {
  id: string;
  part_size_bytes: number;
  files: { path: string; part_urls: string[] }[];
};

export type NewModelVersion = { name: string; version: number };
