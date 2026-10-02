// What `GET /datasets`, `GET /models`, `GET /storage` and the model upload routes answer.
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

export type ModelVersionFile = { path: string; size_bytes: number; url: string };

export type StartedModelUpload = {
  id: string;
  part_size_bytes: number;
  files: { path: string; part_urls: string[] }[];
};

export type NewModelVersion = { name: string; version: number };
