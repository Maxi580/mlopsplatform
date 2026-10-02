// Route paths of the platform API; mirrors packages/core/src/mlp_core/api_paths.py.
export const LOGIN = "/auth/login";
export const SCHEMA = "/schema";
export const SETTINGS = "/settings";
export const VALIDATE_PIPELINE = "/pipelines/validate";
export const DATASETS = "/datasets";
export const PIPELINES = "/pipelines";
export const cancelPipeline = (id: number) => `/pipelines/${id}/cancel`;
export const datasetVersion = (name: string, version: number) =>
  `/datasets/${name}/versions/${version}`;
export const datasetDownload = (name: string, version: number) =>
  `${datasetVersion(name, version)}/download`;
export const MODELS = "/models";
export const modelVersion = (name: string, version: number) =>
  `/models/${name}/versions/${version}`;
export const STORAGE = "/storage";
