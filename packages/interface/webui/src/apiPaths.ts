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
export const modelVersionFiles = (name: string, version: number) =>
  `${modelVersion(name, version)}/files`;
export const MODEL_UPLOADS = "/models/uploads";
export const modelUploadComplete = (id: string) => `/models/uploads/${id}/complete`;
export const STORAGE = "/storage";
