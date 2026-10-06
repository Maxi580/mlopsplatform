// Route paths of the platform API; mirrors packages/core/src/mlp_core/api_paths.py.
export const LOGIN = "/auth/login";
export const LOGOUT = "/auth/logout";
export const VERIFY = "/auth/verify";
export const SCHEMA = "/schema";
export const BENCHMARKS = "/benchmarks";
export const SETTINGS = "/settings";
export const VALIDATE_PIPELINE = "/pipelines/validate";
export const DATASETS = "/datasets";
export const PIPELINES = "/pipelines";
export const pipelineById = (id: number) => `/pipelines/${id}`;
export const cancelPipeline = (id: number) => `${pipelineById(id)}/cancel`;
export const datasetVersions = (name: string) => `/datasets/${name}/versions`;
export const datasetVersion = (name: string, version: number) =>
  `/datasets/${name}/versions/${version}`;
export const datasetDownload = (name: string, version: number) =>
  `${datasetVersion(name, version)}/download`;
export const BASE_MODELS = "/base-models";
export const MODELS = "/models";
export const modelVersion = (name: string, version: number) =>
  `/models/${name}/versions/${version}`;
export const modelVersionFiles = (name: string, version: number) =>
  `${modelVersion(name, version)}/files`;
export const MODEL_UPLOADS = "/models/uploads";
export const modelUploadComplete = (id: string) => `/models/uploads/${id}/complete`;
export const STORAGE = "/storage";
export const MODEL_CACHE = "/cache";
export const CHECKPOINTS = "/checkpoints";
export const checkpoint = (pipelineId: number, phaseIndex: number) =>
  `${CHECKPOINTS}/${pipelineId}/${phaseIndex}`;
export const ENDPOINTS = "/endpoints";
export const endpointByName = (name: string) => `/endpoints/${name}`;
export const stopEndpoint = (name: string) => `${endpointByName(name)}/stop`;
export const modelCacheEntry = (reference: string) =>
  `${MODEL_CACHE}?reference=${encodeURIComponent(reference)}`;
export const endpointStats = (name: string) => `/endpoints/${name}/stats`;
