// The KFP and MLflow UIs, behind the platform's login on the same domain.
export const KUBEFLOW_UI_URL = "/pipeline/";
export const MLFLOW_UI_URL = "/mlflow/";
export const PIPELINE_LIST_REFRESH_MS = 5000;
export const ENDPOINT_LIST_REFRESH_MS = 5000;
// Where the New Pipeline form keeps its draft across reloads; Secrets are never stored.
export const DRAFT_KEY = "mlp:new-pipeline";
export const FINISHED_STATUSES = ["succeeded", "failed", "cancelled"];
// The last part of the name under which a section's further TRL/PEFT settings are kept.
export const MORE_SETTINGS = "*";
// Secret slot -> its label; the values travel beside the Pipeline Request, never inside it.
export const SECRET_SLOTS = {
  hf_token: "Hugging Face token",
  teacher_api_key: "Teacher API key",
};
// Hints for text fields whose schema pattern starts with a Reference prefix.
export const REFERENCE_PLACEHOLDERS: Record<string, string> = {
  "^hf:": "hf:org/name or hf:org/name@revision",
  "^dataset:": "dataset:name or dataset:name@version",
  "^model:": "model:name or model:name@version",
};
// Object store bucket -> what it holds, shown beside its usage.
export const BUCKET_CONTENTS: Record<string, string> = {
  platform: "Datasets and Checkpoints",
  mlflow: "Model Versions and Run artifacts",
  mlpipeline: "Kubeflow run outputs and logs",
};
// What a Model Cache entry holds, by its kind.
export const CACHE_ENTRY_KINDS = { base_model: "Base Model", benchmark: "Benchmark" };
// Object store usage, in percent, from which the meter shows a warning.
export const STORAGE_WARNING_PERCENT = 90;
// How long the New Pipeline form waits after a change before asking what the request downloads.
export const DOWNLOAD_PREVIEW_DELAY_MS = 400;
// The form's value under an optional section's name, e.g. the `serve` Stage, once switched on.
export const SWITCHED_ON = "on";
// An Endpoint's stats page reloads this often and charts the readings of this long while open.
export const ENDPOINT_STATS_REFRESH_MS = 5000;
export const MS_PER_MINUTE = 60 * 1000;
export const ENDPOINT_STATS_HISTORY_MS = 15 * MS_PER_MINUTE;
// KV cache usage, in percent, from which its bar turns amber and then red.
export const KV_CACHE_WARNING_PERCENT = 70;
export const KV_CACHE_DANGER_PERCENT = 90;
// Curated stats that are shares of a whole, shown in percent.
export const PERCENT_STATS = ["kv_cache_usage", "prefix_cache_hit_rate", "acceptance_rate"];
// The stats page's charts, and room for their y-axis labels such as "123 ms".
export const CHART_HEIGHT_PX = 180;
export const CHART_Y_AXIS_WIDTH_PX = 64;
