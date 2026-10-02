// The KFP and MLflow UIs, behind the platform's login on the same domain.
export const KUBEFLOW_UI_URL = "/pipeline/";
export const MLFLOW_UI_URL = "/mlflow/";
export const PIPELINE_LIST_REFRESH_MS = 5000;
// Where the New Pipeline form keeps its draft across reloads; Secrets are never stored.
export const DRAFT_KEY = "mlp:new-pipeline";
export const FINISHED_STATUSES = ["succeeded", "failed", "cancelled"];
// The last part of the name under which a section's further TRL/PEFT settings are kept.
export const MORE_SETTINGS = "*";
// Secret slot -> its label; the values travel beside the Pipeline Request, never inside it.
export const SECRET_SLOTS = { hf_token: "Hugging Face token" };
// Hints for text fields whose schema pattern starts with a Reference prefix.
export const REFERENCE_PLACEHOLDERS: Record<string, string> = {
  "^hf:": "hf:org/name or hf:org/name@revision",
  "^dataset:": "dataset:name or dataset:name@version",
};
