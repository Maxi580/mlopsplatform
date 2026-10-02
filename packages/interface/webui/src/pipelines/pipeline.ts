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
};
