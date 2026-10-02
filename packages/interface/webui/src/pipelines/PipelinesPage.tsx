import { ChartLine, CircleAlert, Plus, Workflow } from "lucide-react";
import { useState } from "react";
import { Link, useLocation } from "react-router-dom";
import { callApi, errorMessage, useApi } from "../api";
import { PIPELINES, cancelPipeline } from "../apiPaths";
import { FINISHED_STATUSES, PIPELINE_LIST_REFRESH_MS } from "../config";
import StatusBadge from "./StatusBadge";
import type { Pipeline } from "./pipeline";

export default function PipelinesPage() {
  const { data: pipelines, error, loadedAt, reload } = useApi<Pipeline[]>(
    PIPELINES,
    PIPELINE_LIST_REFRESH_MS,
  );
  const submitted: number | undefined = useLocation().state?.submitted;
  const [notice, setNotice] = useState("");

  async function cancel(pipeline: Pipeline) {
    try {
      await callApi(cancelPipeline(pipeline.id), {});
      setNotice(`Cancelled Pipeline ${pipeline.id}`);
    } catch (failure) {
      setNotice(errorMessage(failure));
    }
    reload();
  }

  return (
    <>
      <header className="page-header">
        <div>
          <h1>Pipelines</h1>
          <p className="muted">{summary(pipelines)}</p>
        </div>
        <div className="header-actions">
          <span className="live" title={loadedAt && `Updated ${loadedAt.toLocaleTimeString()}`}>
            <span className="live-dot" /> Live
          </span>
          <Link className="button primary" to="/pipelines/new">
            <Plus size={16} /> New Pipeline
          </Link>
        </div>
      </header>

      {submitted !== undefined && (
        <p className="banner success">Submitted Pipeline {submitted}. It starts as soon as a GPU is free.</p>
      )}
      {notice && <p className="banner">{notice}</p>}
      {error && (
        <p className="banner danger">
          <CircleAlert size={16} /> {error.message}
        </p>
      )}

      {pipelines?.length === 0 && (
        <section className="empty card">
          <Workflow size={32} />
          <h2>No Pipelines yet</h2>
          <p className="muted">Start one here or with `mlp run`.</p>
          <Link className="button primary" to="/pipelines/new">
            <Plus size={16} /> New Pipeline
          </Link>
        </section>
      )}

      {!!pipelines?.length && (
        <div className="card table-card">
          <table>
            <thead>
              <tr>
                <th>Pipeline</th>
                <th>Status</th>
                <th>Stages</th>
                <th>Owner</th>
                <th>Created</th>
                <th>Links</th>
                <th aria-label="Actions" />
              </tr>
            </thead>
            <tbody>
              {pipelines.map((pipeline) => (
                <PipelineRow
                  key={pipeline.id}
                  pipeline={pipeline}
                  highlighted={pipeline.id === submitted}
                  onCancel={cancel}
                />
              ))}
            </tbody>
          </table>
        </div>
      )}
    </>
  );
}

function PipelineRow({
  pipeline,
  highlighted,
  onCancel,
}: {
  pipeline: Pipeline;
  highlighted: boolean;
  onCancel: (pipeline: Pipeline) => void;
}) {
  // Cancelling asks once more in place, rather than in a browser dialog.
  const [confirming, setConfirming] = useState(false);
  const created = new Date(pipeline.created_at);

  return (
    <tr className={highlighted ? "highlighted" : undefined}>
      <td>
        <span className="pipeline-name">{pipeline.name}</span>
        <span className="muted mono">#{pipeline.id}</span>
      </td>
      <td>
        <StatusBadge status={pipeline.status} />
      </td>
      <td>
        {pipeline.stages.map((stage) => (
          <span key={stage} className="chip">
            {stage}
          </span>
        ))}
      </td>
      <td>{pipeline.owner}</td>
      <td title={created.toLocaleString()}>{timeAgo(created)}</td>
      <td>
        <div className="links">
        {pipeline.kubeflow_run_url && (
          <a className="link-button" href={pipeline.kubeflow_run_url} target="_blank" rel="noreferrer">
            <Workflow size={14} /> Kubeflow
          </a>
        )}
        {pipeline.mlflow_run_url && (
          <a className="link-button" href={pipeline.mlflow_run_url} target="_blank" rel="noreferrer">
            <ChartLine size={14} /> MLflow
          </a>
        )}
        </div>
      </td>
      <td className="actions">
        {!FINISHED_STATUSES.includes(pipeline.status) &&
          (confirming ? (
            <>
              <button className="button danger small" onClick={() => onCancel(pipeline)}>
                Cancel Pipeline
              </button>
              <button className="button ghost small" onClick={() => setConfirming(false)}>
                Keep
              </button>
            </>
          ) : (
            <button className="button ghost small" onClick={() => setConfirming(true)}>
              Cancel
            </button>
          ))}
      </td>
    </tr>
  );
}

function summary(pipelines?: Pipeline[]): string {
  if (!pipelines) return "Loading…";
  const counts = new Map<string, number>();
  for (const pipeline of pipelines) counts.set(pipeline.status, (counts.get(pipeline.status) ?? 0) + 1);
  const active = [...counts].filter(([status]) => !FINISHED_STATUSES.includes(status));
  const total = `${pipelines.length} Pipeline${pipelines.length === 1 ? "" : "s"}`;
  return active.length ? `${total} · ${active.map(([status, count]) => `${count} ${status}`).join(" · ")}` : total;
}

function timeAgo(date: Date): string {
  const minutes = Math.round((Date.now() - date.getTime()) / 60000);
  if (minutes < 1) return "just now";
  if (minutes < 60) return `${minutes} min ago`;
  if (minutes < 24 * 60) return `${Math.round(minutes / 60)} h ago`;
  return date.toLocaleDateString();
}
