import { ChartLine, CircleAlert, Play, Plus, Trash2, Workflow } from "lucide-react";
import { useState } from "react";
import { Link, useLocation } from "react-router-dom";
import { pathOnThisOrigin } from "../address";
import { callApi, errorMessage, useApi } from "../api";
import { cancelPipeline, PIPELINES, pipelineById, resumePipeline } from "../apiPaths";
import {
  FINISHED_STATUSES,
  PIPELINE_LIST_REFRESH_MS,
  RESUMABLE_STATUSES,
  SECRET_SLOTS,
  TRAINING_STAGES,
} from "../config";
import GpuShareField from "../fields/GpuShareField";
import SecretInput from "../fields/SecretInput";
import FailureRow from "./FailureRow";
import { gpuSharesSummary, type Pipeline, type SweepOutput } from "./pipeline";
import StatusBadge from "./StatusBadge";

export default function PipelinesPage() {
  const {
    data: pipelines,
    error,
    loadedAt,
    reload,
  } = useApi<Pipeline[]>(PIPELINES, PIPELINE_LIST_REFRESH_MS);
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

  async function remove(pipeline: Pipeline) {
    try {
      await callApi(pipelineById(pipeline.id), undefined, "DELETE");
      setNotice(`Deleted Pipeline ${pipeline.id}`);
    } catch (failure) {
      setNotice(errorMessage(failure));
    }
    reload();
  }

  async function resume(pipeline: Pipeline, body: ResumeBody) {
    try {
      const resumed = await callApi<{ id: number }>(resumePipeline(pipeline.id), body);
      setNotice(`Submitted Pipeline ${resumed.id}, which resumes Pipeline ${pipeline.id}`);
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
        <p className="banner success">
          Submitted Pipeline {submitted}. It starts as soon as a GPU is free.
        </p>
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
          <Link className="button primary empty-action" to="/pipelines/new">
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
                  onDelete={remove}
                  onResume={resume}
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
  onDelete,
  onResume,
}: {
  pipeline: Pipeline;
  highlighted: boolean;
  onCancel: (pipeline: Pipeline) => void;
  onDelete: (pipeline: Pipeline) => void;
  onResume: (pipeline: Pipeline, body: ResumeBody) => void;
}) {
  // Cancelling asks once more in place, rather than in a browser dialog.
  const [confirming, setConfirming] = useState(false);
  const [resuming, setResuming] = useState(false);
  const resumable = RESUMABLE_STATUSES.includes(pipeline.status) && pipeline.cases == null;
  const created = new Date(pipeline.created_at);
  const sharesHeld = gpuSharesSummary(pipeline.gpu_shares);

  return (
    <>
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
          {sharesHeld && <div className="muted">{sharesHeld}</div>}
          {pipeline.sweep && <div className="muted">{bestParameters(pipeline.sweep)}</div>}
        </td>
        <td>{pipeline.owner}</td>
        <td title={created.toLocaleString()}>{timeAgo(created)}</td>
        <td>
          <div className="links">
            {pipeline.kubeflow_run_url && (
              <a
                className="link-button"
                href={pipeline.kubeflow_run_url}
                target="_blank"
                rel="noreferrer"
              >
                <Workflow size={14} /> Kubeflow
              </a>
            )}
            {pipeline.mlflow_run_url && (
              <a
                className="link-button"
                href={pathOnThisOrigin(pipeline.mlflow_run_url)}
                target="_blank"
                rel="noreferrer"
              >
                <ChartLine size={14} /> MLflow
              </a>
            )}
          </div>
        </td>
        <td className="actions">
          {resumable && (
            <button
              type="button"
              className="button ghost small"
              aria-expanded={resuming}
              onClick={() => setResuming(!resuming)}
            >
              <Play size={14} /> Resume
            </button>
          )}
          {FINISHED_STATUSES.includes(pipeline.status) ? (
            <button
              type="button"
              className="icon-button"
              aria-label={`Delete Pipeline ${pipeline.id}`}
              title={`Delete Pipeline ${pipeline.id}`}
              onClick={() => onDelete(pipeline)}
            >
              <Trash2 size={16} />
            </button>
          ) : confirming ? (
            <>
              <button
                type="button"
                className="button danger small"
                onClick={() => onCancel(pipeline)}
              >
                Cancel Pipeline
              </button>
              <button
                type="button"
                className="button ghost small"
                onClick={() => setConfirming(false)}
              >
                Keep
              </button>
            </>
          ) : (
            <button
              type="button"
              className="button ghost small"
              onClick={() => setConfirming(true)}
            >
              Cancel
            </button>
          )}
        </td>
      </tr>
      {pipeline.failure && <FailureRow failure={pipeline.failure} columns={7} />}
      {resuming && (
        <tr>
          <td colSpan={7}>
            <ResumeForm
              pipeline={pipeline}
              onResume={(body) => {
                setResuming(false);
                onResume(pipeline, body);
              }}
              onClose={() => setResuming(false)}
            />
          </td>
        </tr>
      )}
    </>
  );
}

// The API takes these beside the Secrets; a share left out keeps the stored one.
type ResumeBody = {
  secrets: Record<string, string>;
  gpu_share?: number;
  serving_shares?: Record<string, number>;
};

/** New GPU Shares for the Stages that hold one, and the Secrets, which are never kept. */
function ResumeForm({
  pipeline,
  onResume,
  onClose,
}: {
  pipeline: Pipeline;
  onResume: (body: ResumeBody) => void;
  onClose: () => void;
}) {
  const [shares, setShares] = useState<Record<string, string>>({});
  const [secrets, setSecrets] = useState<Record<string, string>>({});
  const stored = pipeline.gpu_shares;
  const training = TRAINING_STAGES.find((stage) => stage in stored);
  const serving = ["distill", "evaluate"].filter((stage) => stage in stored);
  // A Teacher's API key only where `distill` may ask one.
  const slots = Object.entries(SECRET_SLOTS).filter(
    ([slot]) => slot !== "teacher_api_key" || pipeline.stages.includes("distill"),
  );
  const entered = (stage: string) => shares[stage]?.trim();

  function body(): ResumeBody {
    const newServing = serving.filter(entered);
    return {
      secrets: Object.fromEntries(Object.entries(secrets).filter(([, value]) => value)),
      ...(training && entered("training") && { gpu_share: Number(entered("training")) }),
      ...(newServing.length > 0 && {
        serving_shares: Object.fromEntries(
          newServing.map((stage) => [stage, Number(entered(stage))]),
        ),
      }),
    };
  }

  return (
    <form
      className="serve-form"
      onSubmit={(event) => {
        event.preventDefault();
        onResume(body());
      }}
    >
      <div className="field-grid">
        {training && (
          <GpuShareField
            id={`${pipeline.id}-gpu-share`}
            label="GPU Share"
            name="gpu_share"
            stored={stored[training]}
            value={shares.training ?? ""}
            onChange={(value) => setShares({ ...shares, training: value })}
          />
        )}
        {serving.map((stage) => (
          <GpuShareField
            key={stage}
            id={`${pipeline.id}-${stage}-gpu-share`}
            label={`GPU Share of ${stage}`}
            name="gpu_memory_utilization"
            stored={stored[stage]}
            value={shares[stage] ?? ""}
            onChange={(value) => setShares({ ...shares, [stage]: value })}
          />
        ))}
        {slots.map(([slot, label]) => (
          <SecretInput
            key={slot}
            slot={slot}
            label={label}
            value={secrets[slot] ?? ""}
            onChange={(value) => setSecrets({ ...secrets, [slot]: value })}
          />
        ))}
      </div>
      <div className="serve-actions">
        <button type="button" className="button ghost small" onClick={onClose}>
          Keep as it is
        </button>
        <button type="submit" className="button primary small">
          <Play size={14} /> Resume Pipeline {pipeline.id}
        </button>
      </div>
    </form>
  );
}

function bestParameters(sweep: SweepOutput): string {
  const values = Object.values(sweep.parameters).flatMap((block) =>
    Object.entries(block).map(([name, value]) => `${name}=${value}`),
  );
  return `Best: ${values.join(", ")} (objective ${sweep.objective})`;
}

function summary(pipelines?: Pipeline[]): string {
  if (!pipelines) return "Loading…";
  const counts = new Map<string, number>();
  for (const pipeline of pipelines)
    counts.set(pipeline.status, (counts.get(pipeline.status) ?? 0) + 1);
  const active = [...counts].filter(([status]) => !FINISHED_STATUSES.includes(status));
  const total = `${pipelines.length} Pipeline${pipelines.length === 1 ? "" : "s"}`;
  return active.length
    ? `${total} · ${active.map(([status, count]) => `${count} ${status}`).join(" · ")}`
    : total;
}

function timeAgo(date: Date): string {
  const minutes = Math.round((Date.now() - date.getTime()) / 60000);
  if (minutes < 1) return "just now";
  if (minutes < 60) return `${minutes} min ago`;
  if (minutes < 24 * 60) return `${Math.round(minutes / 60)} h ago`;
  return date.toLocaleDateString();
}
