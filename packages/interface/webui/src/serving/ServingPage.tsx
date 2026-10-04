import { CircleAlert, LoaderCircle, Play, Square } from "lucide-react";
import { useMemo, useState } from "react";
import { ApiError, callApi, errorMessage, useApi } from "../api";
import { ENDPOINTS, MODEL_CACHE, MODELS, SCHEMA, stopEndpoint } from "../apiPaths";
import { ENDPOINT_LIST_REFRESH_MS } from "../config";
import FormSectionView from "../pipelines/FormSectionView";
import {
  type FieldError,
  type FormSection,
  type FormValues,
  pipelineForm,
  pipelineRequestFromForm,
  placeErrors,
} from "../pipelines/pipelineForm";
import StatusBadge from "../pipelines/StatusBadge";
import type { ModelCache, RegisteredModel } from "../storage/storage";
import type { Endpoint } from "./endpoint";

type Notice = { text: string; failed: boolean };
type PublishedSchema = { $defs: Record<string, Record<string, unknown>> };

export default function ServingPage() {
  const endpoints = useApi<Endpoint[]>(ENDPOINTS, ENDPOINT_LIST_REFRESH_MS);
  const [notice, setNotice] = useState<Notice>();

  async function stop(name: string) {
    try {
      await callApi(stopEndpoint(name), {});
      setNotice({ text: `Stopped ${name}`, failed: false });
    } catch (failure) {
      setNotice({ text: errorMessage(failure), failed: true });
    }
    endpoints.reload();
  }

  return (
    <>
      <header className="page-header">
        <div>
          <h1>Serving</h1>
          <p className="muted">
            OpenAI-compatible Endpoints, behind the platform login; they run until stopped.
          </p>
        </div>
      </header>

      {notice && (
        <p className={`banner ${notice.failed ? "danger" : "success"}`}>
          {notice.failed && <CircleAlert size={16} />} {notice.text}
        </p>
      )}
      {endpoints.error && (
        <p className="banner danger">
          <CircleAlert size={16} /> {endpoints.error.message}
        </p>
      )}

      {endpoints.data?.length === 0 && (
        <p className="muted">No Endpoints yet; start one below or add `serve` to a Pipeline.</p>
      )}
      {!!endpoints.data?.length && (
        <div className="card table-card">
          <table>
            <thead>
              <tr>
                {["Endpoint", "Model", "Status", "URL", "Started (UTC)"].map((column) => (
                  <th key={column}>{column}</th>
                ))}
                <th aria-label="Actions" />
              </tr>
            </thead>
            <tbody>
              {endpoints.data.map((endpoint) => (
                <EndpointRow
                  // A stopped Endpoint keeps its row, and its name may be running again.
                  key={`${endpoint.name}-${endpoint.created_at}`}
                  endpoint={endpoint}
                  onStop={stop}
                />
              ))}
            </tbody>
          </table>
        </div>
      )}

      <section className="storage-section">
        <h2>Start an Endpoint</h2>
        <p className="muted">
          Any Model Version or Base Model; GPUs come from the platform settings.
        </p>
        <StartForm
          onStarted={(name) => {
            setNotice({
              text: `Started ${name}; it is pending until vLLM is ready`,
              failed: false,
            });
            endpoints.reload();
          }}
        />
      </section>
    </>
  );
}

function EndpointRow({ endpoint, onStop }: { endpoint: Endpoint; onStop: (name: string) => void }) {
  const stopped = endpoint.status === "stopped";
  return (
    <tr>
      <td className="pipeline-name">{endpoint.name}</td>
      <td className="mono">{endpoint.model}</td>
      <td>
        <StatusBadge status={endpoint.status} />
      </td>
      <td className="mono">{stopped ? "" : <a href={endpoint.url}>{endpoint.url}</a>}</td>
      <td>{endpoint.created_at.slice(0, 16).replace("T", " ")}</td>
      <td className="actions">
        {!stopped && (
          <button
            type="button"
            className="button ghost small"
            onClick={() => onStop(endpoint.name)}
          >
            <Square size={14} /> Stop
          </button>
        )}
      </td>
    </tr>
  );
}

/** The model to serve, its Endpoint name and the curated options of the `serve` Stage. */
function StartForm({ onStarted }: { onStarted: (name: string) => void }) {
  const schema = useApi<PublishedSchema>(SCHEMA).data;
  const models = useApi<RegisteredModel[]>(MODELS).data;
  const cache = useApi<ModelCache>(MODEL_CACHE).data;
  const form = useMemo(() => schema && servingOptionsForm(schema), [schema]);
  const [model, setModel] = useState("");
  const [values, setValues] = useState<FormValues>({ fields: {}, more: {} });
  const [errors, setErrors] = useState<ReturnType<typeof placeErrors>>();
  const [busy, setBusy] = useState(false);
  const suggestions = [
    ...(models ?? []).flatMap((m) => m.versions.map((v) => `model:${m.name}@${v.version}`)),
    ...(cache?.entries ?? [])
      .filter((entry) => entry.kind === "base_model")
      .map((entry) => entry.reference),
  ];

  if (!form) return <LoaderCircle className="spin" size={16} />;

  async function start(form: FormSection) {
    setBusy(true);
    setErrors(undefined);
    const body = { model: model.trim(), ...pipelineRequestFromForm(form, values) };
    try {
      const started = await callApi<Endpoint>(ENDPOINTS, body);
      onStarted(started.name);
    } catch (failure) {
      const detail = failure instanceof ApiError ? failure.detail : undefined;
      // FastAPI puts the body's fields under `body`.
      const located = Array.isArray(detail)
        ? (detail as FieldError[]).map((e) => ({ ...e, loc: e.loc.slice(1) }))
        : undefined;
      setErrors(
        located ? placeErrors(form, located) : { byName: {}, unplaced: [errorMessage(failure)] },
      );
    }
    setBusy(false);
  }

  return (
    <form
      className="card serve-form"
      onSubmit={(event) => {
        event.preventDefault();
        start(form);
      }}
    >
      {errors?.unplaced.map((message) => (
        <p key={message} className="banner danger" role="alert">
          <CircleAlert size={16} /> {message}
        </p>
      ))}
      <div className="field-grid">
        <div className="field">
          <label className="field-label" htmlFor="served-model">
            Model <code>model</code>
          </label>
          <input
            id="served-model"
            value={model}
            list="served-models"
            autoComplete="off"
            placeholder="model:name@version or hf:org/name"
            onChange={(event) => setModel(event.target.value)}
          />
          <datalist id="served-models">
            {suggestions.map((reference) => (
              <option key={reference} value={reference} />
            ))}
          </datalist>
        </div>
      </div>
      <FormSectionView
        section={form}
        values={values}
        errors={errors?.byName ?? {}}
        datasetReferences={[]}
        onChange={setValues}
      />
      <div className="serve-actions">
        <button type="submit" className="button primary" disabled={busy}>
          {busy ? <LoaderCircle className="spin" size={16} /> : <Play size={16} />}
          Start
        </button>
      </div>
    </form>
  );
}

// The `serve` Stage's options are an Endpoint's, so the published schema describes both.
function servingOptionsForm(schema: PublishedSchema): FormSection {
  return pipelineForm({ ...schema.$defs.Serve, $defs: schema.$defs });
}
