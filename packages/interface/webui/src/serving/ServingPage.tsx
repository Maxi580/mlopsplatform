import { CircleAlert, LoaderCircle, Play, Square, Trash2 } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { ApiError, callApi, errorMessage, useApi } from "../api";
import { ENDPOINTS, endpointByName, MODELS, SCHEMA, stopEndpoint } from "../apiPaths";
import { ENDPOINT_LIST_REFRESH_MS } from "../config";
import { NO_CATALOG } from "../fields/catalog";
import ModelPicker from "../fields/ModelPicker";
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
import type { RegisteredModel } from "../storage/storage";
import CopyValue from "./CopyValue";
import { type Endpoint, formatSeconds, ratePerSecond } from "./endpoint";
import SpeculatorPicker, { type Speculative } from "./SpeculatorPicker";

type Notice = { text: string; failed: boolean };
type PublishedSchema = { $defs: Record<string, Record<string, unknown>> };

export default function ServingPage() {
  const endpoints = useApi<Endpoint[]>(ENDPOINTS, ENDPOINT_LIST_REFRESH_MS);
  const tokensPerSecond = useTokensPerSecond(endpoints.data);
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

  async function remove(name: string) {
    try {
      await callApi(endpointByName(name), undefined, "DELETE");
      setNotice({ text: `Deleted ${name}`, failed: false });
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

      {endpoints.data?.length === 0 && <p className="muted">No Endpoints yet; start one below.</p>}
      {!!endpoints.data?.length && (
        <div className="card table-card">
          <table>
            <thead>
              <tr>
                {[
                  "Endpoint",
                  "Model",
                  "Status",
                  "Load",
                  "Tokens/s",
                  "TTFT",
                  "URL",
                  "Started (UTC)",
                ].map((column) => (
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
                  tokensPerSecond={tokensPerSecond[endpoint.name]}
                  onStop={stop}
                  onDelete={remove}
                />
              ))}
            </tbody>
          </table>
        </div>
      )}

      <section className="storage-section">
        <h2>Start an Endpoint</h2>
        <p className="muted">
          Any Model Version or Base Model, optionally drafting with n-gram or a Speculator; GPUs
          come from the platform settings.
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

/** Generated tokens per second of each Endpoint, between the list's last two readings. */
function useTokensPerSecond(endpoints?: Endpoint[]): Record<string, number | null> {
  type Readings = Record<string, { value: number; at: string }>;
  const [readings, setReadings] = useState<{ earlier: Readings; later: Readings }>({
    earlier: {},
    later: {},
  });
  useEffect(() => {
    if (!endpoints) return;
    const later = Object.fromEntries(
      endpoints.flatMap(({ name, stats }) =>
        stats ? [[name, { value: stats.generation_tokens, at: stats.read_at }]] : [],
      ),
    );
    setReadings((previous) => ({ earlier: previous.later, later }));
  }, [endpoints]);
  return Object.fromEntries(
    Object.entries(readings.later).map(([name, reading]) => [
      name,
      ratePerSecond(readings.earlier[name], reading),
    ]),
  );
}

function EndpointRow({
  endpoint,
  tokensPerSecond,
  onStop,
  onDelete,
}: {
  endpoint: Endpoint;
  tokensPerSecond?: number | null;
  onStop: (name: string) => void;
  onDelete: (name: string) => void;
}) {
  const stopped = endpoint.status === "stopped";
  const { stats } = endpoint;
  return (
    <tr>
      <td className="pipeline-name">
        {stopped ? endpoint.name : <Link to={`/serving/${endpoint.name}`}>{endpoint.name}</Link>}
      </td>
      <td className="mono">{endpoint.model}</td>
      <td>
        <StatusBadge status={endpoint.status} />
      </td>
      <td>{stats ? `${stats.running} running · ${stats.waiting} waiting` : "—"}</td>
      <td>{stats && tokensPerSecond != null ? tokensPerSecond.toFixed(1) : "—"}</td>
      <td>{stats ? formatSeconds(stats.time_to_first_token_p50) : "—"}</td>
      <td>{!stopped && <CopyValue label="URL" value={endpoint.url} />}</td>
      <td>{endpoint.created_at.slice(0, 16).replace("T", " ")}</td>
      <td className="actions">
        {stopped ? (
          <button
            type="button"
            className="icon-button"
            aria-label={`Delete ${endpoint.name}`}
            title={`Delete ${endpoint.name}`}
            onClick={() => onDelete(endpoint.name)}
          >
            <Trash2 size={16} />
          </button>
        ) : (
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

/** The model to serve, its Endpoint name, the serving options, and how it drafts. */
function StartForm({ onStarted }: { onStarted: (name: string) => void }) {
  const schema = useApi<PublishedSchema>(SCHEMA).data;
  const models = useApi<RegisteredModel[]>(MODELS).data;
  const form = useMemo(() => schema && servingOptionsForm(schema), [schema]);
  const [model, setModel] = useState("");
  const [name, setName] = useState("");
  const [speculative, setSpeculative] = useState<Speculative | null>(null);
  const [values, setValues] = useState<FormValues>({ fields: {}, more: {} });
  const [errors, setErrors] = useState<ReturnType<typeof placeErrors>>();
  const [busy, setBusy] = useState(false);

  if (!schema || !form) return <LoaderCircle className="spin" size={16} />;

  async function start(form: FormSection) {
    setBusy(true);
    setErrors(undefined);
    const body = {
      model: model.trim(),
      name: name.trim(),
      ...pipelineRequestFromForm(form, values),
      ...(speculative && { speculative }),
    };
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
          <ModelPicker
            id="served-model"
            label="Model"
            value={model}
            // Our Model Versions first, then open-source models.
            references={["model:", "hf:"]}
            catalog={{ ...NO_CATALOG, models: models ?? [] }}
            onChange={setModel}
          />
        </div>
        <div className="field">
          <label className="field-label" htmlFor="endpoint-name">
            Endpoint name <code>name</code>
          </label>
          <input
            id="endpoint-name"
            value={name}
            autoComplete="off"
            onChange={(event) => setName(event.target.value)}
          />
        </div>
      </div>
      <FormSectionView
        section={form}
        values={values}
        errors={errors?.byName ?? {}}
        catalog={NO_CATALOG}
        onChange={setValues}
      />
      <SpeculatorPicker
        models={models ?? []}
        model={model}
        schema={schema.$defs.Speculative ?? {}}
        value={speculative}
        onChange={setSpeculative}
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

// The serving options of `evaluate` and Teachers are an Endpoint's, so the schema publishes them.
function servingOptionsForm(schema: PublishedSchema): FormSection {
  return pipelineForm({ ...schema.$defs.ServingOptions, $defs: schema.$defs });
}
