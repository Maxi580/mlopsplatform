import {
  CircleAlert,
  CircleCheck,
  Cpu,
  HardDrive,
  KeyRound,
  ListChecks,
  LoaderCircle,
  Rocket,
} from "lucide-react";
import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { ApiError, callApi, errorMessage, useApi } from "../api";
import {
  BENCHMARKS,
  DATASETS,
  ENDPOINTS,
  MODELS,
  PIPELINES,
  SCHEMA,
  VALIDATE_PIPELINE,
} from "../apiPaths";
import { DRAFT_KEY, SECRET_SLOTS } from "../config";
import type { Catalog } from "../fields/catalog";
import SecretInput from "../fields/SecretInput";
import { formatBytes } from "../formatBytes";
import type { Endpoint } from "../serving/endpoint";
import type { Dataset, RegisteredModel } from "../storage/storage";
import { usePreview } from "../usePreview";
import FormSectionView, { SectionSwitch, SectionTitle } from "./FormSectionView";
import { type Benchmark, type GpuShares, gpuSharesSummary } from "./pipeline";
import {
  type FieldError,
  type FormSection,
  type FormValues,
  isSwitchedOn,
  pipelineForm,
  pipelineRequestFromForm,
  placeErrors,
} from "./pipelineForm";

type PlacedErrors = ReturnType<typeof placeErrors>;
type Download = { kind: string; ref: string; bytes: number | null; cached: boolean };
type RequestPreview = {
  downloads: Download[];
  download_bytes: number;
  cached_bytes: number;
  gpu_shares?: GpuShares;
};

const NO_ERRORS: PlacedErrors = { byName: {}, unplaced: [] };

export default function NewPipelinePage() {
  const schema = useApi<Record<string, unknown>>(SCHEMA).data;
  const datasets = useApi<Dataset[]>(DATASETS);
  const models = useApi<RegisteredModel[]>(MODELS).data;
  const endpoints = useApi<Endpoint[]>(ENDPOINTS).data;
  const benchmarks = useApi<Benchmark[]>(BENCHMARKS).data;
  const catalog = {
    datasets: datasets.data ?? [],
    models: models ?? [],
    endpoints: endpoints ?? [],
    reloadDatasets: datasets.reload,
  };

  return schema ? (
    <PipelineBuilder schema={schema} catalog={catalog} benchmarks={benchmarks ?? []} />
  ) : (
    <p className="muted loading">
      <LoaderCircle className="spin" size={16} /> Loading the Pipeline Request schema…
    </p>
  );
}

function PipelineBuilder({
  schema,
  catalog,
  benchmarks,
}: {
  schema: Record<string, unknown>;
  catalog: Catalog;
  benchmarks: Benchmark[];
}) {
  const navigate = useNavigate();
  const [values, setValues] = useState<FormValues>(loadDraft);
  const [secrets, setSecrets] = useState<Record<string, string>>({});
  const [errors, setErrors] = useState<PlacedErrors>(NO_ERRORS);
  const [resolved, setResolved] = useState<unknown>();
  const [busy, setBusy] = useState<string>();
  // Which fields show, and their defaults, depend on the values, e.g. on a Phase's algorithm.
  const form = pipelineForm(schema, values);
  const request = pipelineRequestFromForm(form, values);
  // A Teacher's API key only where the Teacher is a model at an API.
  const slots = Object.entries(SECRET_SLOTS).filter(
    ([slot]) => slot !== "teacher_api_key" || hasField(form, "distill.api_url"),
  );
  const filledSecrets = Object.fromEntries(
    slots.flatMap(([slot]) => (secrets[slot] ? [[slot, secrets[slot]]] : [])),
  );
  // An invalid request has no preview yet; the Validate button explains why.
  const preview = usePreview<RequestPreview>(VALIDATE_PIPELINE, {
    request,
    secrets: filledSecrets,
  });
  const sharesHeld = gpuSharesSummary(preview?.gpu_shares);

  useEffect(() => {
    try {
      localStorage.setItem(DRAFT_KEY, JSON.stringify(values));
    } catch {
      // Storage may be blocked; the draft is only a convenience.
    }
  }, [values]);

  /** Sends the request to `path`; the API checks it the same way for validate and submit. */
  async function send(path: string) {
    // 1. A fresh check: earlier errors and results no longer apply.
    setBusy(path);
    setErrors(NO_ERRORS);
    setResolved(undefined);

    // 2. The API's verdict: the new Pipeline, the resolved request, or every error in place.
    try {
      const answer = await callApi<{ id: number; request: unknown }>(path, {
        request,
        secrets: filledSecrets,
      });
      if (path === PIPELINES) return navigate("/", { state: { submitted: answer.id } });
      setResolved(answer.request);
    } catch (failure) {
      const detail = failure instanceof ApiError ? failure.detail : undefined;
      setErrors(
        Array.isArray(detail)
          ? placeErrors(form, detail as FieldError[])
          : { byName: {}, unplaced: [errorMessage(failure)] },
      );
    }
    setBusy(undefined);
  }

  // Top-level values (the name) get their own card; each Stage gets one too.
  const topLevel: FormSection = {
    ...form,
    children: form.children.filter((node) => node.kind !== "section"),
  };
  const stages = form.children.filter((node): node is FormSection => node.kind === "section");
  const errorCount = Object.values(errors.byName).flat().length + errors.unplaced.length;
  const sectionProps = {
    values,
    errors: errors.byName,
    // A model search also finds what the user's token opens.
    catalog: { ...catalog, hfToken: filledSecrets.hf_token },
    benchmarks,
    onChange: setValues,
  };

  return (
    <>
      <header className="page-header">
        <div>
          <h1>New Pipeline</h1>
          <p className="muted">
            Built from the published schema: the same Pipeline Request as `mlp run`.
          </p>
        </div>
      </header>

      <div className="builder">
        <form
          className="builder-form"
          onSubmit={(event) => {
            event.preventDefault();
            send(PIPELINES);
          }}
        >
          {errors.unplaced.map((message) => (
            <p key={message} className="banner danger" role="alert">
              <CircleAlert size={16} /> {message}
            </p>
          ))}
          <section className="card">
            <h2>Pipeline</h2>
            <FormSectionView section={topLevel} {...sectionProps} />
          </section>
          {stages.map((stage) => (
            <section key={stage.name} className="card">
              <h2>
                <SectionTitle section={stage} /> <span className="chip">Stage</span>
                <SectionSwitch section={stage} values={values} onChange={setValues} />
              </h2>
              {isSwitchedOn(stage, values) && <FormSectionView section={stage} {...sectionProps} />}
            </section>
          ))}
          <section className="card">
            <h2>
              <KeyRound size={18} /> Secrets
            </h2>
            <p className="muted">Sent beside the request and kept only while the Pipeline runs.</p>
            <div className="field-grid">
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
          </section>
        </form>

        <aside className="builder-aside">
          <div className="card sticky">
            <h2>Pipeline Request</h2>
            <pre className="preview">{JSON.stringify(request, null, 2)}</pre>
            {preview && (
              <p className="muted">
                <HardDrive size={16} /> {downloadSummary(preview)}
              </p>
            )}
            {sharesHeld && (
              <p className="muted">
                <Cpu size={16} /> <span>{sharesHeld}</span>
              </p>
            )}
            {errorCount > 0 && (
              <p className="field-error">
                {errorCount} problem{errorCount === 1 ? "" : "s"} to fix
              </p>
            )}
            {resolved !== undefined && (
              <div className="banner success column">
                <span>
                  <CircleCheck size={16} /> Valid. With every Reference pinned:
                </span>
                <pre className="preview small">{JSON.stringify(resolved, null, 2)}</pre>
              </div>
            )}
            <div className="aside-actions">
              <button
                type="button"
                className="button"
                disabled={!!busy}
                onClick={() => send(VALIDATE_PIPELINE)}
              >
                {busy === VALIDATE_PIPELINE ? (
                  <LoaderCircle className="spin" size={16} />
                ) : (
                  <ListChecks size={16} />
                )}
                Validate
              </button>
              <button
                type="button"
                className="button primary"
                disabled={!!busy}
                onClick={() => send(PIPELINES)}
              >
                {busy === PIPELINES ? (
                  <LoaderCircle className="spin" size={16} />
                ) : (
                  <Rocket size={16} />
                )}
                Submit
              </button>
            </div>
          </div>
        </aside>
      </div>
    </>
  );
}

function downloadSummary(preview: RequestPreview): string {
  const toDownload = formatBytes(preview.download_bytes);
  return `${toDownload} to download, ${formatBytes(preview.cached_bytes)} already cached`;
}

function hasField(section: FormSection, name: string): boolean {
  return section.children.some(
    (node) => node.name === name || (node.kind === "section" && hasField(node, name)),
  );
}

function loadDraft(): FormValues {
  try {
    return JSON.parse(localStorage.getItem(DRAFT_KEY) ?? "") as FormValues;
  } catch {
    return { fields: {}, more: {} };
  }
}
