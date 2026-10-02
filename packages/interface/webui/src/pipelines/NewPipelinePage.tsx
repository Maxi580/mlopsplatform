import {
  CircleAlert,
  CircleCheck,
  Eye,
  EyeOff,
  HardDrive,
  KeyRound,
  ListChecks,
  LoaderCircle,
  Rocket,
} from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { ApiError, callApi, errorMessage, useApi } from "../api";
import { DATASETS, PIPELINES, SCHEMA, VALIDATE_PIPELINE } from "../apiPaths";
import { DOWNLOAD_PREVIEW_DELAY_MS, DRAFT_KEY, SECRET_SLOTS, SWITCHED_ON } from "../config";
import { formatBytes } from "../formatBytes";
import FormSectionView from "./FormSectionView";
import {
  type FieldError,
  type FormSection,
  type FormValues,
  isSwitchedOn,
  pipelineForm,
  pipelineRequestFromForm,
  placeErrors,
} from "./pipelineForm";

type Dataset = { name: string; versions: { version: number }[] };
type PlacedErrors = ReturnType<typeof placeErrors>;
type Download = { kind: string; ref: string; bytes: number | null; cached: boolean };
type DownloadPreview = { downloads: Download[]; download_bytes: number; cached_bytes: number };

const NO_ERRORS: PlacedErrors = { byName: {}, unplaced: [] };

export default function NewPipelinePage() {
  const schema = useApi<Record<string, unknown>>(SCHEMA).data;
  const datasets = useApi<Dataset[]>(DATASETS).data;
  const form = useMemo(() => schema && pipelineForm(schema), [schema]);

  return form ? (
    <PipelineBuilder form={form} datasetReferences={datasetReferences(datasets ?? [])} />
  ) : (
    <p className="muted loading">
      <LoaderCircle className="spin" size={16} /> Loading the Pipeline Request schema…
    </p>
  );
}

function PipelineBuilder({
  form,
  datasetReferences,
}: {
  form: FormSection;
  datasetReferences: string[];
}) {
  const navigate = useNavigate();
  const [values, setValues] = useState<FormValues>(loadDraft);
  const [secrets, setSecrets] = useState<Record<string, string>>({});
  const [errors, setErrors] = useState<PlacedErrors>(NO_ERRORS);
  const [resolved, setResolved] = useState<unknown>();
  const [busy, setBusy] = useState<string>();
  const request = pipelineRequestFromForm(form, values);
  const filledSecrets = Object.fromEntries(Object.entries(secrets).filter(([, value]) => value));
  const downloads = useDownloadPreview(request, filledSecrets);

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
  const sectionProps = { values, errors: errors.byName, datasetReferences, onChange: setValues };

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
                {stage.title} <span className="chip">Stage</span>
                {stage.optional && (
                  <label className="stage-switch">
                    <input
                      type="checkbox"
                      checked={isSwitchedOn(stage, values)}
                      onChange={(event) =>
                        setValues({
                          ...values,
                          fields: {
                            ...values.fields,
                            [stage.name]: event.target.checked ? SWITCHED_ON : "",
                          },
                        })
                      }
                    />
                    Run {stage.name}
                  </label>
                )}
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
              {Object.entries(SECRET_SLOTS).map(([slot, label]) => (
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
            {downloads && (
              <p className="muted">
                <HardDrive size={16} /> {downloadSummary(downloads)}
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

/** What the API says the request downloads, asked again shortly after each change. */
function useDownloadPreview(request: unknown, secrets: Record<string, string>) {
  const [preview, setPreview] = useState<DownloadPreview>();
  const submission = JSON.stringify({ request, secrets });
  useEffect(() => {
    let current = true;
    const timer = setTimeout(() => {
      callApi<DownloadPreview>(VALIDATE_PIPELINE, JSON.parse(submission)).then(
        (answer) => current && setPreview(answer),
        // An invalid request has no downloads yet; the Validate button explains why.
        () => current && setPreview(undefined),
      );
    }, DOWNLOAD_PREVIEW_DELAY_MS);
    return () => {
      current = false;
      clearTimeout(timer);
    };
  }, [submission]);
  return preview;
}

function downloadSummary(preview: DownloadPreview): string {
  const toDownload = formatBytes(preview.download_bytes);
  return `${toDownload} to download, ${formatBytes(preview.cached_bytes)} already cached`;
}

function SecretInput(props: {
  slot: string;
  label: string;
  value: string;
  onChange: (value: string) => void;
}) {
  const [visible, setVisible] = useState(false);
  return (
    <div className="field">
      <label className="field-label" htmlFor={props.slot}>
        {props.label} <code>{props.slot}</code>
      </label>
      <div className="input-with-button">
        <input
          id={props.slot}
          type={visible ? "text" : "password"}
          value={props.value}
          autoComplete="off"
          placeholder="optional for public models"
          onChange={(event) => props.onChange(event.target.value)}
        />
        <button
          type="button"
          className="icon-button"
          aria-label={visible ? "Hide" : "Show"}
          onClick={() => setVisible(!visible)}
        >
          {visible ? <EyeOff size={16} /> : <Eye size={16} />}
        </button>
      </div>
    </div>
  );
}

function datasetReferences(datasets: Dataset[]): string[] {
  return datasets.flatMap((dataset) => [
    `dataset:${dataset.name}`,
    ...dataset.versions.map((version) => `dataset:${dataset.name}@${version.version}`),
  ]);
}

function loadDraft(): FormValues {
  try {
    return JSON.parse(localStorage.getItem(DRAFT_KEY) ?? "") as FormValues;
  } catch {
    return { fields: {}, more: {} };
  }
}
