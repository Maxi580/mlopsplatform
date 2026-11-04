import { LoaderCircle, Save } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { callApi, errorMessage, useApi } from "../api";
import { JUDGE, MODELS, SCHEMA } from "../apiPaths";
import { ENDPOINT_LIST_REFRESH_MS } from "../config";
import { NO_CATALOG } from "../fields/catalog";
import ModelPicker from "../fields/ModelPicker";
import FormSectionView from "../pipelines/FormSectionView";
import { type FormValues, pipelineRequestFromForm } from "../pipelines/pipelineForm";
import type { RegisteredModel } from "../storage/storage";
import EndpointPreview from "./EndpointPreview";
import { type PublishedSchema, servingOptionsForm } from "./endpoint";

// What `GET /judge` answers: the settings, and the Judge's status while there is one.
type Judge = {
  enabled: boolean;
  spec: Record<string, unknown> | null;
  system_prompt: string;
  status: string | null;
};
type Draft = { model: string; systemPrompt: string; values: FormValues };

/** Judging on or off, the model the Judge serves with the start form's options, its system prompt. */
export default function JudgeSettings({
  onSaved,
}: {
  onSaved: (text: string, failed: boolean) => void;
}) {
  // Reloaded, as the button shows what's stored, which a Smoke Test or a failed save can change.
  const stored = useApi<Judge>(JUDGE, ENDPOINT_LIST_REFRESH_MS);
  const judge = stored.data;
  const schema = useApi<PublishedSchema>(SCHEMA).data;
  const models = useApi<RegisteredModel[]>(MODELS).data;
  const form = useMemo(() => schema && servingOptionsForm(schema), [schema]);
  const [draft, setDraft] = useState<Draft>();
  const [busy, setBusy] = useState(false);
  const [advanced, setAdvanced] = useState(false);
  useEffect(() => {
    if (judge && !draft) setDraft(draftOf(judge));
  }, [judge, draft]);

  if (!form || !draft || !judge) return <LoaderCircle className="spin" size={16} />;

  async function save(enabled: boolean) {
    if (!form || !draft) return;
    setBusy(true);
    const model = draft.model.trim();
    const spec = model ? { model, ...pipelineRequestFromForm(form, draft.values) } : null;
    try {
      await callApi(JUDGE, { enabled, spec, system_prompt: draft.systemPrompt }, "PUT");
      onSaved(enabled ? "Judging is on; the Judge starts" : "Judging is off", false);
    } catch (failure) {
      onSaved(errorMessage(failure), true);
    }
    await stored.reload();
    setBusy(false);
  }

  return (
    <form
      className="card serve-form"
      onSubmit={(event) => {
        event.preventDefault();
        save(judge.enabled);
      }}
    >
      <button
        type="button"
        className={`button judge-switch ${judge.enabled ? "success" : "danger"}`}
        disabled={busy}
        onClick={() => save(!judge.enabled)}
      >
        {judge.enabled ? "Judge is activated" : "Judge is deactivated"}
      </button>
      <div className="field-grid">
        <div className="field">
          <label className="field-label" htmlFor="judge-model">
            Judge model
          </label>
          <ModelPicker
            id="judge-model"
            label="Judge model"
            value={draft.model}
            references={["model:", "hf:"]}
            catalog={{ ...NO_CATALOG, models: models ?? [] }}
            onChange={(model) => setDraft({ ...draft, model })}
          />
        </div>
      </div>
      <div className="field judge-system-prompt">
        <label className="field-label" htmlFor="judge-system-prompt">
          System prompt <code>system_prompt</code>
        </label>
        <textarea
          id="judge-system-prompt"
          rows={4}
          value={draft.systemPrompt}
          onChange={(event) => setDraft({ ...draft, systemPrompt: event.target.value })}
        />
      </div>
      {/* The start form's other options, built only while open, as the start form shows them too. */}
      <details
        className="judge-advanced"
        onToggle={(event) => setAdvanced(event.currentTarget.open)}
      >
        <summary>+ advanced</summary>
        {advanced && (
          <FormSectionView
            section={form}
            values={draft.values}
            errors={{}}
            catalog={NO_CATALOG}
            onChange={(values) => setDraft({ ...draft, values })}
          />
        )}
      </details>
      <EndpointPreview
        spec={
          draft.model.trim()
            ? { model: draft.model.trim(), ...pipelineRequestFromForm(form, draft.values) }
            : undefined
        }
      />
      <div className="serve-actions">
        <button type="submit" className="button primary" disabled={busy}>
          {busy ? <LoaderCircle className="spin" size={16} /> : <Save size={16} />}
          Save
        </button>
      </div>
    </form>
  );
}

// The stored spec's serving options as the form's edits.
function draftOf(judge: Judge): Draft {
  const { model = "", speculative: _, ...options } = judge.spec ?? {};
  const fields = Object.fromEntries(
    Object.entries(options).map(([key, value]) => [key, String(value)]),
  );
  return {
    model: String(model),
    systemPrompt: judge.system_prompt,
    values: { fields, more: {} },
  };
}
