import { LoaderCircle, Save } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { callApi, errorMessage, useApi } from "../api";
import { JUDGE, MODELS, SCHEMA } from "../apiPaths";
import { NO_CATALOG } from "../fields/catalog";
import ModelPicker from "../fields/ModelPicker";
import FormSectionView from "../pipelines/FormSectionView";
import { type FormValues, pipelineRequestFromForm } from "../pipelines/pipelineForm";
import type { RegisteredModel } from "../storage/storage";
import { type PublishedSchema, servingOptionsForm } from "./endpoint";

// What `GET /judge` answers: the settings, and the Judge's status while there is one.
type Judge = {
  enabled: boolean;
  spec: Record<string, unknown> | null;
  rubric: string;
  status: string | null;
};
type Draft = { enabled: boolean; model: string; rubric: string; values: FormValues };

/** Judging on or off, the model the Judge serves with the start form's options, and its rubric. */
export default function JudgeSettings({
  onSaved,
}: {
  onSaved: (text: string, failed: boolean) => void;
}) {
  const judge = useApi<Judge>(JUDGE).data;
  const schema = useApi<PublishedSchema>(SCHEMA).data;
  const models = useApi<RegisteredModel[]>(MODELS).data;
  const form = useMemo(() => schema && servingOptionsForm(schema), [schema]);
  const [draft, setDraft] = useState<Draft>();
  const [busy, setBusy] = useState(false);
  const [advanced, setAdvanced] = useState(false);
  useEffect(() => {
    if (judge && !draft) setDraft(draftOf(judge));
  }, [judge, draft]);

  if (!form || !draft) return <LoaderCircle className="spin" size={16} />;

  async function save() {
    if (!form || !draft) return;
    setBusy(true);
    const model = draft.model.trim();
    const spec = model ? { model, ...pipelineRequestFromForm(form, draft.values) } : null;
    try {
      await callApi(JUDGE, { enabled: draft.enabled, spec, rubric: draft.rubric }, "PUT");
      onSaved(draft.enabled ? "Judging is on; the Judge starts" : "Judging is off", false);
    } catch (failure) {
      onSaved(errorMessage(failure), true);
    }
    setBusy(false);
  }

  return (
    <form
      className="card serve-form"
      onSubmit={(event) => {
        event.preventDefault();
        save();
      }}
    >
      <label className="judge-switch">
        <input
          type="checkbox"
          checked={draft.enabled}
          onChange={(event) => setDraft({ ...draft, enabled: event.target.checked })}
        />
        Judging enabled
      </label>
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
      <div className="field">
        <label className="field-label" htmlFor="judge-rubric">
          Rubric <code>rubric</code>
        </label>
        <textarea
          id="judge-rubric"
          rows={4}
          value={draft.rubric}
          onChange={(event) => setDraft({ ...draft, rubric: event.target.value })}
        />
      </div>
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
    enabled: judge.enabled,
    model: String(model),
    rubric: judge.rubric,
    values: { fields, more: {} },
  };
}
