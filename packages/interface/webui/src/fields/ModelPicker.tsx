import { useEffect, useState } from "react";
import { callApi } from "../api";
import { BASE_MODELS } from "../apiPaths";
import { MODEL_SEARCH_DELAY_MS } from "../config";
import type { RegisteredModel } from "../storage/storage";
import Combobox, { type Choice, type ChoiceGroup } from "./Combobox";
import type { Catalog } from "./catalog";

// What `GET /base-models` answers: the curated models, or the Hub's for a search.
type BaseModel = { name: string; reference: string; parameters: number | null; gated: boolean };

type Props = {
  id: string;
  label: string;
  value: string;
  // The References the field takes, e.g. `hf:`, `model:`, `endpoint:` and `@finetune`.
  references: string[];
  catalog: Catalog;
  invalid?: boolean;
  describedBy?: string;
  onChange: (value: string) => void;
};

/** A model field's picker: this Pipeline's outputs, our Model Versions, open-source models from
 * the Hub and running Endpoints, each group only where the field takes it. */
export default function ModelPicker({ references, catalog, ...props }: Props) {
  const [typed, setTyped] = useState("");
  const hub = useBaseModels(references.includes("hf:"), typed, catalog.hfToken);
  const groups: ChoiceGroup[] = [];
  const outputs = references.filter((reference) => reference.startsWith("@"));
  if (outputs.length)
    groups.push({ label: "This Pipeline", choices: outputs.map(pipelineOutputChoice) });
  if (references.includes("model:"))
    groups.push({ label: "Our models", choices: modelVersionChoices(catalog.models) });
  if (references.includes("hf:"))
    groups.push({
      label: "Open source",
      choices: hub.models.map(baseModelChoice),
      searched: hub.searched,
    });
  if (references.includes("endpoint:"))
    groups.push({
      label: "Endpoints",
      choices: catalog.endpoints
        .filter((endpoint) => endpoint.status === "running")
        .map((endpoint) => ({
          value: `endpoint:${endpoint.name}`,
          label: endpoint.name,
          detail: endpoint.model,
        })),
    });

  return <Combobox {...props} groups={groups} placeholder="Search models…" onType={setTyped} />;
}

/** The curated models while nothing is typed, else the Hub's for what is, shortly after. */
function useBaseModels(wanted: boolean, typed: string, hfToken?: string) {
  const [found, setFound] = useState<{ models: BaseModel[]; searched: boolean }>({
    models: [],
    searched: false,
  });
  // A picked Reference, e.g. `hf:Qwen/Qwen3-8B`, is no search.
  const search = typed.includes(":") ? "" : typed.trim();
  useEffect(() => {
    if (!wanted) return;
    let current = true;
    const headers: Record<string, string> = hfToken ? { "x-hf-token": hfToken } : {};
    const path = `${BASE_MODELS}?search=${encodeURIComponent(search)}`;
    const timer = setTimeout(
      () =>
        callApi<BaseModel[]>(path, undefined, "GET", headers).then(
          (models) => current && setFound({ models, searched: !!search }),
          // The other groups still work without Hugging Face.
          () => current && setFound({ models: [], searched: !!search }),
        ),
      search ? MODEL_SEARCH_DELAY_MS : 0,
    );
    return () => {
      current = false;
      clearTimeout(timer);
    };
  }, [wanted, search, hfToken]);
  return found;
}

function pipelineOutputChoice(output: string): Choice {
  const stage = output.slice(1);
  return { value: output, label: `Output of ${stage[0].toUpperCase()}${stage.slice(1)}` };
}

// Every Model Version but the Speculators, which only draft for one model.
function modelVersionChoices(models: RegisteredModel[]): Choice[] {
  return models.flatMap((model) =>
    model.versions
      .filter((version) => !version.tags?.speculator)
      .map((version) => {
        const { weights, base_model: base } = version.tags ?? {};
        const kind = weights === "adapter" ? `Adapter on ${base}` : "full weights";
        return {
          value: `model:${model.name}@${version.version}`,
          label: `${model.name}@${version.version}`,
          detail: kind,
        };
      }),
  );
}

function baseModelChoice(model: BaseModel): Choice {
  return {
    value: model.reference,
    label: model.name,
    detail: model.parameters ? formatParameters(model.parameters) : undefined,
    badge: model.gated ? (
      <span className="gated" role="img" aria-label="gated" title="Accept its licence on the Hub">
        🔒
      </span>
    ) : undefined,
  };
}

function formatParameters(parameters: number): string {
  return parameters >= 1e9
    ? `${(parameters / 1e9).toFixed(1)}B`
    : `${Math.round(parameters / 1e6)}M`;
}
