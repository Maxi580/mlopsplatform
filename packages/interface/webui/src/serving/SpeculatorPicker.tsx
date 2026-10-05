import { NGRAM_LOOKUP, SPECULATIVE_TOKENS } from "../config";
import type { RegisteredModel } from "../storage/storage";

// The `speculative` block of an Endpoint start.
export type Speculative = {
  method: string;
  model?: string;
  num_speculative_tokens: number;
  prompt_lookup_min?: number;
  prompt_lookup_max?: number;
};

type Speculator = { reference: string; speculator: string; verifier: string };

const OFF = "";
const NGRAM = "ngram";

/** Every Model Version that is a Speculator, with its type and the model it was trained for. */
export function speculators(models: RegisteredModel[]): Speculator[] {
  return models.flatMap((model) =>
    model.versions
      .filter((version) => version.tags?.speculator)
      .map((version) => ({
        reference: `model:${model.name}@${version.version}`,
        speculator: version.tags.speculator,
        verifier: version.tags.verifier,
      })),
  );
}

// The API pins an unpinned model to its latest version, so only the exact pinned one matches.
export function draftsFor(speculator: Speculator, model: string): boolean {
  return speculator.verifier === model.trim();
}

/** Off, n-gram, or a Speculator; only those trained for the model can be picked. */
export default function SpeculatorPicker({
  models,
  model,
  value,
  onChange,
}: {
  models: RegisteredModel[];
  model: string;
  value: Speculative | null;
  onChange: (value: Speculative | null) => void;
}) {
  const found = speculators(models);
  const tokens = value?.num_speculative_tokens ?? SPECULATIVE_TOKENS;
  const chosen = value ? (value.model ?? NGRAM) : OFF;

  function choose(choice: string) {
    const picked = found.find((speculator) => speculator.reference === choice);
    if (choice === OFF) onChange(null);
    else if (choice === NGRAM)
      onChange({ method: NGRAM, num_speculative_tokens: tokens, ...NGRAM_LOOKUP });
    else if (picked)
      onChange({ method: picked.speculator, model: choice, num_speculative_tokens: tokens });
  }

  return (
    <div className="field-grid">
      <div className="field">
        <label className="field-label" htmlFor="speculative">
          Speculative decoding <code>speculative</code>
        </label>
        <select id="speculative" value={chosen} onChange={(event) => choose(event.target.value)}>
          <option value={OFF}>Off</option>
          <option value={NGRAM}>n-gram, from the context</option>
          {found.map((speculator) => (
            <option
              key={speculator.reference}
              value={speculator.reference}
              disabled={!draftsFor(speculator, model)}
            >
              {speculator.reference} ({speculator.speculator}, for {speculator.verifier})
            </option>
          ))}
        </select>
        <p className="field-info">
          A Speculator drafts only for the pinned model it was trained for; n-gram needs none.
        </p>
      </div>
      {value && (
        <div className="field">
          <label className="field-label" htmlFor="num_speculative_tokens">
            Draft tokens <code>num_speculative_tokens</code>
          </label>
          <input
            id="num_speculative_tokens"
            type="number"
            min={1}
            value={tokens}
            onChange={(event) =>
              onChange({ ...value, num_speculative_tokens: Number(event.target.value) })
            }
          />
        </div>
      )}
    </div>
  );
}
