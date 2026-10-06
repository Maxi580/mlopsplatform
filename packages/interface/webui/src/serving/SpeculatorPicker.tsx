import { NGRAM_LOOKUP } from "../config";
import InfoBox from "../fields/InfoBox";
import NumberInput from "../fields/NumberInput";
import type { Schema } from "../pipelines/pipelineForm";
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
type NumericOption = "num_speculative_tokens" | "prompt_lookup_min" | "prompt_lookup_max";

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

/** Off, n-gram, or a Speculator; only those trained for the model can be picked. Its numbers
 * are described, bounded and defaulted by the published `Speculative` schema. */
export default function SpeculatorPicker({
  models,
  model,
  schema,
  value,
  onChange,
}: {
  models: RegisteredModel[];
  model: string;
  schema: Schema;
  value: Speculative | null;
  onChange: (value: Speculative | null) => void;
}) {
  const found = speculators(models);
  const options: Schema = schema.properties ?? {};
  const tokens = value?.num_speculative_tokens ?? options.num_speculative_tokens?.default;
  const chosen = value ? (value.model ?? NGRAM) : OFF;
  const shown: NumericOption[] = !value
    ? []
    : value.method === NGRAM
      ? ["num_speculative_tokens", "prompt_lookup_min", "prompt_lookup_max"]
      : ["num_speculative_tokens"];

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
        <div className="field-heading">
          <label className="field-label" htmlFor="speculative">
            Speculative decoding <code>speculative</code>
          </label>
          <InfoBox
            id="speculative-info"
            text="A Speculator drafts only for the pinned model it was trained for; n-gram needs none."
          />
        </div>
        <select
          id="speculative"
          value={chosen}
          aria-describedby="speculative-info"
          onChange={(event) => choose(event.target.value)}
        >
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
      </div>
      {value &&
        shown.map((name) => (
          <SpeculativeOption
            key={name}
            name={name}
            schema={options[name] ?? {}}
            value={value[name]}
            onChange={(number) => onChange({ ...value, [name]: number })}
          />
        ))}
    </div>
  );
}

// One number of the speculative config; left empty, the API's default applies.
function SpeculativeOption({
  name,
  schema,
  value,
  onChange,
}: {
  name: NumericOption;
  schema: Schema;
  value?: number;
  onChange: (value: number | undefined) => void;
}) {
  const number = schema.anyOf?.find((option: Schema) => option.type !== "null") ?? schema;
  const title = schema.title ?? name;
  const { minimum, maximum, exclusiveMinimum, exclusiveMaximum } = number;
  return (
    <div className="field">
      <div className="field-heading">
        <label className="field-label" htmlFor={name}>
          {title} <code>{name}</code>
        </label>
        {schema.description && <InfoBox id={`${name}-info`} text={schema.description} />}
      </div>
      <NumberInput
        id={name}
        label={title}
        value={value === undefined ? "" : String(value)}
        from={schema.default != null ? String(schema.default) : undefined}
        bounds={{ minimum, maximum, exclusiveMinimum, exclusiveMaximum }}
        integer={number.type === "integer"}
        aria-describedby={schema.description ? `${name}-info` : undefined}
        // Only numbers are kept; a stray letter changes nothing.
        onChange={(text) => {
          if (text.trim() === "") onChange(undefined);
          else if (!Number.isNaN(Number(text))) onChange(Number(text));
        }}
      />
    </div>
  );
}
