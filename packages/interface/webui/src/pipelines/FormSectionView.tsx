import { Plus, Trash2 } from "lucide-react";
import { REFERENCE_PLACEHOLDERS } from "../config";
import { type FormField, type FormSection, type FormValues, moreName } from "./pipelineForm";

type Props = {
  section: FormSection;
  values: FormValues;
  errors: Record<string, string[]>;
  datasetReferences: string[];
  onChange: (values: FormValues) => void;
};

/** A section's fields in a grid, its subsections below, and its more settings last. */
export default function FormSectionView({
  section,
  values,
  errors,
  datasetReferences,
  onChange,
}: Props) {
  const fields = section.children.filter((node): node is FormField => node.kind !== "section");
  const fixed = fields.filter((field) => field.kind === "fixed");
  const editable = fields.filter((field) => field.kind !== "fixed");
  const subsections = section.children.filter(
    (node): node is FormSection => node.kind === "section",
  );
  const setField = (name: string, value: string) =>
    onChange({ ...values, fields: { ...values.fields, [name]: value } });

  return (
    <>
      <FieldErrors messages={errors[section.name]} />
      {fixed.length > 0 && (
        <div className="fixed-values">
          {fixed.map((field) => (
            <span key={field.name} className="fixed-value" title="Set by the schema">
              {lastPart(field.name)} <strong>{String(field.fixed)}</strong>
            </span>
          ))}
        </div>
      )}
      {editable.length > 0 && (
        <div className="field-grid">
          {editable.map((field) => (
            <FieldInput
              key={field.name}
              field={field}
              value={values.fields[field.name] ?? ""}
              errors={errors[field.name]}
              datasetReferences={datasetReferences}
              onChange={(value) => setField(field.name, value)}
            />
          ))}
        </div>
      )}
      {section.moreSettings && (
        <MoreSettings section={section} values={values} errors={errors} onChange={onChange} />
      )}
      {subsections.flatMap(itemsOfList).map((subsection) => (
        <fieldset key={subsection.name} className="subsection">
          <legend>{subsection.title}</legend>
          <FormSectionView
            section={subsection}
            values={values}
            errors={errors}
            datasetReferences={datasetReferences}
            onChange={onChange}
          />
        </fieldset>
      ))}
    </>
  );
}

function FieldInput({
  field,
  value,
  errors,
  datasetReferences,
  onChange,
}: {
  field: FormField;
  value: string;
  errors?: string[];
  datasetReferences: string[];
  onChange: (value: string) => void;
}) {
  const prefix = Object.keys(REFERENCE_PLACEHOLDERS).find((start) =>
    field.pattern?.startsWith(start),
  );
  const isDataset = prefix === "^dataset:";
  const common = {
    id: field.name,
    name: field.name,
    value,
    "aria-invalid": !!errors,
    "aria-describedby": errors ? `${field.name}-error` : undefined,
    onChange: (event: { target: { value: string } }) => onChange(event.target.value),
  };

  return (
    <div className="field">
      <label className="field-label" htmlFor={field.name}>
        {field.title} <code>{lastPart(field.name)}</code>
      </label>
      {field.kind === "choice" ? (
        <select {...common}>
          <option value="">Choose…</option>
          {field.choices?.map((choice) => (
            <option key={String(choice)}>{String(choice)}</option>
          ))}
        </select>
      ) : (
        <input
          {...common}
          inputMode={field.kind === "integer" || field.kind === "number" ? "decimal" : undefined}
          placeholder={
            field.kind === "list"
              ? "one value, or several separated by commas"
              : prefix && REFERENCE_PLACEHOLDERS[prefix]
          }
          list={isDataset ? "dataset-references" : undefined}
          autoComplete="off"
        />
      )}
      {isDataset && (
        <datalist id="dataset-references">
          {datasetReferences.map((reference) => (
            <option key={reference} value={reference} />
          ))}
        </datalist>
      )}
      <FieldErrors id={`${field.name}-error`} messages={errors} />
    </div>
  );
}

// Settings the schema allows beyond its fields; values read as JSON when they can (0.1, true, [..]).
function MoreSettings({ section, values, errors, onChange }: Omit<Props, "datasetReferences">) {
  const rows = values.more[section.name] ?? [];
  const setRows = (next: typeof rows) =>
    onChange({ ...values, more: { ...values.more, [section.name]: next } });

  return (
    <div className="more-settings">
      <p className="field-label">More settings</p>
      {rows.map((row, index) => (
        // biome-ignore lint/suspicious/noArrayIndexKey: rows are controlled inputs without an identity.
        <div key={index} className="more-row">
          <input
            aria-label="Setting"
            placeholder="weight_decay"
            value={row.key}
            onChange={(event) =>
              setRows(rows.map((r, i) => (i === index ? { ...r, key: event.target.value } : r)))
            }
          />
          <input
            aria-label="Value"
            placeholder="0.01"
            value={row.value}
            onChange={(event) =>
              setRows(rows.map((r, i) => (i === index ? { ...r, value: event.target.value } : r)))
            }
          />
          <button
            type="button"
            className="icon-button"
            aria-label="Remove setting"
            onClick={() => setRows(rows.filter((_, i) => i !== index))}
          >
            <Trash2 size={16} />
          </button>
        </div>
      ))}
      <button
        type="button"
        className="button ghost small"
        onClick={() => setRows([...rows, { key: "", value: "" }])}
      >
        <Plus size={14} /> Add setting
      </button>
      <FieldErrors
        id={`${moreName(section.name)}-error`}
        messages={errors[moreName(section.name)]}
      />
    </div>
  );
}

function FieldErrors({ id, messages }: { id?: string; messages?: string[] }) {
  if (!messages?.length) return null;
  return (
    <p className="field-error" id={id} role="alert">
      {messages.join("; ")}
    </p>
  );
}

// A list's items (e.g. Phase 1, Phase 2) show directly, without a box for the list.
function itemsOfList(section: FormSection): FormSection[] {
  const items = section.children.filter((node): node is FormSection => node.kind === "section");
  const isList = !section.moreSettings && items.length === section.children.length;
  return isList ? items : [section];
}

function lastPart(name: string): string {
  return name.split(".").pop()!;
}
