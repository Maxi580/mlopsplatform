import { Plus, Trash2, X } from "lucide-react";
import { Fragment, type ReactNode, useState } from "react";
import { REFERENCE_PLACEHOLDERS, REWARD_TEMPLATE, SWITCHED_ON } from "../config";
import type { Catalog } from "../fields/catalog";
import DatasetPicker from "../fields/DatasetPicker";
import InfoBox from "../fields/InfoBox";
import ModelPicker from "../fields/ModelPicker";
import NumberInput from "../fields/NumberInput";
import { formatBytes } from "../formatBytes";
import type { Benchmark } from "./pipeline";
import {
  type Entry,
  type FormField,
  type FormSection,
  type FormValues,
  fieldValue,
  isSwitchedOn,
  moreName,
  withListItem,
  withoutListItem,
} from "./pipelineForm";

type Props = {
  section: FormSection;
  values: FormValues;
  errors: Record<string, string[]>;
  // What the model and Dataset pickers offer.
  catalog: Catalog;
  // The catalog the benchmark picker describes its choices from.
  benchmarks?: Benchmark[];
  onChange: (values: FormValues) => void;
};

/** A section's fields in a grid, its subsections below, and its more settings or entries last. */
export default function FormSectionView({
  section,
  values,
  errors,
  catalog,
  benchmarks = [],
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
  // Forgets the edit, so the field shows its default again.
  const resetField = (name: string) => {
    const { [name]: _, ...fields } = values.fields;
    onChange({ ...values, fields });
  };
  const fieldInput = (field: FormField) => (
    <FieldInput
      key={field.name}
      field={field}
      value={fieldValue(field, values)}
      errors={errors[field.name]}
      catalog={catalog}
      benchmarks={benchmarks}
      onChange={(value) => setField(field.name, value)}
      onReset={field.name in values.fields ? () => resetField(field.name) : undefined}
    />
  );

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
      {editable.length > 0 && <div className="field-grid">{editable.map(fieldInput)}</div>}
      {section.allSettings?.length ? (
        <AllSettings section={section} errors={errors} fieldInput={fieldInput}>
          <MoreSettings section={section} values={values} errors={errors} onChange={onChange} />
        </AllSettings>
      ) : (
        section.moreSettings && (
          <MoreSettings section={section} values={values} errors={errors} onChange={onChange} />
        )
      )}
      {section.entry && (
        <Entries section={section} values={values} catalog={catalog} onChange={onChange} />
      )}
      {/* A list's items (e.g. Phase 1, Phase 2) show directly, unless it is switched on. */}
      {subsections.map((subsection) =>
        subsection.list && !subsection.optional ? listItems(subsection) : boxed(subsection),
      )}
    </>
  );

  // A subsection in a box titled with its name, and its switch if it is optional.
  function boxed(subsection: FormSection, action?: ReactNode) {
    const switchedOn = isSwitchedOn(subsection, values) || !!subsection.entry;
    return (
      <fieldset key={subsection.name} className="subsection">
        <legend>
          <SectionTitle section={subsection} />
          {subsection.optional && !subsection.entry && (
            <label className="stage-switch">
              <input
                type="checkbox"
                checked={switchedOn}
                onChange={(event) =>
                  setField(subsection.name, event.target.checked ? SWITCHED_ON : "")
                }
              />
              Use
            </label>
          )}
          {action}
        </legend>
        {switchedOn &&
          (subsection.list ? (
            listItems(subsection)
          ) : (
            <FormSectionView
              section={subsection}
              values={values}
              errors={errors}
              catalog={catalog}
              benchmarks={benchmarks}
              onChange={onChange}
            />
          ))}
      </fieldset>
    );
  }

  // Each item of the list with its Delete, while more than the fewest are left, and Add last.
  function listItems(list: FormSection) {
    const { item, minItems } = list.list!;
    const count = list.children.length;
    return (
      <Fragment key={list.name}>
        {list.children.map((child, index) =>
          boxed(
            child as FormSection,
            <button
              type="button"
              className="icon-button legend-action"
              aria-label={`Delete ${item} ${index + 1}`}
              title={count <= minItems ? `A request has at least ${minItems}` : undefined}
              disabled={count <= minItems}
              onClick={() => onChange(withoutListItem(values, list.name, index, count))}
            >
              <Trash2 size={16} />
            </button>,
          ),
        )}
        <button
          type="button"
          className="button ghost small"
          onClick={() => onChange(withListItem(values, list.name, count))}
        >
          <Plus size={14} /> Add {item.toLowerCase()}
        </button>
      </Fragment>
    );
  }
}

function FieldInput({
  field,
  value,
  errors,
  catalog,
  benchmarks,
  onChange,
  onReset,
}: {
  field: FormField;
  value: string;
  errors?: string[];
  catalog: Catalog;
  benchmarks: Benchmark[];
  onChange: (value: string) => void;
  // Given once the field was edited: puts its default back, or empties it if it has none.
  onReset?: () => void;
}) {
  if (field.kind === "choices")
    return <ChoicesInput {...{ field, value, errors, benchmarks, onChange }} />;
  const prefix = Object.keys(REFERENCE_PLACEHOLDERS).find((start) =>
    field.pattern?.startsWith(start),
  );
  const common = {
    id: field.name,
    name: field.name,
    value,
    "aria-invalid": !!errors,
    "aria-describedby":
      [field.description && `${field.name}-info`, errors && `${field.name}-error`]
        .filter(Boolean)
        .join(" ") || undefined,
    onChange: (event: { target: { value: string } }) => onChange(event.target.value),
  };

  return (
    <div className={field.kind === "code" ? "field wide" : "field"}>
      <div className="field-heading">
        <label className="field-label" htmlFor={field.name}>
          {field.title} <code>{lastPart(field.name)}</code>
        </label>
        {field.description && <InfoBox id={`${field.name}-info`} text={field.description} />}
        {!!field.rowFormats?.length && (
          <InfoBox
            id={`${field.name}-formats`}
            label="Row formats"
            text={<RowFormats formats={field.rowFormats} />}
          />
        )}
        {onReset && (
          <button
            type="button"
            className="reset-button"
            aria-label={field.default ? `Back to ${field.default}` : `Clear ${field.title}`}
            title={field.default ? `Back to ${field.default}` : "Clear"}
            onClick={onReset}
          >
            <X size={14} />
          </button>
        )}
      </div>
      {field.kind === "model" ? (
        <ModelPicker
          id={field.name}
          label={field.title}
          value={value}
          references={field.references ?? []}
          catalog={catalog}
          invalid={!!errors}
          describedBy={common["aria-describedby"]}
          onChange={onChange}
        />
      ) : field.kind === "dataset" ? (
        <DatasetPicker
          id={field.name}
          label={field.title}
          value={value}
          references={field.references ?? []}
          rowFormats={(field.rowFormats ?? []).map((format) => format.name)}
          catalog={catalog}
          invalid={!!errors}
          describedBy={common["aria-describedby"]}
          onChange={onChange}
        />
      ) : field.kind === "code" ? (
        <textarea {...common} className="code-input" rows={3} spellCheck={false} />
      ) : field.kind === "choice" ? (
        <select {...common}>
          <option value="">
            {field.placeholder ? `${field.placeholder} (default)` : "Choose…"}
          </option>
          {field.choices?.map((choice) => (
            <option key={String(choice)}>{String(choice)}</option>
          ))}
        </select>
      ) : field.kind === "integer" || field.kind === "number" ? (
        <NumberInput
          {...common}
          label={field.title}
          from={field.default ?? field.placeholder}
          bounds={field.bounds}
          integer={field.kind === "integer"}
          placeholder={field.placeholder}
          onChange={onChange}
        />
      ) : (
        <input
          {...common}
          placeholder={
            field.placeholder ??
            (field.kind === "list"
              ? "one value, or several separated by commas"
              : prefix && REFERENCE_PLACEHOLDERS[prefix])
          }
          autoComplete="off"
        />
      )}
      <FieldErrors id={`${field.name}-error`} messages={errors} />
    </div>
  );
}

// The row formats a Dataset field reads, each with a line of JSONL.
function RowFormats({ formats }: { formats: { name: string; example: string }[] }) {
  return (
    <>
      One JSON object per line, in one of these row formats:
      {formats.map((format) => (
        <span key={format.name} className="row-format">
          <strong>{format.name}</strong>
          <code>{format.example}</code>
        </span>
      ))}
    </>
  );
}

/** A section's title, with its description as an infobox. */
export function SectionTitle({ section }: { section: FormSection }) {
  return (
    <>
      {section.title}
      {section.description && (
        <InfoBox id={`${section.name || "request"}-info`} text={section.description} />
      )}
    </>
  );
}

// A checkbox per choice, grouped by category, with what the catalog says about each benchmark.
function ChoicesInput({
  field,
  value,
  errors,
  benchmarks,
  onChange,
}: {
  field: FormField;
  value: string;
  errors?: string[];
  benchmarks: Benchmark[];
  onChange: (value: string) => void;
}) {
  const choices = (field.choices ?? []).map(String);
  const chosen = value ? value.split(",") : [];
  const about = Object.fromEntries(benchmarks.map((benchmark) => [benchmark.name, benchmark]));
  const categories = new Map<string, string[]>();
  for (const choice of choices) {
    const category = about[choice]?.category ?? "";
    categories.set(category, [...(categories.get(category) ?? []), choice]);
  }
  // Kept in the order of the choices, whichever is clicked first.
  const toggle = (choice: string) =>
    onChange(
      choices.filter((c) => (c === choice ? !chosen.includes(c) : chosen.includes(c))).join(","),
    );

  return (
    <fieldset className="field wide choices" aria-describedby={errors && `${field.name}-error`}>
      <legend className="field-label">
        {field.title} <code>{lastPart(field.name)}</code>
      </legend>
      {[...categories].map(([category, inCategory]) => (
        <div key={category} className="choice-group">
          {category && <span className="chip">{category}</span>}
          {inCategory.map((choice) => (
            <label key={choice} className="choice">
              <input
                type="checkbox"
                checked={chosen.includes(choice)}
                aria-describedby={about[choice] && `${field.name}-${choice}-about`}
                onChange={() => toggle(choice)}
              />
              <code>{choice}</code>
              {about[choice] && (
                <>
                  <span className="muted" id={`${field.name}-${choice}-about`}>
                    {about[choice].description}
                  </span>
                  <span className="muted">{formatBytes(about[choice].size_bytes)}</span>
                </>
              )}
            </label>
          ))}
        </div>
      ))}
      <FieldErrors id={`${field.name}-error`} messages={errors} />
    </fieldset>
  );
}

// Every other setting of the section, collapsed and filterable by name, with custom keys last;
// open while one of them has an error.
function AllSettings({
  section,
  errors,
  fieldInput,
  children,
}: {
  section: FormSection;
  errors: Record<string, string[]>;
  fieldInput: (field: FormField) => ReactNode;
  children: ReactNode;
}) {
  const [filter, setFilter] = useState("");
  const settings = section.allSettings ?? [];
  const failing = [...settings.map((field) => field.name), moreName(section.name)];
  const shown = settings.filter((field) =>
    lastPart(field.name).includes(filter.trim().toLowerCase().replaceAll(" ", "_")),
  );

  return (
    <details className="all-settings" open={failing.some((name) => errors[name]) || undefined}>
      <summary>
        All settings <span className="muted">{settings.length}</span>
      </summary>
      <input
        type="search"
        className="settings-filter"
        aria-label={`Filter ${section.title.toLowerCase()}`}
        placeholder="Filter by name"
        value={filter}
        onChange={(event) => setFilter(event.target.value)}
      />
      <div className="field-grid">{shown.map(fieldInput)}</div>
      {children}
    </details>
  );
}

// Settings the schema allows beyond its fields; values read as JSON when they can (0.1, true, [..]).
function MoreSettings({
  section,
  values,
  errors,
  onChange,
}: Omit<Props, "catalog" | "benchmarks">) {
  const rows = values.more[section.name] ?? [];
  const setRows = (next: typeof rows) =>
    onChange({ ...values, more: { ...values.more, [section.name]: next } });

  return (
    <div className="more-settings">
      <p className="field-label">Custom settings</p>
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

// A map's entries, e.g. rewards: each a name and the fields of the map's values.
function Entries({
  section,
  values,
  catalog,
  onChange,
}: Pick<Props, "section" | "values" | "catalog" | "onChange">) {
  const entry = section.entry!;
  const rows = values.entries?.[section.name] ?? [];
  const setRows = (next: Entry[]) =>
    onChange({ ...values, entries: { ...values.entries, [section.name]: next } });
  const setRow = (index: number, row: Entry) =>
    setRows(rows.map((r, i) => (i === index ? row : r)));

  return (
    <div className="entries">
      {rows.map((row, index) => {
        const name = `${section.name}.${index}`;
        return (
          // biome-ignore lint/suspicious/noArrayIndexKey: entries are controlled inputs without an identity.
          <fieldset key={index} className="subsection">
            <legend>{row.name || `${entry.title} ${index + 1}`}</legend>
            <div className="field-grid">
              <div className="field">
                <label className="field-label" htmlFor={`${name}.name`}>
                  {entry.title} name
                </label>
                <input
                  id={`${name}.name`}
                  value={row.name}
                  autoComplete="off"
                  onChange={(event) => setRow(index, { ...row, name: event.target.value })}
                />
              </div>
              {(entry.children as FormField[]).map((field) => (
                <FieldInput
                  key={field.name}
                  field={{ ...field, name: `${name}.${field.name}` }}
                  value={row.fields[field.name] ?? ""}
                  catalog={catalog}
                  benchmarks={[]}
                  onChange={(value) =>
                    setRow(index, { ...row, fields: { ...row.fields, [field.name]: value } })
                  }
                />
              ))}
            </div>
            <button
              type="button"
              className="button ghost small"
              onClick={() => setRows(rows.filter((_, i) => i !== index))}
            >
              <Trash2 size={14} /> Remove {entry.title.toLowerCase()}
            </button>
          </fieldset>
        );
      })}
      <button
        type="button"
        className="button ghost small"
        onClick={() => setRows([...rows, { name: "", fields: newEntryFields(entry) }])}
      >
        <Plus size={14} /> Add {entry.title.toLowerCase()}
      </button>
    </div>
  );
}

// A code field, e.g. a reward's source, starts from a template that explains what to write.
function newEntryFields(entry: FormSection): Record<string, string> {
  const code = (entry.children as FormField[]).filter((field) => field.kind === "code");
  return Object.fromEntries(code.map((field) => [field.name, REWARD_TEMPLATE]));
}

function FieldErrors({ id, messages }: { id?: string; messages?: string[] }) {
  if (!messages?.length) return null;
  return (
    <p className="field-error" id={id} role="alert">
      {messages.join("; ")}
    </p>
  );
}

function lastPart(name: string): string {
  return name.split(".").pop()!;
}
