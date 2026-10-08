import { TUNED_RANGE_SPREAD } from "../config";
import type { Choice } from "../fields/Combobox";
import { type Bounds, formatNumber } from "../fields/numbers";
import type { Entry, FormField, FormSection } from "./pipelineForm";

/** The settings a Sweep's parameter at `index` can tune; one its other parameters tune is greyed. */
export function settingChoices(parameters: FormSection, rows: Entry[], index: number): Choice[] {
  const others = rows.filter((_, at) => at !== index).map((row) => row.name.trim());
  return (parameters.tunes ?? []).map((setting) => ({
    value: lastPart(setting.name),
    label: lastPart(setting.name),
    detail: setting.default ?? setting.placeholder,
    disabled: others.includes(lastPart(setting.name)),
  }));
}

/** The parameter named `name`; naming a setting starts it from the setting's default. */
export function withTunedSetting(parameters: FormSection, row: Entry, name: string): Entry {
  const setting = tunedSetting(parameters, name);
  return setting ? { name, fields: { ...row.fields, ...tunedRange(setting) } } : { ...row, name };
}

/** The setting a Sweep's parameter of this name tunes, if any. */
export function tunedSetting(parameters: FormSection, name: string): FormField | undefined {
  return parameters.tunes?.find((setting) => lastPart(setting.name) === name.trim());
}

/** What a Sweep parameter starts with once its setting is picked: every choice as its values, or
 * a range either side of the setting's default, within its bounds; none without a default. */
export function tunedRange(setting: FormField): Record<string, string> {
  const none = { min: "", max: "", values: "" };
  if (setting.kind === "choice") return { ...none, values: setting.choices!.join(",") };
  const from = Number(setting.default ?? setting.placeholder);
  if ((setting.kind !== "integer" && setting.kind !== "number") || !from) return none;
  const [min, max] = [from * (1 - TUNED_RANGE_SPREAD), from * (1 + TUNED_RANGE_SPREAD)]
    .sort((a, b) => a - b)
    .map((value) => withinBounds(value, from, setting.bounds))
    // Rounded, so 2e-4 × 1.5 is 3e-4, not 3.0000000000000003e-4.
    .map((value) =>
      setting.kind === "integer" ? Math.round(value) : Number(value.toPrecision(12)),
    );
  return { ...none, min: formatNumber(min), max: formatNumber(max) };
}

/** A Sweep parameter's fields: a number setting's Min and Max step like it, within its bounds
 * and below and above each other. */
export function tunedParameterFields(
  fields: FormField[],
  setting: FormField | undefined,
  entry: Entry,
): FormField[] {
  if (setting?.kind !== "integer" && setting?.kind !== "number") return fields;
  const bounds = setting.bounds ?? {};
  return fields.map((field) => {
    if (field.name === "min") {
      const exclusiveMaximum = Math.min(
        numberIn(entry, "max") ?? Infinity,
        bounds.exclusiveMaximum ?? Infinity,
      );
      return { ...field, kind: setting.kind, bounds: { ...bounds, exclusiveMaximum } };
    }
    if (field.name === "max") {
      const exclusiveMinimum = Math.max(
        numberIn(entry, "min") ?? -Infinity,
        bounds.exclusiveMinimum ?? -Infinity,
      );
      return { ...field, kind: setting.kind, bounds: { ...bounds, exclusiveMinimum } };
    }
    return field;
  });
}

// An exclusive bound is no value to try, so a range end that reaches it stops halfway.
function withinBounds(value: number, from: number, bounds: Bounds = {}): number {
  const { minimum = -Infinity, maximum = Infinity, exclusiveMinimum, exclusiveMaximum } = bounds;
  const clamped = Math.min(Math.max(value, minimum), maximum);
  if (exclusiveMinimum !== undefined && clamped <= exclusiveMinimum)
    return (exclusiveMinimum + from) / 2;
  if (exclusiveMaximum !== undefined && clamped >= exclusiveMaximum)
    return (exclusiveMaximum + from) / 2;
  return clamped;
}

function numberIn(entry: Entry, name: string): number | undefined {
  const text = entry.fields[name]?.trim();
  return text && !Number.isNaN(Number(text)) ? Number(text) : undefined;
}

function lastPart(name: string): string {
  return name.split(".").pop()!;
}
