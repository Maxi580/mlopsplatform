import { MORE_SETTINGS, SWITCHED_ON } from "../config";

export type FieldKind =
  | "fixed"
  | "choice"
  | "choices"
  | "text"
  | "code"
  | "list"
  | "integer"
  | "number";

export type FormSection = {
  kind: "section";
  name: string;
  title: string;
  children: FormNode[];
  // Whether the section takes settings beyond its fields, e.g. any SFTConfig field.
  moreSettings: boolean;
  // Whether the request may leave it out, e.g. a Stage; it is in once switched on.
  optional: boolean;
  // Shown as an infobox beside its title, e.g. how rewards work.
  description?: string;
  // A map's entries, each a name and these fields, e.g. a Phase's rewards; in once one is named.
  entry?: FormSection;
};

export type FormField = {
  kind: FieldKind;
  name: string;
  title: string;
  fixed?: unknown;
  choices?: unknown[];
  pattern?: string;
  // Shown as an infobox beside the field, e.g. what `assistant_only_loss` does.
  description?: string;
};

export type FormNode = FormSection | FormField;
export type MoreSetting = { key: string; value: string };
export type Entry = { name: string; fields: Record<string, string> };
// Field values by dotted name, more settings and a map's entries by section name.
export type FormValues = {
  fields: Record<string, string>;
  more: Record<string, MoreSetting[]>;
  entries?: Record<string, Entry[]>;
};
// An error of the Pipeline Request at its path, as the API reports it.
export type FieldError = { loc: (string | number)[]; msg: string };

type Schema = Record<string, any>;

/** The form for a Pipeline Request: a section per object of the published schema. */
export function pipelineForm(schema: Schema): FormSection {
  return nodeOf(schema, schema.$defs ?? {}, "", "Pipeline Request") as FormSection;
}

/** The Pipeline Request the form's values describe; empty values are left out, sections never. */
export function pipelineRequestFromForm(
  form: FormSection,
  values: FormValues,
): Record<string, unknown> {
  const tree: Record<string, any> = {};
  for (const node of nodesOf(form, values)) {
    if (node.kind === "section" && node.entry) {
      for (const entry of namedEntries(node, values)) {
        for (const field of node.entry.children as FormField[]) {
          const value = (entry.fields[field.name] ?? "").trim();
          const path = [...parts(node.name), entry.name.trim(), field.name];
          if (value) put(tree, path, readValue(field, value));
        }
      }
    } else if (node.kind === "section") {
      if (node.name) put(tree, parts(node.name), get(tree, parts(node.name)) ?? {});
      for (const { key, value } of values.more[node.name] ?? []) {
        if (key.trim()) put(tree, [...parts(node.name), key.trim()], readSetting(value));
      }
    } else if (node.kind === "fixed") {
      put(tree, parts(node.name), node.fixed);
    } else {
      const value = (values.fields[node.name] ?? "").trim();
      if (value) put(tree, parts(node.name), readValue(node, value));
    }
  }
  return asLists(tree);
}

/** Each error by the name of the field, or section's more settings, that shows it. */
export function placeErrors(form: FormSection, errors: FieldError[]) {
  const names = new Set(
    nodesOf(form).flatMap((node) =>
      node.kind === "section" && node.moreSettings ? [node.name, moreName(node.name)] : [node.name],
    ),
  );
  const byName: Record<string, string[]> = {};
  const unplaced: string[] = [];
  for (const error of errors) {
    const loc = error.loc.map(String);
    const [name, rest] = nearestName(names, loc);
    const message = rest.length ? `${rest.join(".")}: ${error.msg}` : error.msg;
    if (name === null) unplaced.push(message);
    else byName[name] = [...(byName[name] ?? []), message];
  }
  return { byName, unplaced };
}

export function isSwitchedOn(section: FormSection, values: FormValues): boolean {
  if (section.entry) return namedEntries(section, values).length > 0;
  return !section.optional || values.fields[section.name] === SWITCHED_ON;
}

export function namedEntries(section: FormSection, values: FormValues): Entry[] {
  return (values.entries?.[section.name] ?? []).filter((entry) => entry.name.trim());
}

export function moreName(section: string): string {
  return `${section}.${MORE_SETTINGS}`;
}

function nodeOf(schema: Schema, defs: Schema, name: string, title: string): FormNode {
  // An optional value (`X | None`) shows as X; left empty, it is left out.
  const options = schema.anyOf?.filter((option: Schema) => option.type !== "null") ?? [];
  const optional = options.length === 1 && schema.anyOf.length === 2;
  const node = resolveRef(optional ? options[0] : schema, defs);
  const types = new Set((node.anyOf ?? [node]).map((option: Schema) => option.type));
  // A field's own title, not its type's (e.g. "Settings", not "SftSettings").
  title = schema.title ?? humanize(title);
  const child = (key: string) => (name ? `${name}.${key}` : key);
  if ("const" in node) return { kind: "fixed", name, title, fixed: node.const };
  const description = schema.description ?? node.description;
  // A map of objects, e.g. rewards by name: entries of the value's fields. Pydantic writes a
  // map with a key pattern as patternProperties.
  const values =
    typeof node.additionalProperties === "object"
      ? node.additionalProperties
      : Object.values(node.patternProperties ?? {})[0];
  if (types.has("object") && values) {
    const entry = nodeOf(values as Schema, defs, "", title.replace(/s$/, ""));
    return {
      kind: "section",
      name,
      title,
      children: [],
      moreSettings: false,
      optional,
      entry: entry as FormSection,
      ...(description && { description }),
    };
  }
  if (types.has("object")) {
    const children = Object.entries(node.properties ?? {}).map(([key, value]) =>
      nodeOf(value as Schema, defs, child(key), key),
    );
    return {
      kind: "section",
      name,
      title,
      children,
      moreSettings: node.additionalProperties === true,
      optional,
      ...(description && { description }),
    };
  }
  if (node.type === "array" && resolveRef(node.items, defs).type === "object") {
    const children = Array.from({ length: node.minItems ?? 1 }, (_, index) =>
      nodeOf(node.items, defs, child(String(index)), `${title.replace(/s$/, "")} ${index + 1}`),
    );
    return { kind: "section", name, title, children, moreSettings: false, optional };
  }
  return { ...fieldOf(node, defs, types), name, title, ...(description && { description }) };
}

function fieldOf(node: Schema, defs: Schema, types: Set<unknown>) {
  if ("enum" in node) return { kind: "choice" as const, choices: node.enum };
  // A list of values from a fixed set, such as benchmarks from the catalog.
  const items = node.type === "array" ? resolveRef(node.items, defs) : undefined;
  if (items?.enum) return { kind: "choices" as const, choices: items.enum };
  if (types.has("boolean")) return { kind: "choice" as const, choices: [true, false] };
  if (types.has("array")) return { kind: "list" as const };
  if (types.has("integer")) return { kind: "integer" as const };
  if (types.has("number")) return { kind: "number" as const };
  if (node.format === "python") return { kind: "code" as const };
  // An optional value's pattern sits on its non-null option.
  const pattern = node.pattern ?? node.anyOf?.find((option: Schema) => option.pattern)?.pattern;
  return { kind: "text" as const, pattern };
}

function humanize(key: string): string {
  const words = key.replaceAll("_", " ");
  return words.charAt(0).toUpperCase() + words.slice(1);
}

function resolveRef(node: Schema, defs: Schema): Schema {
  return node.$ref ? defs[node.$ref.split("/").pop()] : node;
}

// With values, only the sections switched on and their fields.
function nodesOf(node: FormNode, values?: FormValues): FormNode[] {
  if (node.kind !== "section") return [node];
  if (values && !isSwitchedOn(node, values)) return [];
  return [node, ...node.children.flatMap((child) => nodesOf(child, values))];
}

function nearestName(names: Set<string>, loc: string[]): [string | null, string[]] {
  for (let length = loc.length; length > 0; length--) {
    const name = loc.slice(0, length).join(".");
    if (length < loc.length && names.has(moreName(name)))
      return [moreName(name), loc.slice(length)];
    if (names.has(name)) return [name, loc.slice(length)];
  }
  return [null, loc];
}

// Unreadable numbers stay text, so validation reports them next to their field.
function readValue(field: FormField, text: string): unknown {
  if (field.kind === "integer" || field.kind === "number") {
    const number = Number(text);
    return Number.isNaN(number) || (field.kind === "integer" && !Number.isInteger(number))
      ? text
      : number;
  }
  if (field.kind === "choice")
    return field.choices?.find((choice) => String(choice) === text) ?? text;
  if (field.kind === "choices") return text.split(",");
  if (field.kind === "list" && text.includes(","))
    return text.split(",").map((part) => part.trim());
  return text;
}

// A more setting is JSON when it reads as JSON (numbers, true, lists), else text.
function readSetting(text: string): unknown {
  try {
    return JSON.parse(text);
  } catch {
    return text;
  }
}

function parts(name: string): string[] {
  return name ? name.split(".") : [];
}

function put(tree: Record<string, any>, path: string[], value: unknown) {
  const last = path.pop()!;
  for (const part of path) tree = tree[part] ??= {};
  tree[last] = value;
}

function get(tree: Record<string, any>, path: string[]): unknown {
  return path.reduce<any>((node, part) => node?.[part], tree);
}

function asLists(node: any): any {
  if (node === null || typeof node !== "object" || Array.isArray(node)) return node;
  const keys = Object.keys(node);
  if (keys.length && keys.every((key) => /^\d+$/.test(key))) {
    return keys.sort((a, b) => Number(a) - Number(b)).map((key) => asLists(node[key]));
  }
  return Object.fromEntries(keys.map((key) => [key, asLists(node[key])]));
}
