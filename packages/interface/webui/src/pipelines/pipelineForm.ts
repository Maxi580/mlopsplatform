import { MORE_SETTINGS, SWITCHED_ON } from "../config";
import { formatNumber } from "../fields/numbers";

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
  // Every other setting it takes, shown one click away, e.g. the rest of SFTConfig.
  allSettings?: FormField[];
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
  // Shown until the user edits the field, and sent; × puts it back.
  default?: string;
  // Shown greyed in an empty field and not sent, e.g. the default TRL itself applies.
  placeholder?: string;
};

export type FormNode = FormSection | FormField;
export type MoreSetting = { key: string; value: string };
export type Entry = { name: string; fields: Record<string, string> };
// The user's edits by dotted field name (a field left alone shows its default), more settings
// and a map's entries by section name.
export type FormValues = {
  fields: Record<string, string>;
  more: Record<string, MoreSetting[]>;
  entries?: Record<string, Entry[]>;
};
// An error of the Pipeline Request at its path, as the API reports it.
export type FieldError = { loc: (string | number)[]; msg: string };

type Schema = Record<string, any>;
// What building a node needs besides its own schema: the published schema and the values, as
// some defaults depend on other fields, e.g. a Phase's settings on its algorithm, and the
// defaults its section's algorithm gives its fields, e.g. a Sweep's objective.
type Context = { root: Schema; defs: Schema; values: FormValues; defaults?: Schema };

const NO_VALUES: FormValues = { fields: {}, more: {} };

/** The form for a Pipeline Request with these values: a section per object of the schema. */
export function pipelineForm(schema: Schema, values: FormValues = NO_VALUES): FormSection {
  const context = { root: schema, defs: schema.$defs ?? {}, values };
  return nodeOf(schema, context, "", "Pipeline Request") as FormSection;
}

/** What a field shows and sends: the user's edit, else its default. */
export function fieldValue(field: FormField, values: FormValues): string {
  return values.fields[field.name] ?? field.default ?? "";
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
      const value = fieldValue(node, values).trim();
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

function nodeOf(schema: Schema, context: Context, name: string, title: string): FormNode {
  // An optional value (`X | None`) shows as X; left empty, it is left out.
  const options = schema.anyOf?.filter((option: Schema) => option.type !== "null") ?? [];
  const optional = options.length === 1 && schema.anyOf.length === 2;
  const node = resolveRef(optional ? options[0] : schema, context.defs);
  const types = new Set((node.anyOf ?? [node]).map((option: Schema) => option.type));
  // A field's own title, not its type's (e.g. "Settings", not "SftSettings").
  title = schema.title ?? humanize(title);
  const child = (key: string) => (name ? `${name}.${key}` : key);
  if ("const" in node) return { kind: "fixed", name, title, fixed: node.const };
  const description = schema.description ?? node.description;
  // The settings of a Phase's algorithm, or of its Adapter, as the API publishes them: the common
  // ones shown, every other one a click away; sent whenever they are shown.
  if (schema.trainer_settings) {
    const algorithm = context.root.algorithms?.[algorithmAt(context, parent(name))];
    const fieldsOf = (settings: Schema = {}) =>
      Object.entries(settings).map(([key, value]) =>
        nodeOf(value as Schema, context, child(key), key),
      );
    const more = `more_${schema.trainer_settings}`;
    return {
      kind: "section",
      name,
      title,
      children: fieldsOf(algorithm?.[schema.trainer_settings]),
      allSettings: fieldsOf(algorithm?.[more] ?? context.root[more]) as FormField[],
      moreSettings: true,
      optional: false,
      ...(description && { description }),
    };
  }
  // A map of objects, e.g. rewards by name: entries of the value's fields. Pydantic writes a
  // map with a key pattern as patternProperties.
  const values =
    typeof node.additionalProperties === "object"
      ? node.additionalProperties
      : Object.values(node.patternProperties ?? {})[0];
  if (types.has("object") && values) {
    const entry = nodeOf(values as Schema, context, "", title.replace(/s$/, ""));
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
    // E.g. a Sweep's objective, whose defaults its algorithm publishes as `objective`.
    const algorithm = context.root.algorithms?.[algorithmAt(context, parent(name))];
    const inner = { ...context, defaults: algorithm?.[schema.algorithm_defaults] };
    // Only the fields that apply, e.g. a Teacher only where the algorithm learns from one.
    const children = Object.entries(node.properties ?? {})
      .filter(([, value]) => {
        const trait = (value as Schema).applies_if;
        return !trait || traitsOf(node, context, name)[trait];
      })
      .map(([key, value]) => nodeOf(value as Schema, inner, child(key), key));
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
  if (node.type === "array" && resolveRef(node.items, context.defs).type === "object") {
    const children = Array.from({ length: node.minItems ?? 1 }, (_, index) =>
      nodeOf(node.items, context, child(String(index)), `${title.replace(/s$/, "")} ${index + 1}`),
    );
    return { kind: "section", name, title, children, moreSettings: false, optional };
  }
  const field = { ...fieldOf(node, context.defs, types), name, title };
  const fallback = defaultOf(schema, node, field, context.defaults?.[lastPart(name)]);
  return {
    ...field,
    ...(description && { description }),
    ...(fallback !== undefined && { default: fallback }),
    ...(schema.placeholder != null && { placeholder: formatValue(schema.placeholder) }),
  };
}

function fieldOf(node: Schema, defs: Schema, types: Set<unknown>) {
  // A choice, also where any other text is allowed too, e.g. TRL's `lr_scheduler_type`.
  const choices = node.enum ?? node.anyOf?.find((option: Schema) => option.enum)?.enum;
  if (choices) return { kind: "choice" as const, choices };
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

/** What the field shows until edited: its algorithm's default, the schema's, or for a value
 * that must be chosen, its first choice. */
function defaultOf(
  schema: Schema,
  node: Schema,
  field: { kind: FieldKind; choices?: unknown[] },
  algorithmDefault: unknown,
): string | undefined {
  const value = algorithmDefault ?? schema.default ?? node.default;
  if (value !== undefined && value !== null) return formatValue(value);
  // A value the request must name, e.g. an algorithm, not a setting TRL defaults itself.
  const mustChoose = !("default" in schema) && !schema.anyOf && schema.placeholder == null;
  return mustChoose && node.enum ? String(field.choices?.[0]) : undefined;
}

/** What decides which of a section's fields apply, from what the API publishes: whether its
 * algorithm learns from a Teacher or from rewards, whether its method trains an Adapter, a new one
 * or the previous Phase's, and whether its Teacher is a model at an external API. */
function traitsOf(node: Schema, context: Context, path: string): Record<string, boolean> {
  const siblingValue = (at: string, key: string) =>
    context.values.fields[`${at}.${key}`] ?? String(node.properties?.[key]?.default ?? "");
  const adapterMethods: string[] = context.root.adapter_methods ?? [];
  const trainsAdapter = (at: string) => adapterMethods.includes(siblingValue(at, "method"));
  // A Phase after one that keeps its Adapter continues that Adapter.
  const index = Number(lastPart(path));
  const previous = index > 0 ? `${parent(path)}.${index - 1}` : null;
  const continuesAdapter =
    previous !== null && trainsAdapter(previous) && siblingValue(previous, "output") === "adapter";
  const teacher = siblingValue(path, "teacher").trim();
  const references: string[] = node.properties?.teacher?.references ?? [];
  const algorithm = context.root.algorithms?.[algorithmAt(context, path)] ?? {};
  return {
    learns_from_teacher: !!algorithm.learns_from_teacher,
    learns_from_rewards: !!algorithm.learns_from_rewards,
    trains_adapter: trainsAdapter(path),
    new_adapter: trainsAdapter(path) && !continuesAdapter,
    external_teacher: !!teacher && !references.some((prefix) => teacher.startsWith(prefix)),
  };
}

// The algorithm a Phase or Sweep at `path` trains with: chosen, else the first.
function algorithmAt(context: Context, path: string): string {
  const first = Object.keys(context.root.algorithms ?? {})[0];
  return context.values.fields[path ? `${path}.algorithm` : "algorithm"] || first;
}

function formatValue(value: unknown): string {
  if (typeof value === "number") return formatNumber(value);
  if (Array.isArray(value)) return value.join(",");
  return typeof value === "object" ? JSON.stringify(value) : String(value);
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
  const children = [...node.children, ...(node.allSettings ?? [])];
  return [node, ...children.flatMap((child) => nodesOf(child, values))];
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

function parent(name: string): string {
  return name.split(".").slice(0, -1).join(".");
}

function lastPart(name: string): string {
  return name.split(".").pop() ?? "";
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
