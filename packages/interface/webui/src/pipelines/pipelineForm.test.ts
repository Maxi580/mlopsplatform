import {
  type FormNode,
  type FormValues,
  fieldValue,
  pipelineForm,
  pipelineRequestFromForm,
  placeErrors,
} from "./pipelineForm";

// A trimmed copy of `GET /schema`, with each shape the form must handle.
const schema = {
  type: "object",
  required: ["name", "finetune"],
  properties: {
    schema_version: { const: 1, default: 1, type: "integer", title: "Schema Version" },
    name: { type: "string", title: "Name" },
    finetune: { $ref: "#/$defs/Finetune" },
    serve: { anyOf: [{ $ref: "#/$defs/Serve" }, { type: "null" }], default: null },
  },
  $defs: {
    Serve: {
      type: "object",
      title: "Serve",
      properties: {
        prefix_caching: { type: "boolean", default: true, title: "Prefix Caching" },
        dtype: { anyOf: [{ enum: ["auto", "half"], type: "string" }, { type: "null" }] },
      },
    },
    Finetune: {
      type: "object",
      properties: {
        base_model: {
          type: "string",
          pattern: "^hf:[\\w.-]+/[\\w.-]+(@[\\w.-]+)?$",
          title: "Base Model",
        },
        from: {
          anyOf: [
            { type: "string", pattern: "^model:[a-z0-9][a-z0-9.-]*(@\\d+)?$" },
            { type: "null" },
          ],
          default: null,
          title: "From Model Version",
        },
        backend: { enum: ["hf", "unsloth"], type: "string", title: "Backend" },
        phases: {
          type: "array",
          minItems: 1,
          items: { $ref: "#/$defs/SftPhase" },
          title: "Phases",
        },
      },
    },
    SftPhase: {
      type: "object",
      properties: {
        algorithm: { const: "sft", type: "string", title: "Algorithm" },
        settings: { $ref: "#/$defs/SftSettings" },
        rewards: {
          anyOf: [
            { type: "object", patternProperties: { "^[\\w-]{1,64}$": { $ref: "#/$defs/Reward" } } },
            { type: "null" },
          ],
          default: null,
          title: "Rewards",
          description: "How rewards work.",
        },
      },
    },
    Reward: {
      type: "object",
      additionalProperties: false,
      properties: {
        weight: { type: "number", title: "Weight" },
        source: { type: "string", format: "python", maxLength: 65536, title: "Source" },
      },
    },
    SftSettings: {
      type: "object",
      additionalProperties: true,
      properties: {
        learning_rate: { type: "number", title: "Learning Rate" },
        max_length: { type: "integer", title: "Max Length" },
        target_modules: {
          anyOf: [{ type: "string" }, { type: "array", items: { type: "string" } }],
        },
      },
    },
  },
};

function flatten(node: FormNode): FormNode[] {
  return node.kind === "section" ? [node, ...node.children.flatMap(flatten)] : [node];
}

const form = pipelineForm(schema);
const byName = Object.fromEntries(flatten(form).map((node) => [node.name, node]));

test("every schema value becomes a field of the right kind", () => {
  expect(byName.name.kind).toBe("text");
  expect(byName.schema_version).toMatchObject({ kind: "fixed", fixed: 1 });
  expect(byName["finetune.backend"]).toMatchObject({ kind: "choice", choices: ["hf", "unsloth"] });
  expect(byName["finetune.phases"].kind).toBe("section");
  expect(byName["finetune.phases.0.algorithm"]).toMatchObject({ kind: "fixed", fixed: "sft" });
  expect(byName["finetune.phases.0.settings.learning_rate"].kind).toBe("number");
  expect(byName["finetune.phases.0.settings.max_length"].kind).toBe("integer");
  expect(byName["finetune.phases.0.settings.target_modules"].kind).toBe("list");
  expect(byName["finetune.phases.0.settings"]).toMatchObject({
    kind: "section",
    moreSettings: true,
  });
  expect(byName.finetune).toMatchObject({ kind: "section", moreSettings: false });
  expect(byName["finetune.from"]).toMatchObject({
    kind: "text",
    pattern: expect.stringMatching(/^\^model:/),
  });
});

test("the form's values become a Pipeline Request", () => {
  const request = pipelineRequestFromForm(form, {
    fields: {
      name: "qwen-sft",
      "finetune.base_model": "hf:Qwen/Qwen2.5-0.5B-Instruct",
      "finetune.backend": "hf",
      "finetune.phases.0.settings.learning_rate": "1e-4",
      "finetune.phases.0.settings.max_length": "1024",
      "finetune.phases.0.settings.target_modules": "q_proj, v_proj",
    },
    more: {
      "finetune.phases.0.settings": [
        { key: "weight_decay", value: "0.01" },
        { key: "packing", value: "true" },
        { key: "optim", value: "adamw_torch" },
        { key: "", value: "ignored" },
      ],
    },
  });

  expect(request).toEqual({
    schema_version: 1,
    name: "qwen-sft",
    finetune: {
      base_model: "hf:Qwen/Qwen2.5-0.5B-Instruct",
      backend: "hf",
      phases: [
        {
          algorithm: "sft",
          settings: {
            learning_rate: 0.0001,
            max_length: 1024,
            target_modules: ["q_proj", "v_proj"],
            weight_decay: 0.01,
            packing: true,
            optim: "adamw_torch",
          },
        },
      ],
    },
  });
});

test("empty values are left out and unreadable numbers stay text, so validation names them", () => {
  const request = pipelineRequestFromForm(form, {
    fields: { name: "", "finetune.phases.0.settings.max_length": "long" },
    more: {},
  });

  expect(request).toEqual({
    schema_version: 1,
    finetune: { backend: "hf", phases: [{ algorithm: "sft", settings: { max_length: "long" } }] },
  });
});

test("each error goes to its field, a more setting's error to its section, the rest stay unplaced", () => {
  const placed = placeErrors(form, [
    { loc: ["finetune", "phases", 0, "settings", "max_length"], msg: "not an integer" },
    { loc: ["finetune", "phases", 0, "settings", "warmup_ratio"], msg: "not a SFTConfig setting" },
    { loc: ["finetune", "phases", 0, "settings", "target_modules", "str"], msg: "bad" },
    { loc: [], msg: "Kubeflow did not start Pipeline 3" },
  ]);

  expect(placed.byName).toEqual({
    "finetune.phases.0.settings.max_length": ["not an integer"],
    "finetune.phases.0.settings.*": ["warmup_ratio: not a SFTConfig setting"],
    "finetune.phases.0.settings.target_modules": ["str: bad"],
  });
  expect(placed.unplaced).toEqual(["Kubeflow did not start Pipeline 3"]);
});

test("an optional Stage is a section the request holds only once it is switched on", () => {
  expect(byName.serve).toMatchObject({ kind: "section", optional: true });
  expect(byName.finetune).toMatchObject({ kind: "section", optional: false });
  const values = { fields: { "serve.dtype": "half" }, more: {} };

  expect(pipelineRequestFromForm(form, values)).not.toHaveProperty("serve");
  const switchedOn = { ...values, fields: { ...values.fields, serve: "on" } };
  expect(pipelineRequestFromForm(form, switchedOn).serve).toEqual({
    prefix_caching: true,
    dtype: "half",
  });
});

test("booleans and optional choices are choices, read back as their values", () => {
  expect(byName["serve.prefix_caching"]).toMatchObject({ kind: "choice", choices: [true, false] });
  expect(byName["serve.dtype"]).toMatchObject({ kind: "choice", choices: ["auto", "half"] });
  const values = { fields: { serve: "on", "serve.prefix_caching": "false" }, more: {} };

  expect(pipelineRequestFromForm(form, values).serve).toEqual({ prefix_caching: false });
});

test("a map of objects is a section of named entries; Python source is code", () => {
  expect(byName["finetune.phases.0.rewards"]).toMatchObject({
    kind: "section",
    title: "Rewards",
    description: "How rewards work.",
    entry: {
      title: "Reward",
      children: [
        { kind: "number", name: "weight" },
        { kind: "code", name: "source" },
      ],
    },
  });
});

test("named entries become the map; unnamed ones and an empty map are left out", () => {
  const source = "def reward(sample, item):\n    return 1.0";
  const entries = {
    "finetune.phases.0.rewards": [
      { name: " correct ", fields: { weight: "0.8", source } },
      { name: "", fields: { weight: "1" } },
    ],
  };

  const phase = (values: object) =>
    (
      pipelineRequestFromForm(form, { fields: {}, more: {}, ...values }).finetune as {
        phases: Record<string, unknown>[];
      }
    ).phases[0];

  expect(phase({ entries }).rewards).toEqual({ correct: { weight: 0.8, source } });
  expect(phase({})).not.toHaveProperty("rewards");
});

test("an entry's error goes to its map with the entry's name", () => {
  const placed = placeErrors(form, [
    { loc: ["finetune", "phases", 0, "rewards", "correct", "source"], msg: "is no Python" },
  ]);

  expect(placed.byName).toEqual({ "finetune.phases.0.rewards": ["correct.source: is no Python"] });
});

// A trimmed copy of the published schema's Sweep, with two algorithms' settings and defaults.
const sweepSchema = {
  type: "object",
  properties: { sweep: { $ref: "#/$defs/Sweep" } },
  $defs: {
    Sweep: {
      type: "object",
      properties: {
        algorithm: { enum: ["sft", "dpo"], type: "string" },
        settings: { $ref: "#/$defs/PhaseSettings", trainer_settings: "settings" },
        trials: { type: "integer", default: 10, exclusiveMinimum: 0 },
        objective: {
          anyOf: [{ $ref: "#/$defs/Objective" }, { type: "null" }],
          default: null,
          algorithm_defaults: "objective",
        },
      },
    },
    PhaseSettings: { type: "object", additionalProperties: true, properties: {} },
    Objective: {
      type: "object",
      properties: {
        metric: { type: "string" },
        goal: { enum: ["minimize", "maximize"], type: "string" },
      },
    },
  },
  algorithms: {
    sft: {
      objective: { metric: "eval_loss", goal: "minimize" },
      settings: {
        learning_rate: { type: "number", default: 2e-4 },
        num_train_epochs: { type: "number", default: 3 },
        weight_decay: { type: "number", placeholder: 0 },
      },
    },
    dpo: {
      objective: { metric: "eval_rewards/accuracies", goal: "maximize" },
      settings: {
        learning_rate: { type: "number", default: 5e-6 },
        num_train_epochs: { type: "number", default: 1 },
        beta: { type: "number", default: 0.1 },
      },
    },
  },
};

function fieldsOf(values: FormValues): Record<string, string> {
  const nodes = flatten(pipelineForm(sweepSchema, values)).filter(
    (node) => node.kind !== "section",
  );
  return Object.fromEntries(nodes.map((node) => [node.name, fieldValue(node as never, values)]));
}

test("fields start with their defaults, which the request holds until edited", () => {
  const values = { fields: {}, more: {} };

  expect(fieldsOf(values)).toMatchObject({
    "sweep.algorithm": "sft",
    "sweep.trials": "10",
    "sweep.settings.learning_rate": "2e-4",
    "sweep.settings.weight_decay": "",
  });
  expect(pipelineRequestFromForm(pipelineForm(sweepSchema, values), values)).toEqual({
    sweep: {
      algorithm: "sft",
      trials: 10,
      settings: { learning_rate: 0.0002, num_train_epochs: 3 },
    },
  });
});

test("a setting TRL defaults itself shows that default greyed, and isn't sent", () => {
  const form = pipelineForm(sweepSchema);
  const weightDecay = flatten(form).find((node) => node.name === "sweep.settings.weight_decay");

  expect(weightDecay).toMatchObject({ placeholder: "0" });
  expect(weightDecay).not.toHaveProperty("default");
});

test("switching the algorithm changes the untouched settings and keeps the edited ones", () => {
  const edited = { "sweep.settings.num_train_epochs": "2" };
  const values = { fields: { ...edited, "sweep.algorithm": "dpo" }, more: {} };

  expect(fieldsOf(values)).toMatchObject({
    "sweep.settings.learning_rate": "5e-6",
    "sweep.settings.num_train_epochs": "2",
    "sweep.settings.beta": "0.1",
  });
  expect(fieldsOf(values)).not.toHaveProperty("sweep.settings.weight_decay");
});

test("a Sweep's objective defaults to its algorithm's", () => {
  const values = { fields: { "sweep.algorithm": "dpo", "sweep.objective": "on" }, more: {} };

  expect(fieldsOf(values)).toMatchObject({
    "sweep.objective.metric": "eval_rewards/accuracies",
    "sweep.objective.goal": "maximize",
  });
});
