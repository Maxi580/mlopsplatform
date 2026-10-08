import {
  type FormNode,
  type FormSection,
  type FormValues,
  fieldValue,
  pipelineForm,
  pipelineRequestFromForm,
  placeErrors,
  withoutListItem,
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
  expect(byName["finetune.from"]).toMatchObject({ kind: "model", references: ["model:"] });
  expect(byName["finetune.base_model"]).toMatchObject({ kind: "model", references: ["hf:"] });
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

// A trimmed copy of the published schema's Phases and `distill`, with when each field applies.
const appliesSchema = {
  type: "object",
  properties: {
    distill: { anyOf: [{ $ref: "#/$defs/Distill" }, { type: "null" }], default: null },
    finetune: { $ref: "#/$defs/Finetune" },
  },
  $defs: {
    Distill: {
      type: "object",
      properties: {
        teacher: { type: "string", references: ["hf:", "model:", "endpoint:"] },
        api_url: {
          anyOf: [{ type: "string" }, { type: "null" }],
          default: null,
          applies_if: "external_teacher",
        },
      },
    },
    Finetune: {
      type: "object",
      properties: { phases: { type: "array", minItems: 2, items: { $ref: "#/$defs/Phase" } } },
    },
    Phase: {
      type: "object",
      properties: {
        algorithm: { enum: ["sft", "distillation", "grpo"], type: "string" },
        method: { enum: ["lora", "qlora", "full"], type: "string", default: "lora" },
        lora: {
          anyOf: [{ $ref: "#/$defs/LoraSettings" }, { type: "null" }],
          default: null,
          trainer_settings: "lora_settings",
          applies_if: "new_adapter",
        },
        teacher: {
          anyOf: [{ type: "string" }, { type: "null" }],
          default: null,
          applies_if: "learns_from_teacher",
        },
        rewards: {
          anyOf: [{ type: "object", additionalProperties: { type: "number" } }, { type: "null" }],
          default: null,
          applies_if: "learns_from_rewards",
        },
        output: {
          enum: ["adapter", "merged"],
          type: "string",
          default: "adapter",
          applies_if: "trains_adapter",
        },
      },
    },
    LoraSettings: { type: "object", additionalProperties: true, properties: {} },
  },
  adapter_methods: ["lora", "qlora"],
  algorithms: Object.fromEntries(
    [
      ["sft", false, false],
      ["distillation", true, false],
      ["grpo", false, true],
    ].map(([name, teacher, rewards]) => [
      name,
      {
        learns_from_teacher: teacher,
        learns_from_rewards: rewards,
        lora_settings: { r: { type: "integer", default: 16 } },
      },
    ]),
  ),
};

function shownNames(fields: Record<string, string>): string[] {
  return flatten(pipelineForm(appliesSchema, { fields, more: {} })).map((node) => node.name);
}

test("a Phase shows a Teacher only for distillation, and rewards only for grpo", () => {
  const sft = shownNames({});
  expect(sft).not.toContain("finetune.phases.0.teacher");
  expect(sft).not.toContain("finetune.phases.0.rewards");

  const distillation = shownNames({ "finetune.phases.0.algorithm": "distillation" });
  expect(distillation).toContain("finetune.phases.0.teacher");
  expect(distillation).not.toContain("finetune.phases.0.rewards");
  expect(shownNames({ "finetune.phases.0.algorithm": "grpo" })).toContain(
    "finetune.phases.0.rewards",
  );
});

test("full weights hide LoRA and output; a Phase continuing an Adapter has no LoRA of its own", () => {
  const shown = shownNames({});
  expect(shown).toContain("finetune.phases.0.lora.r");
  expect(shown).toContain("finetune.phases.0.output");
  expect(shown).not.toContain("finetune.phases.1.lora");
  expect(shown).toContain("finetune.phases.1.output");

  const full = shownNames({ "finetune.phases.0.method": "full" });
  expect(full).not.toContain("finetune.phases.0.lora");
  expect(full).not.toContain("finetune.phases.0.output");
  // After full weights, the next Phase trains a new Adapter.
  expect(full).toContain("finetune.phases.1.lora.r");
  expect(shownNames({ "finetune.phases.0.output": "merged" })).toContain("finetune.phases.1.lora");
});

test("a Teacher at an external API shows its URL", () => {
  expect(shownNames({ distill: "on", "distill.teacher": "hf:Qwen/Qwen3-8B" })).not.toContain(
    "distill.api_url",
  );
  expect(shownNames({ distill: "on", "distill.teacher": "gpt-4o" })).toContain("distill.api_url");
});

test("a value typed into a field that then gets hidden isn't sent", () => {
  const values = {
    fields: { "finetune.phases.0.teacher": "hf:Qwen/Qwen3-8B", "finetune.phases.0.method": "full" },
    more: {},
  };

  const request = pipelineRequestFromForm(pipelineForm(appliesSchema, values), values);

  expect((request.finetune as { phases: object[] }).phases[0]).toEqual({
    algorithm: "sft",
    method: "full",
  });
});

// A Phase whose algorithm publishes its common settings, every other one, and LoRA's.
const settingsSchema = {
  type: "object",
  properties: { phase: { $ref: "#/$defs/Phase" } },
  $defs: {
    Phase: {
      type: "object",
      properties: {
        algorithm: { enum: ["sft"], type: "string" },
        settings: { $ref: "#/$defs/PhaseSettings", trainer_settings: "settings" },
        lora: {
          anyOf: [{ $ref: "#/$defs/LoraSettings" }, { type: "null" }],
          default: null,
          trainer_settings: "lora_settings",
        },
      },
    },
    PhaseSettings: { type: "object", additionalProperties: true, properties: {} },
    LoraSettings: { type: "object", additionalProperties: true, properties: {} },
  },
  algorithms: {
    sft: {
      settings: { learning_rate: { type: "number", default: 2e-4 } },
      more_settings: {
        optim: {
          anyOf: [{ enum: ["adamw_torch", "adafactor"], type: "string" }, { type: "string" }],
          placeholder: "adamw_torch",
        },
        packing_strategy: { type: "string", placeholder: "bfd" },
        use_liger_kernel: { type: "boolean", placeholder: false },
        adam_beta1: { type: "number", placeholder: 0.9 },
        // Shapes that once reached TRL unasked: objects, defaults inside, a choice without default.
        lr_scheduler_kwargs: {
          anyOf: [{ additionalProperties: true, type: "object" }, { type: "null" }],
          placeholder: {},
        },
        accelerator_config: {
          type: "object",
          properties: { split_batches: { type: "boolean", default: false } },
        },
        report_to: { enum: ["none", "mlflow"], type: "string" },
      },
      lora_settings: { r: { type: "integer", enum: [8, 16, 32], default: 16 } },
    },
  },
  more_lora_settings: {
    use_rslora: { type: "boolean", placeholder: false },
    loftq_config: { anyOf: [{ type: "object" }, { type: "object" }], placeholder: {} },
  },
};

test("a Phase left alone sends only its shown settings, none of All settings", () => {
  const values = { fields: {}, more: {} };
  const form = pipelineForm(settingsSchema, values);

  expect(pipelineRequestFromForm(form, values)).toEqual({
    phase: { algorithm: "sft", settings: { learning_rate: 0.0002 }, lora: { r: 16 } },
  });
});

test("All settings start not set: no default, the library's shown as a placeholder", () => {
  const form = pipelineForm(settingsSchema);
  const sections = flatten(form).filter((node) => node.kind === "section") as FormSection[];
  const settings = sections.flatMap((section) => section.allSettings ?? []);

  expect(settings.filter((field) => "default" in field)).toEqual([]);
  expect(Object.fromEntries(settings.map((field) => [lastName(field), field.placeholder]))).toEqual(
    {
      optim: "adamw_torch",
      packing_strategy: "bfd",
      use_liger_kernel: "false",
      adam_beta1: "0.9",
      lr_scheduler_kwargs: "{}",
      accelerator_config: undefined,
      report_to: undefined,
      use_rslora: "false",
      loftq_config: "{}",
    },
  );
});

test("an object setting is typed as JSON and sent as what it reads", () => {
  const values = {
    fields: {
      "phase.settings.lr_scheduler_kwargs": '{"min_lr": 0.00001}',
      "phase.settings.use_liger_kernel": "true",
      "phase.lora.loftq_config": "{}",
    },
    more: {},
  };
  const form = pipelineForm(settingsSchema, values);
  const phase = pipelineRequestFromForm(form, values).phase as { settings: unknown; lora: unknown };

  expect(phase.settings).toEqual({
    learning_rate: 0.0002,
    lr_scheduler_kwargs: { min_lr: 0.00001 },
    use_liger_kernel: true,
  });
  expect(phase.lora).toEqual({ r: 16, loftq_config: {} });
});

test("every other setting is one click away, typed, and sent only once edited", () => {
  const values = { fields: { "phase.settings.adam_beta1": "0.95" }, more: {} };
  const form = pipelineForm(settingsSchema, values);
  const settings = flatten(form).find((node) => node.name === "phase.settings") as FormSection;

  expect(settings.children.map((node) => node.name)).toEqual(["phase.settings.learning_rate"]);
  expect(
    Object.fromEntries(settings.allSettings!.map((field) => [lastName(field), field.kind])),
  ).toEqual({
    optim: "choice",
    packing_strategy: "text",
    use_liger_kernel: "choice",
    adam_beta1: "number",
    lr_scheduler_kwargs: "json",
    accelerator_config: "json",
    report_to: "choice",
  });
  expect(pipelineRequestFromForm(form, values)).toEqual({
    phase: {
      algorithm: "sft",
      settings: { learning_rate: 0.0002, adam_beta1: 0.95 },
      lora: { r: 16 },
    },
  });
});

test("LoRA's rank is a choice of the ranks vLLM serves, sent as a number", () => {
  const values = { fields: { "phase.lora.r": "32" }, more: {} };
  const form = pipelineForm(settingsSchema, values);
  const lora = flatten(form).find((node) => node.name === "phase.lora") as FormSection;

  expect(lora.children[0]).toMatchObject({ kind: "choice", choices: [8, 16, 32] });
  expect(lora.allSettings!.map(lastName)).toEqual(["use_rslora", "loftq_config"]);
  expect((pipelineRequestFromForm(form, values).phase as { lora: unknown }).lora).toEqual({
    r: 32,
  });
});

function lastName(node: FormNode): string {
  return node.name.split(".").pop()!;
}

test("deleting a list item moves every value of the later items up one place", () => {
  const values = {
    fields: {
      "finetune.phases.0.dataset": "dataset:a",
      "finetune.phases.1.dataset": "dataset:b",
      "finetune.phases.2.dataset": "dataset:c",
      "finetune.phases.10x": "kept",
    },
    more: { "finetune.phases.2.settings": [{ key: "seed", value: "1" }] },
    entries: { "finetune.phases.2.rewards": [{ name: "short", fields: {} }] },
    counts: { "finetune.phases": 3 },
  };

  expect(withoutListItem(values, "finetune.phases", 1, 3)).toEqual({
    fields: {
      "finetune.phases.0.dataset": "dataset:a",
      "finetune.phases.1.dataset": "dataset:c",
      "finetune.phases.10x": "kept",
    },
    more: { "finetune.phases.1.settings": [{ key: "seed", value: "1" }] },
    entries: { "finetune.phases.1.rewards": [{ name: "short", fields: {} }] },
    counts: { "finetune.phases": 2 },
  });
});

test("an optional fixed value, e.g. `params_from: @sweep`, is a choice that is left out unless chosen", () => {
  const optionalSchema = {
    type: "object",
    properties: {
      params_from: {
        anyOf: [{ const: "@sweep", type: "string" }, { type: "null" }],
        default: null,
      },
    },
  };
  const values = { fields: { sweep: "on" }, more: {} };
  const form = pipelineForm(optionalSchema, values);

  expect(form.children[0]).toMatchObject({ kind: "choice", choices: ["@sweep"] });
  expect(form.children[0]).not.toHaveProperty("default");
  expect(pipelineRequestFromForm(form, values)).toEqual({});
});

// A trimmed copy of the published schema's `quantize`, whose Calibration only some schemes take.
const quantizeSchema = {
  type: "object",
  properties: {
    quantize: { anyOf: [{ $ref: "#/$defs/Quantize" }, { type: "null" }], default: null },
  },
  $defs: {
    Quantize: {
      type: "object",
      properties: {
        scheme: { enum: ["fp8-dynamic", "w4a16-gptq"], type: "string", default: "fp8-dynamic" },
        calibration: {
          anyOf: [{ $ref: "#/$defs/Calibration" }, { type: "null" }],
          default: null,
          applies_if: "calibrates",
          required_if_applies: true,
        },
      },
    },
    Calibration: {
      type: "object",
      properties: {
        dataset: {
          type: "string",
          pattern: "^dataset:[A-Za-z0-9][A-Za-z0-9_.-]*(@d+)?$",
          default: "dataset:llm-compression-calibration",
        },
        samples: { type: "integer", default: 512 },
      },
    },
  },
  uncalibrated_schemes: ["fp8-dynamic"],
};

function quantizeRequest(fields: Record<string, string>) {
  const values = { fields: { quantize: "on", ...fields }, more: {} };
  return pipelineRequestFromForm(pipelineForm(quantizeSchema, values), values).quantize;
}

test("a calibrating scheme's Calibration is required and sent with its defaults; fp8-dynamic's is left out", () => {
  const calibrated = { "quantize.scheme": "w4a16-gptq" };
  const form = pipelineForm(quantizeSchema, { fields: calibrated, more: {} });
  const byName = Object.fromEntries(flatten(form).map((node) => [node.name, node]));
  expect(byName["quantize.calibration"]).toMatchObject({
    title: "Calibration (required)",
    optional: false,
  });
  expect(quantizeRequest(calibrated)).toEqual({
    scheme: "w4a16-gptq",
    calibration: { dataset: "dataset:llm-compression-calibration", samples: 512 },
  });

  // Switched on before the scheme changed, it still isn't sent.
  expect(quantizeRequest({ "quantize.calibration": "on" })).toEqual({ scheme: "fp8-dynamic" });
});
