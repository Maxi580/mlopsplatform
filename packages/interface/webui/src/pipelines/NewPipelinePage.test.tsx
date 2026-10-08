import { screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { REWARD_TEMPLATE } from "../config";
import { fakeApi, renderApp } from "../testApi";

const schema = {
  type: "object",
  properties: {
    name: { type: "string", title: "Name" },
    finetune: {
      type: "object",
      title: "Finetune",
      properties: {
        base_model: { type: "string", title: "Base Model", pattern: "^hf:x" },
        settings: {
          type: "object",
          title: "Settings",
          additionalProperties: true,
          properties: { learning_rate: { type: "number", title: "Learning Rate" } },
        },
      },
    },
  },
};

function api(submit: (body: unknown) => [number, unknown]) {
  return fakeApi({
    "GET /schema": [200, schema],
    "GET /datasets": [200, []],
    "GET /settings": [200, { gpu_count: 1 }],
    "GET /pipelines": [200, []],
    "POST /pipelines": submit,
  });
}

beforeEach(() => localStorage.clear());

test("the form submits the Pipeline Request with its Secrets", async () => {
  const calls = api(() => [202, { id: 3 }]);
  renderApp("/pipelines/new");

  await userEvent.type(await screen.findByLabelText(/^Name/), "qwen-sft");
  await userEvent.type(screen.getByLabelText(/^Base Model/), "hf:Qwen/Qwen2.5-0.5B-Instruct");
  await userEvent.type(screen.getByLabelText(/^Learning Rate/), "1e-4");
  await userEvent.click(screen.getByRole("button", { name: /add setting/i }));
  await userEvent.type(screen.getByLabelText("Setting"), "weight_decay");
  await userEvent.type(screen.getByLabelText("Value"), "0.01");
  await userEvent.type(screen.getByLabelText(/^Hugging Face token/), "hf_secret");
  await userEvent.click(screen.getByRole("button", { name: /submit/i }));

  expect(await screen.findByText(/Submitted Pipeline 3/)).toBeInTheDocument();
  expect(calls.find((call) => call.route === "POST /pipelines")!.body).toEqual({
    request: {
      name: "qwen-sft",
      finetune: {
        base_model: "hf:Qwen/Qwen2.5-0.5B-Instruct",
        settings: { learning_rate: 0.0001, weight_decay: 0.01 },
      },
    },
    secrets: { hf_token: "hf_secret" },
  });
});

test("validation errors appear next to their fields", async () => {
  const notASetting = "`warmup_ratio` is not a SFTConfig setting";
  api(() => [
    422,
    {
      detail: [
        { loc: ["finetune", "base_model"], msg: "Field required" },
        { loc: ["finetune", "settings", "warmup_ratio"], msg: notASetting },
        { loc: [], msg: "Kubeflow did not start Pipeline 4" },
      ],
    },
  ]);
  renderApp("/pipelines/new");

  await userEvent.click(await screen.findByRole("button", { name: /submit/i }));

  expect(await screen.findByLabelText(/^Base Model/)).toHaveAccessibleDescription("Field required");
  expect(screen.getByText(`warmup_ratio: ${notASetting}`)).toBeInTheDocument();
  expect(screen.getByText(/Kubeflow did not start Pipeline 4/)).toBeInTheDocument();
  expect(screen.getByRole("heading", { name: "New Pipeline" })).toBeInTheDocument();
});

test("the form shows what the request still downloads as it changes", async () => {
  const GB = 1024 ** 3;
  fakeApi({
    "GET /schema": [200, schema],
    "GET /datasets": [200, []],
    "POST /pipelines/validate": ({ request }) => {
      const cached = request.finetune?.base_model === "hf:x/cached";
      const bytes = cached ? 2.1 * GB : 9.4 * GB;
      return [
        200,
        {
          request,
          downloads: [{ kind: "base_model", ref: request.finetune?.base_model, bytes, cached }],
          download_bytes: cached ? 0 : bytes,
          cached_bytes: cached ? bytes : 0,
        },
      ];
    },
  });
  renderApp("/pipelines/new");

  await userEvent.type(await screen.findByLabelText(/^Base Model/), "hf:x/new");
  expect(await screen.findByText("9.4 GB to download, 0 B already cached")).toBeInTheDocument();

  await userEvent.clear(screen.getByLabelText(/^Base Model/));
  await userEvent.type(screen.getByLabelText(/^Base Model/), "hf:x/cached");
  expect(await screen.findByText("0 B to download, 2.1 GB already cached")).toBeInTheDocument();
});

test("an optional Stage joins the request only once it is switched on", async () => {
  const withSpeculate = {
    ...schema,
    properties: {
      ...schema.properties,
      speculate: { anyOf: [{ $ref: "#/$defs/Speculate" }, { type: "null" }], default: null },
    },
    $defs: {
      Speculate: {
        type: "object",
        properties: { samples: { type: "integer", title: "Samples" } },
      },
    },
  };
  const calls = fakeApi({
    "GET /schema": [200, withSpeculate],
    "GET /datasets": [200, []],
    "GET /pipelines": [200, []],
    "POST /pipelines": [202, { id: 5 }],
  });
  renderApp("/pipelines/new");

  const speculate = await screen.findByRole("button", { name: /add speculate/i });
  expect(screen.queryByLabelText(/^Samples/)).not.toBeInTheDocument();
  await userEvent.click(speculate);
  await userEvent.type(screen.getByLabelText(/^Samples/), "4096");
  await userEvent.click(screen.getByRole("button", { name: /submit/i }));

  await screen.findByText(/Submitted Pipeline 5/);
  const submitted = calls.find((call) => call.route === "POST /pipelines")!.body;
  expect(submitted.request.speculate).toEqual({ samples: 4096 });
});

test("the benchmark picker shows each benchmark's category, description and size", async () => {
  const withEvaluate = {
    ...schema,
    properties: {
      ...schema.properties,
      evaluate: { anyOf: [{ $ref: "#/$defs/Evaluate" }, { type: "null" }], default: null },
    },
    $defs: {
      Evaluate: {
        type: "object",
        properties: {
          benchmarks: {
            type: "array",
            title: "Benchmarks",
            minItems: 1,
            items: { enum: ["lm_eval:gsm8k", "lm_eval:mmlu"], type: "string" },
          },
        },
      },
    },
  };
  const catalog = [
    {
      name: "lm_eval:gsm8k",
      category: "maths",
      description: "Grade-school maths word problems.",
      licence: "MIT",
      size_bytes: 2.6 * 2 ** 20,
    },
    {
      name: "lm_eval:mmlu",
      category: "knowledge",
      description: "Questions on 57 subjects.",
      licence: "MIT",
      size_bytes: 49.1 * 2 ** 20,
    },
  ];
  const calls = fakeApi({
    "GET /schema": [200, withEvaluate],
    "GET /benchmarks": [200, catalog],
    "GET /datasets": [200, []],
    "GET /pipelines": [200, []],
    "POST /pipelines": [202, { id: 6 }],
  });
  renderApp("/pipelines/new");

  await userEvent.click(await screen.findByRole("button", { name: /add evaluate/i }));
  const gsm8k = screen.getByRole("checkbox", { name: /lm_eval:gsm8k/ });
  expect(gsm8k).toHaveAccessibleDescription(/Grade-school maths word problems\./);
  expect(screen.getByText("maths")).toBeInTheDocument();
  expect(screen.getByText("knowledge")).toBeInTheDocument();
  expect(screen.getByText("49.1 MB")).toBeInTheDocument();
  await userEvent.click(screen.getByRole("checkbox", { name: /lm_eval:mmlu/ }));
  await userEvent.click(gsm8k);
  await userEvent.click(screen.getByRole("button", { name: /submit/i }));

  await screen.findByText(/Submitted Pipeline 6/);
  const submitted = calls.find((call) => call.route === "POST /pipelines")!.body;
  expect(submitted.request.evaluate).toEqual({ benchmarks: ["lm_eval:gsm8k", "lm_eval:mmlu"] });
});

test("a setting with a description shows it as an infobox next to the setting", async () => {
  const infobox = "Off: the model learns from every token of a conversation.";
  const settings = {
    type: "object",
    title: "Settings",
    properties: {
      assistant_only_loss: {
        anyOf: [{ type: "boolean" }, { type: "null" }],
        default: null,
        title: "Assistant-only loss (sft)",
        description: infobox,
      },
    },
  };
  const withAssistantOnly = {
    type: "object",
    properties: { finetune: { type: "object", title: "Finetune", properties: { settings } } },
  };
  const calls = fakeApi({
    "GET /schema": [200, withAssistantOnly],
    "GET /datasets": [200, []],
    "GET /pipelines": [200, []],
    "POST /pipelines": [202, { id: 7 }],
  });
  renderApp("/pipelines/new");

  const setting = await screen.findByLabelText(/^Assistant-only loss/);
  expect(setting).toHaveAccessibleDescription(infobox);
  // Shown on hover and focus of its ⓘ, never printed inline.
  expect(screen.getByText(infobox)).toHaveAttribute("role", "tooltip");
  expect(screen.getByRole("button", { name: "About" })).toHaveAccessibleDescription(infobox);
  await userEvent.selectOptions(setting, "true");
  await userEvent.click(screen.getByRole("button", { name: /submit/i }));

  await screen.findByText(/Submitted Pipeline 7/);
  const submitted = calls.find((call) => call.route === "POST /pipelines")!.body;
  expect(submitted.request.finetune.settings).toEqual({ assistant_only_loss: true });
});

test("rewards are named entries with Python source, explained by an infobox", async () => {
  const infobox = "Each reward defines def reward(sample, item).";
  const rewards = {
    anyOf: [
      { type: "object", patternProperties: { "^[w-]{1,64}$": { $ref: "#/$defs/Reward" } } },
      { type: "null" },
    ],
    default: null,
    title: "Rewards",
    description: infobox,
  };
  const withRewards = {
    type: "object",
    properties: { finetune: { type: "object", title: "Finetune", properties: { rewards } } },
    $defs: {
      Reward: {
        type: "object",
        properties: {
          weight: { type: "number", title: "Weight" },
          source: { type: "string", format: "python", title: "Source" },
        },
      },
    },
  };
  const calls = fakeApi({
    "GET /schema": [200, withRewards],
    "GET /datasets": [200, []],
    "GET /pipelines": [200, []],
    "POST /pipelines": [202, { id: 8 }],
  });
  renderApp("/pipelines/new");

  expect(await screen.findByText(infobox)).toBeInTheDocument();
  await userEvent.click(screen.getByRole("button", { name: /add reward/i }));
  await userEvent.type(screen.getByLabelText(/^Reward name/), "correct");
  await userEvent.type(screen.getByLabelText(/^Weight/), "0.8");
  const source = screen.getByLabelText(/^Source/) as HTMLTextAreaElement;
  expect(source.tagName).toBe("TEXTAREA");
  expect(source.rows).toBe(3);
  expect(source.value).toBe(REWARD_TEMPLATE);
  await userEvent.clear(source);
  await userEvent.type(source, "def reward(sample, item):{Enter}    return 1.0");
  await userEvent.click(screen.getByRole("button", { name: /submit/i }));

  await screen.findByText(/Submitted Pipeline 8/);
  const submitted = calls.find((call) => call.route === "POST /pipelines")!.body;
  expect(submitted.request.finetune.rewards).toEqual({
    correct: { weight: 0.8, source: "def reward(sample, item):\n    return 1.0" },
  });
});

test("a Phase starts with its algorithm's defaults, and × puts a changed one back", async () => {
  const withPhase = {
    type: "object",
    properties: { finetune: { $ref: "#/$defs/Finetune" } },
    $defs: {
      Finetune: {
        type: "object",
        properties: { phases: { type: "array", minItems: 1, items: { $ref: "#/$defs/Phase" } } },
      },
      Phase: {
        type: "object",
        properties: {
          algorithm: { enum: ["sft", "dpo"], type: "string", title: "Algorithm" },
          settings: { $ref: "#/$defs/PhaseSettings", trainer_settings: "settings" },
        },
      },
      PhaseSettings: { type: "object", additionalProperties: true, properties: {} },
    },
    algorithms: {
      sft: {
        settings: {
          learning_rate: { type: "number", title: "Learning rate", default: 2e-4 },
          num_train_epochs: { type: "number", title: "Epochs", default: 3 },
        },
      },
      dpo: {
        settings: {
          learning_rate: { type: "number", title: "Learning rate", default: 5e-6 },
          num_train_epochs: { type: "number", title: "Epochs", default: 1 },
        },
      },
    },
  };
  fakeApi({
    "GET /schema": [200, withPhase],
    "GET /datasets": [200, []],
    "GET /pipelines": [200, []],
  });
  renderApp("/pipelines/new");

  const learningRate = await screen.findByLabelText(/^Learning rate/);
  expect(learningRate).toHaveValue("2e-4");
  await userEvent.clear(screen.getByLabelText(/^Epochs/));
  await userEvent.type(screen.getByLabelText(/^Epochs/), "5");
  await userEvent.selectOptions(screen.getByLabelText(/^Algorithm/), "dpo");

  expect(screen.getByLabelText(/^Learning rate/)).toHaveValue("5e-6");
  expect(screen.getByLabelText(/^Epochs/)).toHaveValue("5");
  await userEvent.click(screen.getByRole("button", { name: "Back to 1" }));
  expect(screen.getByLabelText(/^Epochs/)).toHaveValue("1");
  expect(screen.queryByRole("button", { name: "Back to 1" })).not.toBeInTheDocument();
});

test("the Teacher API key and URL show only for a Teacher at an external API", async () => {
  const withDistill = {
    type: "object",
    properties: {
      distill: { anyOf: [{ $ref: "#/$defs/Distill" }, { type: "null" }], default: null },
    },
    $defs: {
      Distill: {
        type: "object",
        properties: {
          teacher: { type: "string", title: "Teacher", references: ["hf:", "model:"] },
          api_url: {
            anyOf: [{ type: "string" }, { type: "null" }],
            default: null,
            title: "API URL",
            applies_if: "external_teacher",
          },
        },
      },
    },
  };
  fakeApi({
    "GET /schema": [200, withDistill],
    "GET /datasets": [200, []],
    "GET /pipelines": [200, []],
  });
  renderApp("/pipelines/new");

  await userEvent.click(await screen.findByRole("button", { name: /add distill/i }));
  await userEvent.type(screen.getByLabelText(/^Teacher/), "hf:Qwen/Qwen3-8B");
  expect(screen.queryByLabelText(/^API URL/)).not.toBeInTheDocument();
  expect(screen.queryByLabelText(/^Teacher API key/)).not.toBeInTheDocument();

  await userEvent.clear(screen.getByLabelText(/^Teacher/));
  await userEvent.type(screen.getByLabelText(/^Teacher/), "gpt-4o");
  expect(screen.getByLabelText(/^API URL/)).toBeInTheDocument();
  expect(screen.getByLabelText(/^Teacher API key/)).toBeInTheDocument();
});

test("All settings filters by name, and a custom key still reaches the request", async () => {
  const withSettings = {
    type: "object",
    properties: { finetune: { $ref: "#/$defs/Finetune" } },
    $defs: {
      Finetune: {
        type: "object",
        properties: { settings: { $ref: "#/$defs/Settings", trainer_settings: "settings" } },
      },
      Settings: { type: "object", additionalProperties: true, properties: {} },
    },
    algorithms: {
      sft: {
        settings: { learning_rate: { type: "number", title: "Learning rate", default: 2e-4 } },
        more_settings: {
          adam_beta1: { type: "number", title: "Adam beta1", placeholder: 0.9 },
          optim: {
            enum: ["adamw_torch", "adafactor"],
            type: "string",
            title: "Optim",
            placeholder: "adamw_torch",
          },
        },
      },
    },
  };
  const calls = fakeApi({
    "GET /schema": [200, withSettings],
    "GET /datasets": [200, []],
    "GET /pipelines": [200, []],
    "POST /pipelines": [202, { id: 9 }],
  });
  renderApp("/pipelines/new");

  await userEvent.type(await screen.findByLabelText(/^Filter settings/), "adam");
  expect(screen.getByLabelText(/^Adam beta1/)).toHaveAttribute(
    "placeholder",
    "Not set (default: 0.9)",
  );
  expect(screen.queryByLabelText(/^Optim/)).not.toBeInTheDocument();
  await userEvent.type(screen.getByLabelText(/^Adam beta1/), "0.95");
  await userEvent.click(screen.getByRole("button", { name: /add setting/i }));
  await userEvent.type(screen.getByLabelText("Setting"), "my_flag");
  await userEvent.type(screen.getByLabelText("Value"), "true");
  await userEvent.click(screen.getByRole("button", { name: /submit/i }));

  await screen.findByText(/Submitted Pipeline 9/);
  const submitted = calls.find((call) => call.route === "POST /pipelines")!.body;
  expect(submitted.request.finetune.settings).toEqual({
    learning_rate: 0.0002,
    adam_beta1: 0.95,
    my_flag: true,
  });
});

test("All settings send nothing until set; a yes/no starts on Not set", async () => {
  const withSettings = {
    type: "object",
    properties: { finetune: { $ref: "#/$defs/Finetune" } },
    $defs: {
      Finetune: {
        type: "object",
        properties: {
          settings: { $ref: "#/$defs/Settings", trainer_settings: "settings" },
          lora: { $ref: "#/$defs/Settings", trainer_settings: "lora_settings" },
        },
      },
      Settings: { type: "object", additionalProperties: true, properties: {} },
    },
    algorithms: {
      sft: {
        settings: { learning_rate: { type: "number", title: "Learning rate", default: 2e-4 } },
        more_settings: {
          use_liger_kernel: { type: "boolean", title: "Use liger kernel", placeholder: false },
          seed: { type: "integer", title: "Seed", placeholder: 42 },
        },
        lora_settings: { r: { type: "integer", title: "R", default: 16 } },
      },
    },
    more_lora_settings: {
      use_rslora: { type: "boolean", title: "Use rslora", placeholder: false },
      lora_dropout: { type: "number", title: "Lora dropout", placeholder: 0 },
    },
  };
  const calls = fakeApi({
    "GET /schema": [200, withSettings],
    "GET /datasets": [200, []],
    "GET /pipelines": [200, []],
    "POST /pipelines": [202, { id: 9 }],
  });
  renderApp("/pipelines/new");

  const liger = await screen.findByLabelText(/^Use liger kernel/);
  expect(liger).toHaveValue("");
  expect(liger).toHaveDisplayValue("Not set (default: false)");
  expect(screen.getByLabelText(/^Use rslora/)).toHaveDisplayValue("Not set (default: false)");
  expect(screen.getByLabelText(/^Seed/)).toHaveAttribute("placeholder", "Not set (default: 42)");
  expect(screen.getByLabelText(/^Lora dropout/)).toHaveAttribute(
    "placeholder",
    "Not set (default: 0)",
  );
  await userEvent.selectOptions(screen.getByLabelText(/^Use rslora/), "true");
  await userEvent.selectOptions(liger, "true");
  await userEvent.selectOptions(liger, "");
  expect(screen.getByLabelText(/^Learning rate/)).not.toHaveAttribute("placeholder");
  await userEvent.type(screen.getByLabelText(/^Seed/), "7");
  await userEvent.clear(screen.getByLabelText(/^Seed/));
  await userEvent.click(screen.getByRole("button", { name: /submit/i }));

  await screen.findByText(/Submitted Pipeline 9/);
  const submitted = calls.find((call) => call.route === "POST /pipelines")!.body;
  expect(submitted.request.finetune).toEqual({
    settings: { learning_rate: 0.0002 },
    lora: { r: 16, use_rslora: true },
  });
});

test("number fields step in their smallest place shown, from the default when empty", async () => {
  const withNumbers = {
    type: "object",
    properties: {
      finetune: {
        type: "object",
        title: "Finetune",
        properties: {
          learning_rate: { type: "number", title: "Learning rate", default: 2e-4 },
          trials: { type: "integer", title: "Trials", default: 1, exclusiveMinimum: 0 },
        },
      },
    },
  };
  fakeApi({
    "GET /schema": [200, withNumbers],
    "GET /datasets": [200, []],
    "GET /pipelines": [200, []],
  });
  renderApp("/pipelines/new");

  const learningRate = await screen.findByLabelText(/^Learning rate/);
  await userEvent.click(screen.getByRole("button", { name: "Increase Learning rate" }));
  expect(learningRate).toHaveValue("3e-4");
  await userEvent.clear(learningRate);
  await userEvent.type(learningRate, "{ArrowDown}");
  expect(learningRate).toHaveValue("1e-4");
  await userEvent.click(screen.getByRole("button", { name: "Decrease Trials" }));
  expect(screen.getByLabelText(/^Trials/)).toHaveValue("1");
});

const phasesSchema = {
  type: "object",
  properties: { finetune: { $ref: "#/$defs/Finetune" } },
  $defs: {
    Finetune: {
      type: "object",
      properties: {
        base_model: { type: "string", title: "Base Model" },
        phases: { type: "array", minItems: 1, items: { $ref: "#/$defs/Phase" } },
      },
    },
    Phase: {
      type: "object",
      properties: {
        algorithm: { enum: ["sft", "dpo", "kto"], type: "string", title: "Algorithm" },
        dataset: { type: "string", title: "Dataset" },
      },
    },
  },
};

function phaseApi(schema: object = phasesSchema) {
  return fakeApi({
    "GET /schema": [200, schema],
    "GET /datasets": [200, []],
    "GET /pipelines": [200, []],
    "POST /pipelines": [202, { id: 4 }],
  });
}

function legends(): string[] {
  return [...document.querySelectorAll("legend")].map((legend) => legend.textContent ?? "");
}

test("Add Phase appends Phases, submitted in order, each saying what it continues from", async () => {
  const calls = phaseApi();
  renderApp("/pipelines/new");

  await userEvent.type(await screen.findByLabelText(/^Base Model/), "hf:Qwen/Qwen3-0.6B");
  await userEvent.click(screen.getByRole("button", { name: /add phase/i }));
  await userEvent.click(screen.getByRole("button", { name: /add phase/i }));
  const algorithms = screen.getAllByLabelText(/^Algorithm/);
  await userEvent.selectOptions(algorithms[1], "dpo");
  await userEvent.selectOptions(algorithms[2], "kto");

  expect(legends()).toEqual([
    expect.stringContaining("Phase 1 · SFT · starts from hf:Qwen/Qwen3-0.6B"),
    expect.stringContaining("Phase 2 · DPO · continues from Phase 1 (SFT)"),
    expect.stringContaining("Phase 3 · KTO · continues from Phase 2 (DPO)"),
  ]);
  await userEvent.click(screen.getByRole("button", { name: /submit/i }));
  await screen.findByText(/Submitted Pipeline 4/);
  const submitted = calls.find((call) => call.route === "POST /pipelines")!.body;
  expect(submitted.request.finetune.phases).toEqual([
    { algorithm: "sft" },
    { algorithm: "dpo" },
    { algorithm: "kto" },
  ]);
});

test("deleting the middle Phase keeps the others' values and renumbers them", async () => {
  phaseApi();
  renderApp("/pipelines/new");

  await userEvent.click(await screen.findByRole("button", { name: /add phase/i }));
  await userEvent.click(screen.getByRole("button", { name: /add phase/i }));
  const datasets = screen.getAllByLabelText(/^Dataset/);
  await userEvent.type(datasets[0], "dataset:a");
  await userEvent.type(datasets[1], "dataset:b");
  await userEvent.type(datasets[2], "dataset:c");
  await userEvent.click(screen.getByRole("button", { name: "Delete Phase 2" }));

  expect(
    screen.getAllByLabelText(/^Dataset/).map((input) => (input as HTMLInputElement).value),
  ).toEqual(["dataset:a", "dataset:c"]);
  expect(legends()[1]).toContain("Phase 2 · SFT · continues from Phase 1 (SFT)");
});

test("deleting the last Phase switches Finetune off, and its plus brings a Phase back", async () => {
  const optionalFinetune = { anyOf: [{ $ref: "#/$defs/Finetune" }, { type: "null" }] };
  phaseApi({ ...phasesSchema, properties: { finetune: optionalFinetune } });
  renderApp("/pipelines/new");

  await userEvent.click(await screen.findByRole("button", { name: "Add Finetune" }));
  // Its Phases have trashcans, so the Stage has none of its own.
  expect(screen.queryByRole("button", { name: "Remove Finetune" })).not.toBeInTheDocument();
  await userEvent.click(screen.getByRole("button", { name: "Delete Phase 1" }));

  expect(screen.queryByLabelText(/^Base Model/)).not.toBeInTheDocument();
  await userEvent.click(screen.getByRole("button", { name: "Add Finetune" }));
  expect(legends()).toEqual([expect.stringContaining("Phase 1")]);
});

test("an optional section switches on with its plus and off with its trashcan", async () => {
  const optionalSettings = { anyOf: [{ $ref: "#/$defs/Settings" }, { type: "null" }] };
  fakeApi({
    "GET /schema": [
      200,
      {
        type: "object",
        properties: { finetune: { $ref: "#/$defs/Finetune" } },
        $defs: {
          Finetune: { type: "object", properties: { lora: optionalSettings } },
          Settings: { type: "object", properties: { rank: { type: "integer", title: "Rank" } } },
        },
      },
    ],
    "GET /datasets": [200, []],
    "GET /pipelines": [200, []],
  });
  renderApp("/pipelines/new");

  await userEvent.click(await screen.findByRole("button", { name: "Add Lora" }));
  expect(screen.getByLabelText(/^Rank/)).toBeInTheDocument();
  await userEvent.click(screen.getByRole("button", { name: "Remove Lora" }));
  expect(screen.queryByLabelText(/^Rank/)).not.toBeInTheDocument();
  expect(screen.getByRole("button", { name: "Add Lora" })).toBeInTheDocument();
});

const HF = "^hf:[w.-]+/[w.-]+(@[w.-]+)?$";
const MODEL = "^model:[a-z0-9][a-z0-9.-]*(@d+)?$";
const modelsSchema = {
  type: "object",
  properties: {
    finetune: { anyOf: [{ $ref: "#/$defs/Finetune" }, { type: "null" }], default: null },
    quantize: { anyOf: [{ $ref: "#/$defs/Quantize" }, { type: "null" }], default: null },
  },
  $defs: {
    Finetune: {
      type: "object",
      properties: {
        base_model: {
          anyOf: [{ type: "string", pattern: HF }, { type: "null" }],
          default: null,
          title: "Base Model",
        },
      },
    },
    Quantize: {
      type: "object",
      properties: {
        model: {
          anyOf: [
            { type: "string", pattern: HF },
            { type: "string", pattern: MODEL },
            { const: "@finetune", type: "string" },
            { type: "null" },
          ],
          default: null,
          title: "Model",
        },
      },
    },
  },
};

function modelsApi() {
  return fakeApi({
    "GET /schema": [200, modelsSchema],
    "GET /datasets": [200, []],
    "GET /pipelines": [200, []],
    "GET /models": [
      200,
      [{ name: "qwen-sft", versions: [{ version: 2, tags: { weights: "full" } }] }],
    ],
    "GET /base-models?search=": [
      200,
      [{ name: "Qwen/Qwen3-8B", reference: "hf:Qwen/Qwen3-8B", parameters: 8.19e9, gated: false }],
    ],
    "GET /base-models?search=qwen%207b": [
      200,
      [
        {
          name: "Qwen/Qwen2.5-7B-Instruct",
          reference: "hf:Qwen/Qwen2.5-7B-Instruct",
          parameters: 7.6e9,
          gated: false,
        },
        {
          name: "Qwen/Qwen-7B-Gated",
          reference: "hf:Qwen/Qwen-7B-Gated",
          parameters: 7.7e9,
          gated: true,
        },
      ],
    ],
  });
}

function groupsOf(listbox: HTMLElement): string[] {
  return [...listbox.querySelectorAll("[role=group]")].map((group) => group.ariaLabel ?? "");
}

test("a Base Model field lists curated models, then the Hub's for what is typed", async () => {
  modelsApi();
  renderApp("/pipelines/new");
  await userEvent.click(await screen.findByRole("button", { name: /add finetune/i }));

  await userEvent.click(screen.getByRole("combobox", { name: /^Base Model/ }));
  const listbox = await screen.findByRole("listbox");
  expect(groupsOf(listbox)).toEqual(["Open source"]);
  expect(await within(listbox).findByText("Qwen/Qwen3-8B")).toBeInTheDocument();

  await userEvent.type(screen.getByRole("combobox", { name: /^Base Model/ }), "qwen 7b");
  const gated = await screen.findByRole("option", { name: /Qwen-7B-Gated/ });
  expect(gated).toHaveTextContent("7.7B");
  expect(within(gated).getByRole("img", { name: "gated" })).toBeInTheDocument();
  expect(screen.getByRole("option", { name: /Qwen2.5-7B-Instruct/ })).toHaveTextContent("7.6B");
});

test("each model field offers its groups, and Quantize preselects Finetune's output", async () => {
  const calls = modelsApi();
  renderApp("/pipelines/new");
  await userEvent.click(await screen.findByRole("button", { name: /add finetune/i }));
  await userEvent.click(screen.getByRole("button", { name: /add quantize/i }));

  const model = screen.getByRole("combobox", { name: /^Model/ });
  expect(model).toHaveValue("@finetune");
  await userEvent.click(screen.getByRole("button", { name: "Clear Model" }));
  expect(model).toHaveValue("");
  await userEvent.click(model);
  expect(groupsOf(await screen.findByRole("listbox"))).toEqual([
    "This Pipeline",
    "Our models",
    "Open source",
  ]);
  expect(screen.getByRole("option", { name: /Output of Finetune/ })).toBeInTheDocument();
  expect(screen.getByRole("option", { name: /qwen-sft@2/ })).toHaveTextContent("full weights");

  // The keyboard alone picks: down to Our models' first entry, then Enter.
  await userEvent.keyboard("{ArrowDown}{Enter}");
  expect(model).toHaveValue("model:qwen-sft@2");
  // The browser asks the API for Hub models, never huggingface.co itself.
  expect(calls.map((call) => call.route)).toContain("GET /base-models?search=");
});

const DATASET = "^dataset:[A-Za-z0-9][A-Za-z0-9_.-]*(@d+)?$";
const datasetsSchema = {
  type: "object",
  properties: {
    distill: { anyOf: [{ $ref: "#/$defs/Distill" }, { type: "null" }], default: null },
    finetune: { anyOf: [{ $ref: "#/$defs/Finetune" }, { type: "null" }], default: null },
  },
  $defs: {
    Distill: {
      type: "object",
      properties: {
        dataset: {
          type: "string",
          pattern: DATASET,
          title: "Prompts",
          row_formats: ["prompt_only"],
        },
      },
    },
    Finetune: {
      type: "object",
      properties: { phases: { type: "array", minItems: 1, items: { $ref: "#/$defs/Phase" } } },
    },
    Phase: {
      type: "object",
      properties: {
        algorithm: { enum: ["sft", "dpo"], type: "string", title: "Algorithm" },
        dataset: {
          anyOf: [
            { type: "string", pattern: DATASET },
            { const: "@distill", type: "string" },
          ],
          title: "Dataset",
        },
      },
    },
  },
  algorithms: {
    sft: { row_formats: ["messages", "prompt_completion", "text"] },
    dpo: { row_formats: ["preference"] },
  },
  row_formats: {
    messages: { example: '{"messages": []}' },
    prompt_completion: { example: '{"prompt": "Hi", "completion": "Hello"}' },
    text: { example: '{"text": "Hi"}' },
    preference: { example: '{"chosen": "a", "rejected": "b"}' },
    prompt_only: { example: '{"prompt": "Hi"}' },
  },
};

function datasetsApi(routes: Parameters<typeof fakeApi>[0] = {}) {
  let datasets = [
    { name: "chat", versions: [{ version: 1, size_bytes: 9, row_format: "messages" }] },
  ];
  return fakeApi({
    "GET /schema": [200, datasetsSchema],
    "GET /datasets": () => [200, datasets],
    "GET /pipelines": [200, []],
    "POST /datasets/support/versions?row_formats=messages,prompt_completion,text": () => {
      datasets = [
        ...datasets,
        { name: "support", versions: [{ version: 1, size_bytes: 9, row_format: "text" }] },
      ];
      return [201, { version: 1 }];
    },
    "POST /datasets/pairs/versions?row_formats=messages,prompt_completion,text": [
      422,
      {
        detail:
          "Rejected pairs: has preference rows; it takes messages, prompt_completion, text rows",
      },
    ],
    ...routes,
  });
}

async function uploadIn(field: RegExp, fileName: string) {
  await userEvent.click(screen.getByRole("combobox", { name: field }));
  await userEvent.click(await screen.findByRole("button", { name: "Upload…" }));
  const dialog = screen.getByRole("dialog");
  const file = new File(['{"text": "Hi"}\n'], fileName, { type: "application/jsonl" });
  await userEvent.upload(within(dialog).getByLabelText("JSONL file"), file);
  return dialog;
}

test("uploading a JSONL file in a Dataset field selects it", async () => {
  const calls = datasetsApi();
  renderApp("/pipelines/new");
  await userEvent.click(await screen.findByRole("button", { name: /add finetune/i }));

  const dialog = await uploadIn(/^Dataset/, "support.jsonl");
  expect(within(dialog).getByLabelText("Name")).toHaveValue("support");
  await userEvent.click(within(dialog).getByRole("button", { name: /upload/i }));

  expect(await screen.findByRole("combobox", { name: /^Dataset/ })).toHaveValue("dataset:support");
  expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  const upload = calls.find((call) => call.route.startsWith("POST /datasets/support"))!;
  expect(upload.body).toBeInstanceOf(File);
  // The dialog's submit isn't the Pipeline's.
  expect(calls.map((call) => call.route)).not.toContain("POST /pipelines");
});

test("a wrong row format shows the API's error in the dialog", async () => {
  datasetsApi();
  renderApp("/pipelines/new");
  await userEvent.click(await screen.findByRole("button", { name: /add finetune/i }));

  const dialog = await uploadIn(/^Dataset/, "pairs.jsonl");
  await userEvent.click(within(dialog).getByRole("button", { name: /upload/i }));

  expect(await within(dialog).findByRole("alert")).toHaveTextContent("has preference rows");
});

test("uploading under an existing name says it replaces it, keeping the old data", async () => {
  datasetsApi();
  renderApp("/pipelines/new");
  await userEvent.click(await screen.findByRole("button", { name: /add finetune/i }));

  const dialog = await uploadIn(/^Dataset/, "chat.jsonl");

  expect(dialog).toHaveTextContent(
    "Replaces chat (the old data stays with Pipelines that used it)",
  );
});

test("the format ⓘ shows the row formats the algorithm reads, with an example each", async () => {
  datasetsApi();
  renderApp("/pipelines/new");
  await userEvent.click(await screen.findByRole("button", { name: /add finetune/i }));

  const formats = () => screen.getByRole("button", { name: "Row formats" });
  for (const example of [
    '{"messages": []}',
    '{"prompt": "Hi", "completion": "Hello"}',
    '{"text": "Hi"}',
  ])
    expect(formats()).toHaveAccessibleDescription(expect.stringContaining(example));
  await userEvent.selectOptions(screen.getByLabelText(/^Algorithm/), "dpo");
  expect(formats()).toHaveAccessibleDescription(
    expect.stringContaining('{"chosen": "a", "rejected": "b"}'),
  );
  expect(formats()).not.toHaveAccessibleDescription(expect.stringContaining("messages"));
});

test("with Distill on, Finetune's Dataset preselects the Distillation Dataset", async () => {
  datasetsApi();
  renderApp("/pipelines/new");
  await userEvent.click(await screen.findByRole("button", { name: /add finetune/i }));
  expect(screen.getByRole("combobox", { name: /^Dataset/ })).toHaveValue("");

  await userEvent.click(screen.getByRole("button", { name: /add distill/i }));
  const dataset = screen.getByRole("combobox", { name: /^Dataset/ });
  expect(dataset).toHaveValue("@distill");
  await userEvent.click(dataset);
  expect(
    screen.getByRole("option", { name: /Distillation Dataset from this Pipeline/ }),
  ).toBeInTheDocument();
  // Only the Datasets the algorithm reads; Versions stay out of sight.
  expect(screen.getByRole("option", { name: /^chat/ })).toHaveTextContent("messages");
});

test("Quantize shows Calibration, required, only for a scheme that calibrates", async () => {
  const quantizeSchema = {
    type: "object",
    properties: {
      quantize: { anyOf: [{ $ref: "#/$defs/Quantize" }, { type: "null" }], default: null },
    },
    $defs: {
      Quantize: {
        type: "object",
        properties: {
          scheme: {
            enum: ["fp8-dynamic", "w4a16-gptq"],
            type: "string",
            default: "fp8-dynamic",
            title: "Scheme",
          },
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
        properties: { samples: { type: "integer", default: 512, title: "Samples" } },
      },
    },
    uncalibrated_schemes: ["fp8-dynamic"],
  };
  const calls = fakeApi({
    "GET /schema": [200, quantizeSchema],
    "GET /datasets": [200, []],
    "GET /pipelines": [200, []],
    "POST /pipelines": [202, { id: 10 }],
  });
  renderApp("/pipelines/new");
  await userEvent.click(await screen.findByRole("button", { name: /add quantize/i }));
  expect(screen.queryByText(/^Calibration/)).not.toBeInTheDocument();

  await userEvent.selectOptions(screen.getByLabelText(/^Scheme/), "w4a16-gptq");
  expect(screen.getByText("Calibration (required)")).toBeInTheDocument();
  expect(screen.queryByRole("button", { name: /remove calibration/i })).not.toBeInTheDocument();
  expect(screen.getByLabelText(/^Samples/)).toHaveValue("512");

  await userEvent.selectOptions(screen.getByLabelText(/^Scheme/), "fp8-dynamic");
  expect(screen.queryByText(/^Calibration/)).not.toBeInTheDocument();
  await userEvent.selectOptions(screen.getByLabelText(/^Scheme/), "w4a16-gptq");
  await userEvent.click(screen.getByRole("button", { name: /submit/i }));

  await screen.findByText(/Submitted Pipeline 10/);
  const submitted = calls.find((call) => call.route === "POST /pipelines")!.body;
  expect(submitted.request.quantize).toEqual({
    scheme: "w4a16-gptq",
    calibration: { samples: 512 },
  });
});

test("a Sweep tunes a setting picked by name, from a range around its default", async () => {
  const range = { anyOf: [{ type: "integer" }, { type: "number" }, { type: "null" }] };
  const withSweep = {
    type: "object",
    properties: { sweep: { $ref: "#/$defs/Sweep" } },
    $defs: {
      Sweep: {
        type: "object",
        properties: {
          algorithm: { enum: ["sft"], type: "string" },
          settings: { $ref: "#/$defs/PhaseSettings", trainer_settings: "settings" },
          parameters: { $ref: "#/$defs/SweepParameters" },
        },
      },
      PhaseSettings: { type: "object", additionalProperties: true, properties: {} },
      SweepParameters: {
        type: "object",
        properties: {
          settings: { type: "object", additionalProperties: { $ref: "#/$defs/SweepParameter" } },
        },
      },
      SweepParameter: {
        type: "object",
        properties: { min: { ...range, title: "Min" }, max: { ...range, title: "Max" } },
      },
    },
    algorithms: {
      sft: {
        settings: {
          learning_rate: { type: "number", title: "Learning rate", default: 2e-4, minimum: 0 },
          num_train_epochs: { type: "number", title: "Epochs", default: 3 },
        },
      },
    },
  };
  const calls = fakeApi({
    "GET /schema": [200, withSweep],
    "GET /datasets": [200, []],
    "GET /pipelines": [200, []],
    "POST /pipelines": [202, { id: 9 }],
  });
  renderApp("/pipelines/new");
  const parameters = await screen.findByRole("group", { name: /^Parameters/ });
  const addParameter = within(parameters).getByRole("button", { name: /add setting/i });

  await userEvent.click(addParameter);
  await userEvent.type(screen.getByRole("combobox", { name: /^Setting name/ }), "le");
  await userEvent.click(screen.getByRole("option", { name: /learning_rate/ }));
  expect(screen.getByLabelText(/^Min/)).toHaveValue("1e-4");
  expect(screen.getByLabelText(/^Max/)).toHaveValue("3e-4");
  // Min steps up, but never to Max.
  await userEvent.click(screen.getByRole("button", { name: "Increase Min" }));
  await userEvent.click(screen.getByRole("button", { name: "Increase Min" }));
  expect(screen.getByLabelText(/^Min/)).toHaveValue("2e-4");

  const fixed = screen.getByLabelText(/^Learning rate/);
  expect(fixed).toBeDisabled();
  expect(fixed).toHaveValue("");
  expect(fixed).toHaveAttribute("placeholder", "Tuned by this Sweep");
  await userEvent.click(addParameter);
  await userEvent.click(screen.getAllByRole("combobox", { name: /^Setting name/ })[1]);
  const tuned = screen.getByRole("option", { name: /learning_rate/ });
  expect(tuned).toHaveAttribute("aria-disabled", "true");
  await userEvent.click(tuned);
  expect(screen.getAllByRole("combobox", { name: /^Setting name/ })[1]).toHaveValue("");

  // Removed, the setting is the Sweep's to fix again, or to tune in another parameter.
  await userEvent.click(screen.getAllByRole("button", { name: /remove setting/i })[0]);
  expect(screen.getByLabelText(/^Learning rate/)).toBeEnabled();
  expect(screen.getByLabelText(/^Learning rate/)).toHaveValue("2e-4");
  await userEvent.click(screen.getByRole("combobox", { name: /^Setting name/ }));
  await userEvent.click(screen.getByRole("option", { name: /learning_rate/ }));
  expect(screen.getByLabelText(/^Learning rate/)).toBeDisabled();

  await userEvent.click(screen.getByRole("button", { name: /submit/i }));
  await screen.findByText(/Submitted Pipeline 9/);
  const submitted = calls.find((call) => call.route === "POST /pipelines")!.body;
  expect(submitted.request.sweep.settings).toEqual({ num_train_epochs: 3 });
  expect(submitted.request.sweep.parameters.settings).toEqual({
    learning_rate: { min: 1e-4, max: 3e-4 },
  });
});
