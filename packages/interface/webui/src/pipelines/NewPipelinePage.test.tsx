import { screen } from "@testing-library/react";
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

  const speculate = await screen.findByRole("checkbox", { name: /run speculate/i });
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

  await userEvent.click(await screen.findByRole("checkbox", { name: /run evaluate/i }));
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

  await userEvent.click(await screen.findByRole("checkbox", { name: /run distill/i }));
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
  expect(screen.getByLabelText(/^Adam beta1/)).toHaveAttribute("placeholder", "0.9");
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

function phaseApi() {
  return fakeApi({
    "GET /schema": [200, phasesSchema],
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

test("the last Phase can't be deleted", async () => {
  phaseApi();
  renderApp("/pipelines/new");

  expect(await screen.findByRole("button", { name: "Delete Phase 1" })).toBeDisabled();
});
