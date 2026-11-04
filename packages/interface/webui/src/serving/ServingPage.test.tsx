import { act, fireEvent, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { ENDPOINT_LIST_REFRESH_MS } from "../config";
import { fakeApi, renderApp } from "../testApi";
import type { Endpoint, EndpointStatsSummary } from "./endpoint";

const schema = {
  type: "object",
  properties: { evaluate: { $ref: "#/$defs/Evaluate" } },
  $defs: {
    Evaluate: {
      type: "object",
      properties: {
        serving: { anyOf: [{ $ref: "#/$defs/ServingOptions" }, { type: "null" }] },
      },
    },
    ServingOptions: {
      type: "object",
      properties: {
        max_model_len: {
          anyOf: [{ type: "integer", exclusiveMinimum: 0 }, { type: "null" }],
          default: null,
          title: "Max Model Len",
          description: "vLLM reads it from the model when left empty.",
        },
        prefix_caching: { type: "boolean", default: true, title: "Prefix Caching" },
        gpu_memory_utilization: {
          anyOf: [{ type: "number", exclusiveMinimum: 0, maximum: 1 }, { type: "null" }],
          default: 0.9,
          title: "Gpu Memory Utilization",
          description: "The share of GPU memory vLLM may take.",
        },
      },
    },
    Speculative: {
      type: "object",
      properties: {
        num_speculative_tokens: {
          type: "integer",
          exclusiveMinimum: 0,
          default: 3,
          title: "Draft tokens",
          description: "Tokens drafted per step.",
        },
        prompt_lookup_min: {
          anyOf: [{ type: "integer", exclusiveMinimum: 0 }, { type: "null" }],
          default: null,
          title: "Prompt Lookup Min",
        },
        prompt_lookup_max: {
          anyOf: [{ type: "integer", exclusiveMinimum: 0 }, { type: "null" }],
          default: null,
          title: "Prompt Lookup Max",
        },
      },
    },
  },
};
const DEFAULTS = { prefix_caching: true, gpu_memory_utilization: 0.9 };
const ENDPOINT_PATH = "/serving/0123456789abcdef0123456789abcdef/v1";
// Where the page shows and copies it: the address the page was reached at, not the domain.
const URL_HERE = `${window.location.origin}${ENDPOINT_PATH}`;
const chat: Endpoint = {
  name: "chat",
  owner: "shared",
  model: "model:qwen-sft@2",
  status: "running",
  gpu_share: 0.85,
  url: `https://platform.test${ENDPOINT_PATH}`,
  key: "key-of-chat",
  created_at: "2026-10-02T08:30:00+00:00",
  stats: null,
  judge: false,
  judging: { judged: false, every: 10, max_per_hour: 60 },
};
const stats: EndpointStatsSummary = {
  running: 3,
  waiting: 1,
  generation_tokens: 1000,
  read_at: "2026-10-02T09:00:00+00:00",
  time_to_first_token_p50: 0.123,
};
const routes: Parameters<typeof fakeApi>[0] = {
  "GET /schema": [200, schema],
  "GET /endpoints": [200, [chat, { ...chat, name: "old", status: "stopped" }]],
  "GET /models": [
    200,
    [
      { name: "qwen-sft", versions: [{ version: 2, tags: { weights: "full" } }] },
      {
        name: "qwen-sft-speculator",
        versions: [
          { version: 1, tags: { speculator: "eagle3", verifier: "model:qwen-sft@2" } },
          { version: 2, tags: { speculator: "dflash", verifier: "hf:Qwen/Qwen3@abc" } },
        ],
      },
    ],
  ],
  "GET /base-models?search=": [
    200,
    [{ name: "Qwen/Qwen3-8B", reference: "hf:Qwen/Qwen3-8B", parameters: 8.2e9, gated: false }],
  ],
  "GET /settings": [200, { gpu_count: 1 }],
  "GET /judge": [200, { enabled: false, spec: null, system_prompt: "Rate it.", status: null }],
};

test("Endpoints are listed with model, status and URL", async () => {
  fakeApi(routes);
  renderApp("/serving");

  const row = (await screen.findByText("chat")).closest("tr")!;
  for (const text of ["model:qwen-sft@2", "running", URL_HERE]) {
    expect(within(row).getByText(text)).toBeInTheDocument();
  }
  const stopped = screen.getByText("old").closest("tr")!;
  expect(within(stopped).queryByRole("button", { name: /stop/i })).not.toBeInTheDocument();
  expect(within(stopped).queryByRole("button", { name: "Copy URL" })).not.toBeInTheDocument();
  // Its stats page keeps its history.
  expect(within(stopped).getByRole("link", { name: "old" })).toHaveAttribute(
    "href",
    "/serving/old",
  );
});

test("an Endpoint's URL is copied from its row", async () => {
  const user = userEvent.setup();
  fakeApi(routes);
  renderApp("/serving");

  const row = (await screen.findByText("chat")).closest("tr")!;
  await user.click(within(row).getByRole("button", { name: "Copy URL" }));

  expect(await navigator.clipboard.readText()).toBe(URL_HERE);
});

test("an Endpoint's model is copied from its row, also once stopped", async () => {
  const user = userEvent.setup();
  fakeApi(routes);
  renderApp("/serving");

  const stopped = (await screen.findByText("old")).closest("tr")!;
  await user.click(within(stopped).getByRole("button", { name: "Copy model" }));

  expect(await navigator.clipboard.readText()).toBe("model:qwen-sft@2");
});

test("a long name, model and URL are cut off, and hovering shows them in full", async () => {
  fakeApi(routes);
  renderApp("/serving");

  const row = (await screen.findByText("chat")).closest("tr")!;
  for (const value of ["model:qwen-sft@2", URL_HERE]) {
    expect(within(row).getByText(value)).toHaveAttribute("title", value);
    expect(within(row).getByText(value).closest("td")).toHaveClass("cut-off");
  }
  expect(within(row).getByText("chat").closest(".cut-off-name")).toBeInTheDocument();
  expect(within(row).getByText("chat").closest("td")).toHaveAttribute("title", "chat");
});

test("an Endpoint's key is copied and refreshed from its row", async () => {
  const user = userEvent.setup();
  const calls = fakeApi({
    ...routes,
    "POST /endpoints/chat/refresh-key": [200, { ...chat, key: "new-key" }],
  });
  renderApp("/serving");

  const row = (await screen.findByText("chat")).closest("tr")!;
  expect(within(row).queryByText("key-of-chat")).not.toBeInTheDocument();
  await user.click(within(row).getByRole("button", { name: "Copy Endpoint Key" }));
  expect(await navigator.clipboard.readText()).toBe("key-of-chat");

  await user.click(within(row).getByRole("button", { name: "Refresh Endpoint Key" }));
  expect(
    await screen.findByText("New Endpoint Key for chat; the old one no longer works"),
  ).toBeInTheDocument();
  expect(calls.map((call) => call.route)).toContain("POST /endpoints/chat/refresh-key");
});

test("an Endpoint stops from its row", async () => {
  const calls = fakeApi({ ...routes, "POST /endpoints/chat/stop": [200, { status: "stopped" }] });
  renderApp("/serving");

  const row = (await screen.findByText("chat")).closest("tr")!;
  await userEvent.click(within(row).getByRole("button", { name: /stop/i }));

  expect(await screen.findByText("Stopped chat")).toBeInTheDocument();
  expect(calls.map((call) => call.route)).toContain("POST /endpoints/chat/stop");
});

test("a failure without a reason still says what went wrong", async () => {
  fakeApi({ ...routes, "POST /endpoints/chat/stop": [500, null] });
  renderApp("/serving");

  const row = (await screen.findByText("chat")).closest("tr")!;
  await userEvent.click(within(row).getByRole("button", { name: /stop/i }));

  expect(await screen.findByText("The API answered 500")).toBeInTheDocument();
});

test("a stopped Endpoint is deleted with its trashcan, and only a stopped one has one", async () => {
  let listed = [chat, { ...chat, name: "old", status: "stopped" }];
  const calls = fakeApi({
    ...routes,
    "GET /endpoints": () => [200, listed],
    "DELETE /endpoints/old": () => {
      listed = [chat];
      return [204, null];
    },
  });
  renderApp("/serving");

  const running = (await screen.findByText("chat")).closest("tr")!;
  expect(within(running).queryByRole("button", { name: /delete/i })).not.toBeInTheDocument();
  const stopped = screen.getByText("old").closest("tr")!;
  await userEvent.click(within(stopped).getByRole("button", { name: "Delete old" }));

  expect(await screen.findByText("Deleted old")).toBeInTheDocument();
  expect(screen.queryByText("old")).not.toBeInTheDocument();
  expect(calls.map((call) => call.route)).toContain("DELETE /endpoints/old");
});

test("a failed Endpoint says why and links its saved log", async () => {
  const failed = { ...chat, status: "failed", failure: "Out of GPU memory." };
  fakeApi({ ...routes, "GET /endpoints": [200, [failed]] });
  renderApp("/serving");

  expect(await screen.findByText("Out of GPU memory.")).toBeInTheDocument();
  expect(screen.getByRole("link", { name: /full log/i })).toHaveAttribute(
    "href",
    "/endpoints/chat/log",
  );
});

test("an Endpoint that did not fail shows no failure or log", async () => {
  fakeApi({ ...routes, "GET /endpoints": [200, [chat]] });
  renderApp("/serving");

  await screen.findByText("chat");
  expect(screen.queryByRole("link", { name: /full log/i })).not.toBeInTheDocument();
});

test.each([
  ["running", ["Judge", "Stop"]],
  ["pending", ["Judge", "Stop"]],
  ["waiting for GPU", ["Judge", "Stop"]],
  ["failed", ["Judge", "Restart", "Stop"]],
  ["stopped", ["Judge", "Restart", "Delete chat"]],
])("a %s Endpoint's row offers %j", async (status, actions) => {
  fakeApi({ ...routes, "GET /endpoints": [200, [{ ...chat, status }]] });
  renderApp("/serving");

  const row = (await screen.findByText("chat")).closest("tr")!;
  const cell = row.querySelector(".actions") as HTMLElement;
  const buttons = within(cell).getAllByRole("button");
  expect(buttons.map((button) => button.textContent?.trim() || button.ariaLabel)).toEqual(actions);
});

test("a stopped Endpoint restarts from its row", async () => {
  const calls = fakeApi({
    ...routes,
    "POST /endpoints/old/restart": [200, { ...chat, name: "old", status: "pending" }],
  });
  renderApp("/serving");

  const stopped = (await screen.findByText("old")).closest("tr")!;
  await userEvent.click(within(stopped).getByRole("button", { name: /restart/i }));

  expect(
    await screen.findByText("Restarted old; it answers once vLLM is ready"),
  ).toBeInTheDocument();
  expect(calls.map((call) => call.route)).toContain("POST /endpoints/old/restart");
});

test("the model picker lists our Model Versions first, then open-source models", async () => {
  fakeApi(routes);
  renderApp("/serving");

  await userEvent.click(await screen.findByRole("combobox", { name: /^Model/ }));
  const listbox = await screen.findByRole("listbox");
  const groups = [...listbox.querySelectorAll("[role=group]")].map((group) => group.ariaLabel);
  expect(groups).toEqual(["Our models", "Open source"]);
  // Speculators only draft, so they aren't offered to serve.
  expect(
    within(listbox)
      .getAllByRole("option")
      .map((option) => option.textContent),
  ).toEqual(["qwen-sft@2full weights", "Qwen/Qwen3-8B8.2B"]);
});

test("serving options start with vLLM's defaults, each with an ⓘ and numbers with steppers", async () => {
  fakeApi(routes);
  renderApp("/serving");

  const share = await screen.findByLabelText(/^Gpu Memory Utilization/);
  expect(share).toHaveValue("0.9");
  expect(share).toHaveAccessibleDescription("The share of GPU memory vLLM may take.");
  expect(screen.getByLabelText(/^Max Model Len/)).toHaveAccessibleDescription(
    "vLLM reads it from the model when left empty.",
  );
  await userEvent.click(screen.getByRole("button", { name: "Increase Gpu Memory Utilization" }));
  await userEvent.click(screen.getByRole("button", { name: "Increase Gpu Memory Utilization" }));
  expect(share).toHaveValue("1");
});

test("an Endpoint starts with untouched defaults, or with the options changed", async () => {
  const calls = fakeApi({ ...routes, "POST /endpoints": [201, { ...chat, name: "sft" }] });
  renderApp("/serving");

  await userEvent.type(await screen.findByRole("combobox", { name: /^Model/ }), "model:qwen-sft@2");
  await userEvent.type(screen.getByLabelText(/^Endpoint name/), "sft");
  await userEvent.click(screen.getByRole("button", { name: /^start/i }));
  expect(await screen.findByText(/Started sft/)).toBeInTheDocument();
  const sent = () => calls.filter((call) => call.route === "POST /endpoints").at(-1)!.body;
  expect(sent()).toEqual({ model: "model:qwen-sft@2", name: "sft", ...DEFAULTS });

  await userEvent.type(screen.getByLabelText(/^Max Model Len/), "8192");
  await userEvent.selectOptions(screen.getByLabelText(/^Prefix Caching/), "false");
  await userEvent.click(screen.getByRole("button", { name: /^start/i }));
  await waitFor(() =>
    expect(sent()).toEqual({
      model: "model:qwen-sft@2",
      name: "sft",
      ...DEFAULTS,
      max_model_len: 8192,
      prefix_caching: false,
    }),
  );
});

test("only a Speculator trained for the model can be picked to draft for it", async () => {
  const calls = fakeApi({ ...routes, "POST /endpoints": [201, chat] });
  renderApp("/serving");

  await userEvent.type(await screen.findByRole("combobox", { name: /^Model/ }), "model:qwen-sft@2");
  const picker = screen.getByLabelText(/^Speculative decoding/);
  const options = within(picker).getAllByRole("option") as HTMLOptionElement[];
  expect(options.map((option) => [option.value, option.disabled])).toEqual([
    ["", false],
    ["ngram", false],
    ["model:qwen-sft-speculator@1", false],
    ["model:qwen-sft-speculator@2", true],
  ]);
  await userEvent.selectOptions(picker, "model:qwen-sft-speculator@1");
  const tokens = screen.getByLabelText(/^Draft tokens/);
  expect(tokens).toHaveValue("3");
  expect(tokens).toHaveAccessibleDescription("Tokens drafted per step.");
  await userEvent.click(screen.getByRole("button", { name: "Increase Draft tokens" }));
  await userEvent.click(screen.getByRole("button", { name: "Increase Draft tokens" }));
  await userEvent.click(screen.getByRole("button", { name: /^start/i }));

  await screen.findByText(/Started chat/);
  expect(calls.find((call) => call.route === "POST /endpoints")!.body.speculative).toEqual({
    method: "eagle3",
    model: "model:qwen-sft-speculator@1",
    num_speculative_tokens: 5,
  });
});

test("n-gram drafts without a Speculator", async () => {
  const calls = fakeApi({ ...routes, "POST /endpoints": [201, chat] });
  renderApp("/serving");

  await userEvent.selectOptions(await screen.findByLabelText(/^Speculative decoding/), "ngram");
  await userEvent.click(screen.getByRole("button", { name: /^start/i }));

  await screen.findByText(/Started chat/);
  expect(calls.find((call) => call.route === "POST /endpoints")!.body.speculative).toEqual({
    method: "ngram",
    num_speculative_tokens: 3,
    prompt_lookup_min: 2,
    prompt_lookup_max: 4,
  });
});

test("the benchmark's results show for an FP8 KV cache and a Speculator's method", async () => {
  const kvCache = {
    anyOf: [{ enum: ["auto", "fp8", "fp8_e5m2"], type: "string" }, { type: "null" }],
    default: "auto",
    title: "Kv Cache Dtype",
  };
  const serving = schema.$defs.ServingOptions;
  const benchmark = {
    intro: "We benchmarked these methods on 7 open models.",
    terms: "Ranges are across the models in each size class.",
    results: {
      kv_cache_dtype: {
        fp8: {
          note: "Measured on top of FP8 weights.",
          rows: [{ label: "Speedup, all benchmarks", small: "1.35–1.56×", medium: "1.47–1.64×" }],
        },
      },
      "speculative.method": {
        eagle3: { rows: [{ label: "Quality loss", small: "None", medium: "None" }] },
      },
    },
  };
  fakeApi({
    ...routes,
    "GET /schema": [
      200,
      {
        ...schema,
        $defs: {
          ...schema.$defs,
          ServingOptions: {
            ...serving,
            properties: { ...serving.properties, kv_cache_dtype: kvCache },
          },
        },
        acceleration_benchmark: benchmark,
      },
    ],
  });
  renderApp("/serving");

  await screen.findByLabelText(/^Kv Cache Dtype/);
  expect(screen.queryByText(/We benchmarked these methods/)).not.toBeInTheDocument();
  expect(screen.queryByText(/Ranges are across the models/)).not.toBeInTheDocument();
  // `auto` is no choice the benchmark judged.
  expect(screen.queryByText("Not measured in the benchmark")).not.toBeInTheDocument();
  await userEvent.selectOptions(screen.getByLabelText(/^Kv Cache Dtype/), "fp8");
  expect(screen.getByRole("table", { name: "Benchmark results of fp8" })).toHaveTextContent(
    "Measured on top of FP8 weights.",
  );
  await userEvent.selectOptions(screen.getByLabelText(/^Kv Cache Dtype/), "fp8_e5m2");
  expect(screen.getByText("Not measured in the benchmark")).toBeInTheDocument();

  await userEvent.type(screen.getByRole("combobox", { name: /^Model/ }), "model:qwen-sft@2");
  const picker = screen.getByLabelText(/^Speculative decoding/);
  await userEvent.selectOptions(picker, "model:qwen-sft-speculator@1");
  expect(screen.getByRole("table", { name: "Benchmark results of eagle3" })).toBeInTheDocument();
  await userEvent.selectOptions(picker, "ngram");
  expect(screen.getAllByText("Not measured in the benchmark")).toHaveLength(2);
});

test("a refused start shows why, next to the option when it names one", async () => {
  fakeApi({
    ...routes,
    "POST /endpoints": [
      422,
      { detail: [{ loc: ["body", "max_model_len"], msg: "should be greater than 0" }] },
    ],
  });
  renderApp("/serving");

  await userEvent.type(await screen.findByLabelText(/^Max Model Len/), "0");
  await userEvent.click(screen.getByRole("button", { name: /^start/i }));

  expect(await screen.findByLabelText(/^Max Model Len/)).toHaveAccessibleDescription(
    expect.stringContaining("should be greater than 0"),
  );
});

test("a start the API refuses outright shows its reason", async () => {
  fakeApi({ ...routes, "POST /endpoints": [502, { detail: "Kubernetes did not create it" }] });
  renderApp("/serving");

  await userEvent.click(await screen.findByRole("button", { name: /^start/i }));

  expect(await screen.findByText("Kubernetes did not create it")).toBeInTheDocument();
});

test("a start under a taken name shows why next to the name", async () => {
  const detail = "Endpoint chat already exists; restart it or delete it first";
  fakeApi({ ...routes, "POST /endpoints": [409, { detail }] });
  renderApp("/serving");

  await userEvent.type(await screen.findByLabelText(/^Endpoint name/), "chat");
  await userEvent.click(screen.getByRole("button", { name: /^start/i }));

  expect(await screen.findByLabelText(/^Endpoint name/)).toHaveAccessibleDescription(detail);
});

test("an Endpoint's load and TTFT show in its row, and its name links to its stats", async () => {
  fakeApi({ ...routes, "GET /endpoints": [200, [{ ...chat, stats }]] });
  renderApp("/serving");

  const name = await screen.findByRole("link", { name: "chat" });
  const cells = within(name.closest("tr")!).getAllByRole("cell");
  expect(cells.slice(4, 7).map((cell) => cell.textContent)).toEqual([
    "3 · 1",
    // Tokens/s takes two readings.
    "—",
    "123 ms",
  ]);
  expect(cells[4]).toHaveAttribute("title", "3 running · 1 waiting");
  expect(cells[9]).toHaveTextContent("10-02 08:30");
  expect(cells[9]).toHaveAttribute("title", "2026-10-02 08:30");
  expect(name).toHaveAttribute("href", "/serving/chat");
});

test("an Endpoint without stats shows dashes", async () => {
  fakeApi({ ...routes, "GET /endpoints": [200, [{ ...chat, status: "pending" }]] });
  renderApp("/serving");

  const row = (await screen.findByText("chat")).closest("tr")!;
  const cells = within(row).getAllByRole("cell");
  expect(cells.slice(4, 7).map((cell) => cell.textContent)).toEqual(["—", "—", "—"]);
});

test("tokens/s is the generated tokens between two readings", async () => {
  vi.useFakeTimers({ shouldAdvanceTime: true });
  const readings = [stats, { ...stats, generation_tokens: 1500, read_at: "2026-10-02T09:00:10Z" }];
  fakeApi({ ...routes, "GET /endpoints": () => [200, [{ ...chat, stats: readings.shift() }]] });
  renderApp("/serving");
  await screen.findByText("3 · 1");

  await act(() => vi.advanceTimersByTimeAsync(ENDPOINT_LIST_REFRESH_MS));

  const row = screen.getByRole("link", { name: "chat" }).closest("tr")!;
  expect(within(row).getAllByRole("cell")[5]).toHaveTextContent("50.0");
  vi.useRealTimers();
});

test("the Judge is labelled in the list, without a URL, key or buttons", async () => {
  const judge = { ...chat, name: "judge", judge: true };
  fakeApi({ ...routes, "GET /endpoints": [200, [judge, chat]] });
  renderApp("/serving");

  const row = (await screen.findByText("Judge", { selector: "strong" })).closest("tr")!;
  expect(within(row).getByText("running")).toBeInTheDocument();
  for (const name of [/stop/i, /restart/i, /delete/i, /Endpoint Key/, "Copy URL"]) {
    expect(within(row).queryByRole("button", { name })).not.toBeInTheDocument();
  }
  expect(within(row).queryByRole("link")).not.toBeInTheDocument();
  expect(within(row).getByRole("button", { name: "Copy model" })).toBeInTheDocument();
});

test("Save stores the Judge's model, options and system prompt, and leaves it off", async () => {
  const user = userEvent.setup();
  const calls = fakeApi({ ...routes, "PUT /judge": [200, {}] });
  renderApp("/serving");

  await user.type(await screen.findByRole("combobox", { name: "Judge model" }), "hf:Qwen/Qwen3-8B");
  await user.click(screen.getByText("+ advanced"));
  const systemPrompt = screen.getByRole("textbox", { name: /System prompt/ });
  expect(systemPrompt).toHaveValue("Rate it.");
  await user.clear(systemPrompt);
  await user.type(systemPrompt, "Be strict.");
  await user.click(screen.getByRole("button", { name: "Save" }));

  await waitFor(() => expect(calls.some((call) => call.route === "PUT /judge")).toBe(true));
  const put = calls.find((call) => call.route === "PUT /judge")!;
  expect(put.body).toEqual({
    enabled: false,
    spec: { model: "hf:Qwen/Qwen3-8B", ...DEFAULTS },
    system_prompt: "Be strict.",
  });
  expect(await screen.findByText("Judging is off")).toBeInTheDocument();
  expect(screen.getByRole("button", { name: "Judge is deactivated" })).toBeInTheDocument();
});

const JUDGE_ON = {
  enabled: true,
  spec: { model: "hf:Qwen/Qwen3-8B", max_model_len: 4096 },
  system_prompt: "Rate it.",
  status: "running",
};
const JUDGE_OFF = { ...JUDGE_ON, enabled: false, status: null };

// The Judge's routes, which store what's saved.
function judgeRoutes(judge: object) {
  let stored = judge;
  return {
    "GET /judge": (): [number, unknown] => [200, stored],
    "PUT /judge": (body: object): [number, unknown] => {
      stored = body;
      return [200, stored];
    },
  };
}

test("the button activates the Judge at once, with the form's model and system prompt", async () => {
  const user = userEvent.setup();
  const calls = fakeApi({ ...routes, ...judgeRoutes({ ...JUDGE_OFF, spec: null }) });
  renderApp("/serving");

  const off = await screen.findByRole("button", { name: "Judge is deactivated" });
  expect(off).toHaveClass("danger");
  await user.type(screen.getByRole("combobox", { name: "Judge model" }), "hf:Qwen/Qwen3-8B");
  await user.click(off);

  const on = await screen.findByRole("button", { name: "Judge is activated" });
  expect(on).toHaveClass("success");
  expect(calls.find((call) => call.route === "PUT /judge")?.body).toEqual({
    enabled: true,
    spec: { model: "hf:Qwen/Qwen3-8B", ...DEFAULTS },
    system_prompt: "Rate it.",
  });
});

test("the button deactivates the Judge at once, keeping its model", async () => {
  const user = userEvent.setup();
  const calls = fakeApi({ ...routes, ...judgeRoutes(JUDGE_ON) });
  renderApp("/serving");

  await user.click(await screen.findByRole("button", { name: "Judge is activated" }));

  expect(await screen.findByRole("button", { name: "Judge is deactivated" })).toHaveClass("danger");
  const put = calls.find((call) => call.route === "PUT /judge")!;
  expect(put.body).toMatchObject({ enabled: false, spec: { model: "hf:Qwen/Qwen3-8B" } });
});

test("activating the Judge without a model shows the API's error, and the button stays red", async () => {
  const user = userEvent.setup();
  // As FastAPI answers a failed body validation.
  const msg = "Value error, The Judge needs a `spec` naming the model it serves";
  const detail = [{ type: "value_error", loc: ["body"], msg }];
  fakeApi({ ...routes, "PUT /judge": [422, { detail }] });
  renderApp("/serving");

  await user.click(await screen.findByRole("button", { name: "Judge is deactivated" }));

  expect(
    await screen.findByText("The Judge needs a `spec` naming the model it serves"),
  ).toBeInTheDocument();
  expect(screen.getByRole("button", { name: "Judge is deactivated" })).toHaveClass("danger");
});

test("a save the API refuses but stores otherwise shows the stored state", async () => {
  const user = userEvent.setup();
  const detail = "Can't serve the Judge's model: not found";
  let stored: object = JUDGE_ON;
  fakeApi({
    ...routes,
    "GET /judge": () => [200, stored],
    "PUT /judge": () => {
      stored = JUDGE_OFF;
      return [422, { detail }];
    },
  });
  renderApp("/serving");
  await screen.findByRole("button", { name: "Judge is activated" });

  await user.click(screen.getByRole("button", { name: "Save" }));

  expect(await screen.findByText(detail)).toBeInTheDocument();
  expect(await screen.findByRole("button", { name: "Judge is deactivated" })).toHaveClass("danger");
});

test("the button follows judging switched on elsewhere, keeping the form's edits", async () => {
  vi.useFakeTimers({ shouldAdvanceTime: true });
  const answers = [JUDGE_OFF];
  fakeApi({ ...routes, "GET /judge": () => [200, answers.shift() ?? JUDGE_ON] });
  renderApp("/serving");
  const systemPrompt = await screen.findByRole("textbox", { name: /System prompt/ });
  fireEvent.change(systemPrompt, { target: { value: "Being edited." } });

  await act(() => vi.advanceTimersByTimeAsync(ENDPOINT_LIST_REFRESH_MS));

  expect(screen.getByRole("button", { name: "Judge is activated" })).toBeInTheDocument();
  expect(systemPrompt).toHaveValue("Being edited.");
  vi.useRealTimers();
});

test("the Judge's stored settings fill its form", async () => {
  fakeApi({ ...routes, "GET /judge": [200, JUDGE_ON] });
  renderApp("/serving");

  expect(await screen.findByRole("button", { name: "Judge is activated" })).toBeInTheDocument();
  expect(screen.getByRole("combobox", { name: "Judge model" })).toHaveValue("hf:Qwen/Qwen3-8B");
  await userEvent.click(screen.getByText("+ advanced"));
  // The Judge section comes before the start form.
  expect(screen.getAllByLabelText(/^Max Model Len/)[0]).toHaveDisplayValue("4096");
});

test("an Endpoint opts in to judging from its row, and says when the Judge is stopped", async () => {
  const user = userEvent.setup();
  const judged = { ...chat, judging: { judged: true, every: 10, max_per_hour: 60 } };
  const calls = fakeApi({
    ...routes,
    "GET /endpoints": [200, [judged]],
    "PUT /endpoints/chat/judging": [200, judged],
  });
  renderApp("/serving");

  const row = (await screen.findByText("chat")).closest("tr")!;
  expect(within(row).getByText("Judge stopped")).toBeInTheDocument();
  const settings = await openJudging(user, row);
  const every = within(settings).getByLabelText(/^Every Nth answer/);
  await user.clear(every);
  await user.type(every, "3");
  await user.click(within(settings).getByRole("button", { name: "Save" }));

  await waitFor(() =>
    expect(calls.find((call) => call.route === "PUT /endpoints/chat/judging")?.body).toEqual({
      judged: true,
      every: 3,
      max_per_hour: 60,
    }),
  );
});

// Opens an Endpoint's judging settings, which sit directly below its row.
async function openJudging(user: ReturnType<typeof userEvent.setup>, row: HTMLElement) {
  await user.click(within(row).getByRole("button", { name: "Judge" }));
  return row.nextElementSibling as HTMLElement;
}

test("an Endpoint's judging settings are absent until its Judge button opens them", async () => {
  const user = userEvent.setup();
  fakeApi(routes);
  renderApp("/serving");
  const row = (await screen.findByText("chat")).closest("tr")!;
  const button = within(row).getByRole("button", { name: "Judge" });

  expect(button).toHaveAttribute("aria-expanded", "false");
  expect(screen.queryByText("Judge Configuration:")).not.toBeInTheDocument();
  expect(screen.queryByLabelText(/^Every Nth answer/)).not.toBeInTheDocument();

  // The rest of the row doesn't react to clicks.
  await user.click(within(row).getByText("running"));
  expect(screen.queryByText("Judge Configuration:")).not.toBeInTheDocument();

  const settings = await openJudging(user, row);
  expect(button).toHaveAttribute("aria-expanded", "true");
  expect(within(settings).getByText("Judge Configuration:")).toBeInTheDocument();
  // One cell spanning every column of the table.
  expect(within(settings).getByRole("cell")).toHaveAttribute(
    "colspan",
    String(screen.getAllByRole("columnheader").length),
  );
  expect(within(settings).getByLabelText(/^Judge this Endpoint/)).not.toBeChecked();
  expect(within(settings).getByLabelText(/^Every Nth answer/)).toHaveValue(10);
  expect(within(settings).getByLabelText(/^At most per hour/)).toHaveValue(60);

  await user.click(button);
  expect(button).toHaveAttribute("aria-expanded", "false");
  expect(screen.queryByText("Judge Configuration:")).not.toBeInTheDocument();
});

test("two Endpoints' judging settings can be open at once, each with its own values", async () => {
  const user = userEvent.setup();
  const other = { ...chat, name: "other", judging: { judged: true, every: 5, max_per_hour: 7 } };
  fakeApi({ ...routes, "GET /endpoints": [200, [chat, other]] });
  renderApp("/serving");

  const chatSettings = await openJudging(user, (await screen.findByText("chat")).closest("tr")!);
  const otherSettings = await openJudging(user, screen.getByText("other").closest("tr")!);

  expect(screen.getAllByText("Judge Configuration:")).toHaveLength(2);
  expect(within(chatSettings).getByLabelText(/^Every Nth answer/)).toHaveValue(10);
  expect(within(otherSettings).getByLabelText(/^Every Nth answer/)).toHaveValue(5);
});

test("the platform-managed Judge's row has no judging button", async () => {
  fakeApi({ ...routes, "GET /endpoints": [200, [{ ...chat, name: "judge", judge: true }]] });
  renderApp("/serving");

  const row = (await screen.findByText("Judge", { selector: "strong" })).closest("tr")!;
  expect(within(row).queryByRole("button", { name: "Judge" })).not.toBeInTheDocument();
});

test("a successful save collapses the settings and the banner says so", async () => {
  const user = userEvent.setup();
  fakeApi({ ...routes, "PUT /endpoints/chat/judging": [200, chat] });
  renderApp("/serving");

  const row = (await screen.findByText("chat")).closest("tr")!;
  const settings = await openJudging(user, row);
  await user.click(within(settings).getByLabelText(/^Judge this Endpoint/));
  await user.click(within(settings).getByRole("button", { name: "Save" }));

  expect(await screen.findByText("The Judge scores 1 in 10 answers of chat")).toBeInTheDocument();
  expect(screen.queryByText("Judge Configuration:")).not.toBeInTheDocument();
});

test("reopening right after a save shows what was saved", async () => {
  let stored = chat;
  fakeApi({
    ...routes,
    "GET /endpoints": () => [200, [stored]],
    "PUT /endpoints/chat/judging": (judging) => {
      stored = { ...chat, judging };
      return [200, stored];
    },
  });
  renderApp("/serving");

  const row = (await screen.findByText("chat")).closest("tr")!;
  const button = within(row).getByRole("button", { name: "Judge" });
  fireEvent.click(button);
  const settings = row.nextElementSibling as HTMLElement;
  fireEvent.change(within(settings).getByLabelText(/^Every Nth answer/), {
    target: { value: "3" },
  });

  // The list's reload after the save answers only when released.
  let release = () => {};
  const held = new Promise<void>((resolve) => {
    release = resolve;
  });
  const answer = globalThis.fetch;
  vi.stubGlobal("fetch", async (path: string, init: RequestInit) => {
    if (path === "/endpoints") await held;
    return answer(path, init);
  });
  fireEvent.click(within(settings).getByRole("button", { name: "Save" }));

  expect(await screen.findByText("chat is no longer judged")).toBeInTheDocument();
  expect(button).toHaveAttribute("aria-expanded", "true");

  release();
  await waitFor(() => expect(button).toHaveAttribute("aria-expanded", "false"));
  fireEvent.click(button);
  const reopened = row.nextElementSibling as HTMLElement;
  expect(within(reopened).getByLabelText(/^Every Nth answer/)).toHaveValue(3);
});

test("a failed save keeps the settings open with the edits and the banner shows the error", async () => {
  const user = userEvent.setup();
  fakeApi({ ...routes, "PUT /endpoints/chat/judging": [409, { detail: "Can't judge chat" }] });
  renderApp("/serving");

  const row = (await screen.findByText("chat")).closest("tr")!;
  const settings = await openJudging(user, row);
  const perHour = within(settings).getByLabelText(/^At most per hour/);
  fireEvent.change(perHour, { target: { value: "25" } });
  await user.click(within(settings).getByRole("button", { name: "Save" }));

  expect(await screen.findByText("Can't judge chat")).toBeInTheDocument();
  expect(within(settings).getByLabelText(/^At most per hour/)).toHaveValue(25);
});

test("closing without saving discards the edits", async () => {
  const user = userEvent.setup();
  fakeApi(routes);
  renderApp("/serving");

  const row = (await screen.findByText("chat")).closest("tr")!;
  const edited = await openJudging(user, row);
  fireEvent.change(within(edited).getByLabelText(/^Every Nth answer/), { target: { value: "3" } });
  await user.click(within(row).getByRole("button", { name: "Judge" }));

  const reopened = await openJudging(user, row);
  expect(within(reopened).getByLabelText(/^Every Nth answer/)).toHaveValue(10);
});

test("edits in open judging settings survive the list's refresh", async () => {
  vi.useFakeTimers({ shouldAdvanceTime: true });
  const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime });
  const refreshed = { ...chat, judging: { judged: true, every: 20, max_per_hour: 60 } };
  const answers = [[chat]];
  const calls = fakeApi({
    ...routes,
    "GET /endpoints": () => [200, answers.shift() ?? [refreshed]],
  });
  renderApp("/serving");

  const row = (await screen.findByText("chat")).closest("tr")!;
  const settings = await openJudging(user, row);
  const every = within(settings).getByLabelText(/^Every Nth answer/);
  fireEvent.change(every, { target: { value: "3" } });

  await act(() => vi.advanceTimersByTimeAsync(ENDPOINT_LIST_REFRESH_MS));

  // The list did refresh: the button took the stored colour, the form kept its edit.
  expect(calls.filter((call) => call.route === "GET /endpoints").length).toBeGreaterThan(1);
  expect(within(row).getByRole("button", { name: "Judge" })).toHaveClass("judged");
  expect(every).toHaveValue(3);
  expect(within(settings).getByLabelText(/^Judge this Endpoint/)).not.toBeChecked();
  vi.useRealTimers();
});

test("an Endpoint's row shows its GPU Share", async () => {
  fakeApi(routes);
  renderApp("/serving");

  const row = (await screen.findByText("chat")).closest("tr")!;
  expect(within(row).getByText("85%")).toBeInTheDocument();
  expect(screen.getByRole("columnheader", { name: "GPU Share" })).toBeInTheDocument();
});

test("the GPU Share and length a start would pin show before starting", async () => {
  const calls = fakeApi({
    ...routes,
    "POST /endpoints/preview": [
      200,
      { gpu_memory_utilization: 0.95, max_model_len: 8192, kv_cache_dtype: "fp8" },
    ],
  });
  renderApp("/serving");

  await userEvent.type(await screen.findByRole("combobox", { name: /^Model/ }), "hf:x/y");

  expect(
    await screen.findByText(/GPU Share 95%, up to 8192 tokens, KV cache in fp8/),
  ).toBeInTheDocument();
  const asked = calls.filter((call) => call.route === "POST /endpoints/preview").at(-1)!.body;
  expect(asked).toEqual({ model: "hf:x/y", ...DEFAULTS });
});

test("a model that can't be served shows no GPU Share before starting", async () => {
  const calls = fakeApi({
    ...routes,
    "POST /endpoints/preview": [422, { detail: "Can't serve hf:x/y" }],
  });
  renderApp("/serving");

  await screen.findByRole("combobox", { name: /^Model/ });
  expect(calls.some((call) => call.route === "POST /endpoints/preview")).toBe(false);

  await userEvent.type(screen.getByRole("combobox", { name: /^Model/ }), "hf:x/y");

  await waitFor(() =>
    expect(calls.some((call) => call.route === "POST /endpoints/preview")).toBe(true),
  );
  expect(screen.queryByText(/GPU Share \d+%/)).not.toBeInTheDocument();
});

test("the GPU Share the Judge would run on shows before saving", async () => {
  fakeApi({
    ...routes,
    "POST /endpoints/preview": [200, { gpu_memory_utilization: 0.2, max_model_len: 4096 }],
  });
  renderApp("/serving");

  await userEvent.type(await screen.findByRole("combobox", { name: /^Judge model/ }), "hf:x/y");

  expect(await screen.findByText("GPU Share 20%, up to 4096 tokens")).toBeInTheDocument();
});
