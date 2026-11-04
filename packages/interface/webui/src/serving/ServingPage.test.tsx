import { act, screen, waitFor, within } from "@testing-library/react";
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
const URL = "https://platform.test/serving/0123456789abcdef0123456789abcdef/v1";
const chat: Endpoint = {
  name: "chat",
  owner: "shared",
  model: "model:qwen-sft@2",
  status: "running",
  url: URL,
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
  "GET /judge": [200, { enabled: false, spec: null, rubric: "Rate it.", status: null }],
};

test("Endpoints are listed with model, status and URL", async () => {
  fakeApi(routes);
  renderApp("/serving");

  const row = (await screen.findByText("chat")).closest("tr")!;
  for (const text of ["model:qwen-sft@2", "running", URL]) {
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

  expect(await navigator.clipboard.readText()).toBe(URL);
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
  for (const value of ["model:qwen-sft@2", URL]) {
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

test.each([
  ["running", ["Stop"]],
  ["pending", ["Stop"]],
  ["failed", ["Restart", "Stop"]],
  ["stopped", ["Restart", "Delete chat"]],
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
    await screen.findByText("Restarted old; it is pending until vLLM is ready"),
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
  expect(cells.slice(3, 6).map((cell) => cell.textContent)).toEqual([
    "3 · 1",
    // Tokens/s takes two readings.
    "—",
    "123 ms",
  ]);
  expect(cells[3]).toHaveAttribute("title", "3 running · 1 waiting");
  expect(cells[8]).toHaveTextContent("10-02 08:30");
  expect(cells[8]).toHaveAttribute("title", "2026-10-02 08:30");
  expect(name).toHaveAttribute("href", "/serving/chat");
});

test("an Endpoint without stats shows dashes", async () => {
  fakeApi({ ...routes, "GET /endpoints": [200, [{ ...chat, status: "pending" }]] });
  renderApp("/serving");

  const row = (await screen.findByText("chat")).closest("tr")!;
  const cells = within(row).getAllByRole("cell");
  expect(cells.slice(3, 6).map((cell) => cell.textContent)).toEqual(["—", "—", "—"]);
});

test("tokens/s is the generated tokens between two readings", async () => {
  vi.useFakeTimers({ shouldAdvanceTime: true });
  const readings = [stats, { ...stats, generation_tokens: 1500, read_at: "2026-10-02T09:00:10Z" }];
  fakeApi({ ...routes, "GET /endpoints": () => [200, [{ ...chat, stats: readings.shift() }]] });
  renderApp("/serving");
  await screen.findByText("3 · 1");

  await act(() => vi.advanceTimersByTimeAsync(ENDPOINT_LIST_REFRESH_MS));

  const row = screen.getByRole("link", { name: "chat" }).closest("tr")!;
  expect(within(row).getAllByRole("cell")[4]).toHaveTextContent("50.0");
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

test("the Judge's settings are saved with its model, options and rubric", async () => {
  const user = userEvent.setup();
  const saved = { enabled: true, spec: { model: "hf:Qwen/Qwen3-8B" }, rubric: "Be strict." };
  const calls = fakeApi({ ...routes, "PUT /judge": [200, { ...saved, status: "pending" }] });
  renderApp("/serving");

  await user.click(await screen.findByRole("checkbox", { name: "Judging enabled" }));
  await user.type(screen.getByRole("combobox", { name: "Judge model" }), "hf:Qwen/Qwen3-8B");
  await user.click(screen.getByText("+ advanced"));
  const rubric = screen.getByRole("textbox", { name: /Rubric/ });
  expect(rubric).toHaveValue("Rate it.");
  await user.clear(rubric);
  await user.type(rubric, "Be strict.");
  await user.click(screen.getByRole("button", { name: "Save" }));

  await waitFor(() => expect(calls.some((call) => call.route === "PUT /judge")).toBe(true));
  const put = calls.find((call) => call.route === "PUT /judge")!;
  expect(put.body).toEqual({
    enabled: true,
    spec: { model: "hf:Qwen/Qwen3-8B", ...DEFAULTS },
    rubric: "Be strict.",
  });
  expect(await screen.findByText("Judging is on; the Judge starts")).toBeInTheDocument();
});

test("the Judge's stored settings fill its form", async () => {
  const stored = { enabled: true, spec: { model: "hf:Qwen/Qwen3-8B", max_model_len: 4096 } };
  fakeApi({ ...routes, "GET /judge": [200, { ...stored, rubric: "Rate it.", status: "running" }] });
  renderApp("/serving");

  expect(await screen.findByRole("checkbox", { name: "Judging enabled" })).toBeChecked();
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
  await user.click(within(row).getByText("Judge"));
  const every = within(row).getByLabelText("Every Nth answer");
  await user.clear(every);
  await user.type(every, "3");
  await user.click(within(row).getByRole("button", { name: "Save" }));

  await waitFor(() =>
    expect(calls.find((call) => call.route === "PUT /endpoints/chat/judging")?.body).toEqual({
      judged: true,
      every: 3,
      max_per_hour: 60,
    }),
  );
});
