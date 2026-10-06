import { act, screen, within } from "@testing-library/react";
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
        max_model_len: { anyOf: [{ type: "integer" }, { type: "null" }], title: "Max Model Len" },
        prefix_caching: { type: "boolean", default: true, title: "Prefix Caching" },
      },
    },
  },
};
const chat: Endpoint = {
  name: "chat",
  owner: "shared",
  model: "model:qwen-sft@2",
  status: "running",
  url: "/endpoints/chat/v1",
  created_at: "2026-10-02T08:30:00+00:00",
  stats: null,
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
  "GET /cache": [200, { entries: [{ kind: "base_model", reference: "hf:Qwen/Qwen3@abc" }] }],
  "GET /settings": [200, { gpu_count: 1 }],
};

test("Endpoints are listed with model, status and URL", async () => {
  fakeApi(routes);
  renderApp("/serving");

  const row = (await screen.findByText("chat")).closest("tr")!;
  for (const text of ["model:qwen-sft@2", "running"]) {
    expect(within(row).getByText(text)).toBeInTheDocument();
  }
  expect(within(row).getByRole("link", { name: "/endpoints/chat/v1" })).toHaveAttribute(
    "href",
    "/endpoints/chat/v1",
  );
  const stopped = screen.getByText("old").closest("tr")!;
  expect(within(stopped).queryByRole("button", { name: /stop/i })).not.toBeInTheDocument();
  expect(within(stopped).queryByRole("link")).not.toBeInTheDocument();
});

test("an Endpoint stops from its row", async () => {
  const calls = fakeApi({ ...routes, "POST /endpoints/chat/stop": [200, { status: "stopped" }] });
  renderApp("/serving");

  const row = (await screen.findByText("chat")).closest("tr")!;
  await userEvent.click(within(row).getByRole("button", { name: /stop/i }));

  expect(await screen.findByText("Stopped chat")).toBeInTheDocument();
  expect(calls.map((call) => call.route)).toContain("POST /endpoints/chat/stop");
});

test("an Endpoint starts for a Model Version or Base Model with the curated options", async () => {
  const calls = fakeApi({ ...routes, "POST /endpoints": [201, { ...chat, name: "sft" }] });
  renderApp("/serving");

  const model = await screen.findByLabelText(/^Model/);
  const suggestions = [...document.querySelectorAll("#served-models option")].map(
    (option) => (option as HTMLOptionElement).value,
  );
  expect(suggestions).toEqual(["model:qwen-sft@2", "hf:Qwen/Qwen3@abc"]);
  await userEvent.type(model, "model:qwen-sft@2");
  await userEvent.type(screen.getByLabelText(/^Endpoint name/), "sft");
  await userEvent.type(screen.getByLabelText(/^Max Model Len/), "8192");
  await userEvent.selectOptions(screen.getByLabelText(/^Prefix Caching/), "false");
  await userEvent.click(screen.getByRole("button", { name: /start/i }));

  expect(await screen.findByText(/Started sft/)).toBeInTheDocument();
  expect(calls.find((call) => call.route === "POST /endpoints")!.body).toEqual({
    model: "model:qwen-sft@2",
    name: "sft",
    max_model_len: 8192,
    prefix_caching: false,
  });
});

test("only a Speculator trained for the model can be picked to draft for it", async () => {
  const calls = fakeApi({ ...routes, "POST /endpoints": [201, chat] });
  renderApp("/serving");

  await userEvent.type(await screen.findByLabelText(/^Model/), "model:qwen-sft@2");
  const picker = screen.getByLabelText(/^Speculative decoding/);
  const options = within(picker).getAllByRole("option") as HTMLOptionElement[];
  expect(options.map((option) => [option.value, option.disabled])).toEqual([
    ["", false],
    ["ngram", false],
    ["model:qwen-sft-speculator@1", false],
    ["model:qwen-sft-speculator@2", true],
  ]);
  await userEvent.selectOptions(picker, "model:qwen-sft-speculator@1");
  await userEvent.clear(screen.getByLabelText(/^Draft tokens/));
  await userEvent.type(screen.getByLabelText(/^Draft tokens/), "5");
  await userEvent.click(screen.getByRole("button", { name: /start/i }));

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
  await userEvent.click(screen.getByRole("button", { name: /start/i }));

  await screen.findByText(/Started chat/);
  expect(calls.find((call) => call.route === "POST /endpoints")!.body.speculative).toEqual({
    method: "ngram",
    num_speculative_tokens: 3,
    prompt_lookup_min: 2,
    prompt_lookup_max: 4,
  });
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
  await userEvent.click(screen.getByRole("button", { name: /start/i }));

  expect(await screen.findByLabelText(/^Max Model Len/)).toHaveAccessibleDescription(
    "should be greater than 0",
  );
});

test("a start the API refuses outright shows its reason", async () => {
  fakeApi({ ...routes, "POST /endpoints": [409, { detail: "Endpoint chat is already running" }] });
  renderApp("/serving");

  await userEvent.click(await screen.findByRole("button", { name: /start/i }));

  expect(await screen.findByText("Endpoint chat is already running")).toBeInTheDocument();
});

test("an Endpoint's load and TTFT show in its row, and its name links to its stats", async () => {
  fakeApi({ ...routes, "GET /endpoints": [200, [{ ...chat, stats }]] });
  renderApp("/serving");

  const name = await screen.findByRole("link", { name: "chat" });
  const cells = within(name.closest("tr")!).getAllByRole("cell");
  expect(cells.slice(3, 6).map((cell) => cell.textContent)).toEqual([
    "3 running · 1 waiting",
    // Tokens/s takes two readings.
    "—",
    "123 ms",
  ]);
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
  await screen.findByText("3 running · 1 waiting");

  await act(() => vi.advanceTimersByTimeAsync(ENDPOINT_LIST_REFRESH_MS));

  const row = screen.getByRole("link", { name: "chat" }).closest("tr")!;
  expect(within(row).getAllByRole("cell")[4]).toHaveTextContent("50.0");
  vi.useRealTimers();
});
