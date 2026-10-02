import { screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { fakeApi, renderApp } from "../testApi";
import type { Endpoint } from "./endpoint";

const schema = {
  type: "object",
  properties: { serve: { anyOf: [{ $ref: "#/$defs/Serve" }, { type: "null" }] } },
  $defs: {
    Serve: {
      type: "object",
      properties: {
        max_model_len: { anyOf: [{ type: "integer" }, { type: "null" }], title: "Max Model Len" },
        prefix_caching: { type: "boolean", default: true, title: "Prefix Caching" },
        name: { anyOf: [{ type: "string" }, { type: "null" }], title: "Endpoint name" },
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
};
const routes: Parameters<typeof fakeApi>[0] = {
  "GET /schema": [200, schema],
  "GET /endpoints": [200, [chat, { ...chat, name: "old", status: "stopped" }]],
  "GET /models": [200, [{ name: "qwen-sft", versions: [{ version: 2 }] }]],
  "GET /cache/base-models": [200, { base_models: [{ reference: "hf:Qwen/Qwen3@abc" }] }],
  "GET /settings": [200, { gpu_count: 1 }],
};

test("Endpoints are listed with model, status and URL", async () => {
  fakeApi(routes);
  renderApp("/serving");

  const row = (await screen.findByText("chat")).closest("tr")!;
  for (const text of ["model:qwen-sft@2", "running"]) {
    expect(within(row).getByText(text)).toBeInTheDocument();
  }
  expect(within(row).getByRole("link")).toHaveAttribute("href", "/endpoints/chat/v1");
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
