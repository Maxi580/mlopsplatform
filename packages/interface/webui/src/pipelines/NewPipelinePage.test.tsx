import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
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
  const withServe = {
    ...schema,
    properties: {
      ...schema.properties,
      serve: { anyOf: [{ $ref: "#/$defs/Serve" }, { type: "null" }], default: null },
    },
    $defs: {
      Serve: {
        type: "object",
        properties: { max_model_len: { type: "integer", title: "Max Model Len" } },
      },
    },
  };
  const calls = fakeApi({
    "GET /schema": [200, withServe],
    "GET /datasets": [200, []],
    "GET /pipelines": [200, []],
    "POST /pipelines": [202, { id: 5 }],
  });
  renderApp("/pipelines/new");

  const serve = await screen.findByRole("checkbox", { name: /run serve/i });
  expect(screen.queryByLabelText(/^Max Model Len/)).not.toBeInTheDocument();
  await userEvent.click(serve);
  await userEvent.type(screen.getByLabelText(/^Max Model Len/), "4096");
  await userEvent.click(screen.getByRole("button", { name: /submit/i }));

  await screen.findByText(/Submitted Pipeline 5/);
  const submitted = calls.find((call) => call.route === "POST /pipelines")!.body;
  expect(submitted.request.serve).toEqual({ max_model_len: 4096 });
});
