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
