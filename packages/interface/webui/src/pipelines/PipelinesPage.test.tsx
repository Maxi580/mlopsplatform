import { act, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { PIPELINE_LIST_REFRESH_MS } from "../config";
import { fakeApi, renderApp } from "../testApi";
import type { Pipeline } from "./pipeline";

const running: Pipeline = {
  id: 7,
  name: "qwen-sft",
  owner: "shared",
  status: "running",
  stages: ["finetune"],
  gpu_shares: { finetune: 0.3 },
  created_at: new Date().toISOString(),
  kubeflow_run_url: "/pipeline/#/runs/details/run-1",
  mlflow_run_url: "https://platform.test/mlflow/#/runs/abc",
};

test("the page shows the GPU count and links the KFP and MLflow UIs", async () => {
  fakeApi({ "GET /pipelines": [200, []], "GET /settings": [200, { gpu_count: 4 }] });
  renderApp("/");

  expect(await screen.findByText("4 GPUs")).toBeInTheDocument();
  const sidebar = screen.getByRole("complementary");
  expect(within(sidebar).getByRole("link", { name: /kubeflow/i })).toHaveAttribute(
    "href",
    "/pipeline/",
  );
  expect(within(sidebar).getByRole("link", { name: /mlflow/i })).toHaveAttribute(
    "href",
    "/mlflow/",
  );
});

test("each Pipeline shows its Owner, status, Stages and links", async () => {
  fakeApi({ "GET /pipelines": [200, [running]], "GET /settings": [200, { gpu_count: 1 }] });
  renderApp("/");

  const row = (await screen.findByText("qwen-sft")).closest("tr")!;
  for (const text of ["shared", "running", "finetune", "#7"]) {
    expect(within(row).getByText(text)).toBeInTheDocument();
  }
  const links = within(row).getAllByRole("link");
  expect(links.map((link) => link.getAttribute("href"))).toEqual([
    running.kubeflow_run_url,
    "/mlflow/#/runs/abc",
  ]);
});

test("a failed Pipeline says why", async () => {
  const failed: Pipeline = { ...running, status: "failed", failure: "Out of GPU memory." };
  fakeApi({ "GET /pipelines": [200, [failed]], "GET /settings": [200, { gpu_count: 1 }] });
  renderApp("/");

  expect(await screen.findByText("Out of GPU memory.")).toBeInTheDocument();
});

const shareSchema = {
  properties: { gpu_share: { description: "Share to reserve for training." } },
  $defs: {
    ServingOptions: {
      properties: { gpu_memory_utilization: { description: "Share to reserve for vLLM." } },
    },
  },
};
const resumeRoutes = (pipeline: Pipeline): Parameters<typeof fakeApi>[0] => ({
  "GET /pipelines": [200, [pipeline]],
  "GET /settings": [200, { gpu_count: 1 }],
  "GET /schema": [200, shareSchema],
  "GET /gpus/reserved": [200, { reserved_percent: 35, recommended_max: 0.85 }],
  "POST /pipelines/7/resume": [202, { id: 8 }],
});

test.each([
  ["failed", true],
  ["cancelled", true],
  ["succeeded", false],
  ["running", false],
])("a %s Pipeline offering Resume is %s", async (status, offered) => {
  fakeApi(resumeRoutes({ ...running, status }));
  renderApp("/");

  await screen.findByText("qwen-sft");
  expect(!!screen.queryByRole("button", { name: "Resume" })).toBe(offered);
});

test("a Smoke Test can't be resumed", async () => {
  fakeApi(resumeRoutes({ ...running, status: "failed", cases: { sft: "failed" } }));
  renderApp("/");

  await screen.findByText("qwen-sft");
  expect(screen.queryByRole("button", { name: "Resume" })).not.toBeInTheDocument();
});

test("a Resume left empty keeps the stored shares", async () => {
  const calls = fakeApi(resumeRoutes({ ...running, status: "failed" }));
  renderApp("/");

  await userEvent.click(await screen.findByRole("button", { name: "Resume" }));
  expect(screen.getByRole("textbox", { name: /^GPU Share/ })).toHaveAttribute("placeholder", "0.3");
  await userEvent.click(screen.getByRole("button", { name: "Resume Pipeline 7" }));

  expect(
    await screen.findByText("Submitted Pipeline 8, which resumes Pipeline 7"),
  ).toBeInTheDocument();
  const resumed = calls.find((call) => call.route === "POST /pipelines/7/resume");
  expect(resumed?.body).toEqual({ secrets: {} });
});

test("a Resume sends new shares for each Stage holding one, and Secrets", async () => {
  const failed = {
    ...running,
    status: "failed",
    stages: ["distill", "finetune", "evaluate"],
    gpu_shares: { distill: 0.2, finetune: 0.3, evaluate: 0.25 },
  };
  const calls = fakeApi(resumeRoutes(failed));
  renderApp("/");

  await userEvent.click(await screen.findByRole("button", { name: "Resume" }));
  await userEvent.type(screen.getByRole("textbox", { name: /^GPU Share gpu_share/ }), "0.7");
  await userEvent.type(screen.getByRole("textbox", { name: /^GPU Share of evaluate/ }), "0.4");
  await userEvent.type(screen.getByLabelText(/Hugging Face token/), "hf_secret");
  await userEvent.click(screen.getByRole("button", { name: "Resume Pipeline 7" }));

  await screen.findByText("Submitted Pipeline 8, which resumes Pipeline 7");
  const resumed = calls.find((call) => call.route === "POST /pipelines/7/resume");
  expect(resumed?.body).toEqual({
    secrets: { hf_token: "hf_secret" },
    gpu_share: 0.7,
    serving_shares: { evaluate: 0.4 },
  });
});

test("the Resume fields say what they mean and how full the fullest machine is", async () => {
  fakeApi(resumeRoutes({ ...running, status: "cancelled" }));
  renderApp("/");

  await userEvent.click(await screen.findByRole("button", { name: "Resume" }));

  expect(await screen.findByText("Share to reserve for training.")).toBeInTheDocument();
  expect(
    await screen.findByText("Currently reserved: 35% of this GPU · Recommended max: 85%"),
  ).toBeInTheDocument();
});

test("a Pipeline that swept shows the best parameters it found", async () => {
  const swept: Pipeline = {
    ...running,
    stages: ["sweep"],
    sweep: {
      parameters: { settings: { learning_rate: 0.0003 }, lora: { r: 16 } },
      objective: 0.42,
    },
  };
  fakeApi({ "GET /pipelines": [200, [swept]], "GET /settings": [200, { gpu_count: 1 }] });
  renderApp("/");

  const row = (await screen.findByText("qwen-sft")).closest("tr")!;
  expect(
    within(row).getByText("Best: learning_rate=0.0003, r=16 (objective 0.42)"),
  ).toBeInTheDocument();
});

test("the list updates live", async () => {
  vi.useFakeTimers({ shouldAdvanceTime: true });
  let status = "running";
  fakeApi({
    "GET /pipelines": () => [200, [{ ...running, status }]],
    "GET /settings": [200, { gpu_count: 1 }],
  });
  renderApp("/");
  expect(await screen.findByText("running")).toBeInTheDocument();

  status = "succeeded";
  await act(() => vi.advanceTimersByTimeAsync(PIPELINE_LIST_REFRESH_MS));

  expect(await screen.findByText("succeeded")).toBeInTheDocument();
  vi.useRealTimers();
});

test("a finished Pipeline is deleted with its trashcan, and only a finished one has one", async () => {
  let listed = [
    { ...running, id: 8, name: "qwen-dpo" },
    { ...running, status: "succeeded" },
  ];
  const calls = fakeApi({
    "GET /pipelines": () => [200, listed],
    "GET /settings": [200, { gpu_count: 1 }],
    "DELETE /pipelines/7": () => {
      listed = listed.filter((pipeline) => pipeline.id !== 7);
      return [204, null];
    },
  });
  renderApp("/");

  const unfinished = (await screen.findByText("qwen-dpo")).closest("tr")!;
  expect(within(unfinished).queryByRole("button", { name: /delete/i })).not.toBeInTheDocument();
  await userEvent.click(screen.getByRole("button", { name: "Delete Pipeline 7" }));

  expect(await screen.findByText("Deleted Pipeline 7")).toBeInTheDocument();
  expect(screen.queryByText("qwen-sft")).not.toBeInTheDocument();
  expect(calls.map((call) => call.route)).toContain("DELETE /pipelines/7");
});

test("cancel asks once more, then cancels", async () => {
  let status = "running";
  const calls = fakeApi({
    "GET /pipelines": () => [200, [{ ...running, status }]],
    "GET /settings": [200, { gpu_count: 1 }],
    "POST /pipelines/7/cancel": () => {
      status = "cancelled";
      return [200, { id: 7, status }];
    },
  });
  renderApp("/");

  await userEvent.click(await screen.findByRole("button", { name: "Cancel" }));
  await userEvent.click(screen.getByRole("button", { name: "Cancel Pipeline" }));

  expect(await screen.findByText("Cancelled Pipeline 7")).toBeInTheDocument();
  expect(await screen.findByText("cancelled")).toBeInTheDocument();
  expect(calls.map((call) => call.route)).toContain("POST /pipelines/7/cancel");
  expect(screen.queryByRole("button", { name: "Cancel" })).not.toBeInTheDocument();
});

test("a Pipeline's row shows the GPU Share of its Stages", async () => {
  fakeApi({
    "GET /pipelines": [
      200,
      [
        {
          ...running,
          stages: ["finetune", "evaluate"],
          gpu_shares: { finetune: 0.3, evaluate: 0.25 },
        },
      ],
    ],
    "GET /settings": [200, { gpu_count: 1 }],
  });
  renderApp("/");

  expect(await screen.findByText("GPU Share: finetune 30%, evaluate 25%")).toBeInTheDocument();
});
