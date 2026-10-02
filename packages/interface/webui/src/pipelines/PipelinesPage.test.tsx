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
  created_at: new Date().toISOString(),
  kubeflow_run_url: "/pipeline/#/runs/details/run-1",
  mlflow_run_url: "https://platform.test/mlflow/#/runs/abc",
};

test("the page shows the GPU count and links the KFP and MLflow UIs", async () => {
  fakeApi({ "GET /pipelines": [200, []], "GET /settings": [200, { gpu_count: 4 }] });
  renderApp("/");

  expect(await screen.findByText("4 GPUs")).toBeInTheDocument();
  const sidebar = screen.getByRole("complementary");
  expect(within(sidebar).getByRole("link", { name: /kubeflow/i })).toHaveAttribute("href", "/pipeline/");
  expect(within(sidebar).getByRole("link", { name: /mlflow/i })).toHaveAttribute("href", "/mlflow/");
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
    running.mlflow_run_url,
  ]);
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
