import { screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { fakeApi, renderApp } from "../testApi";
import type { Checkpoint, Dataset, ModelCache, RegisteredModel, Storage } from "./storage";

const datasets: Dataset[] = [
  { name: "chat", versions: [{ version: 1, size_bytes: 2048, row_format: "messages" }] },
];
const models: RegisteredModel[] = [
  {
    name: "qwen-sft",
    versions: [
      {
        version: 3,
        size_bytes: 5 * 2 ** 20,
        tags: { weights: "adapter", base_model: "hf:Qwen/Qwen3@abc", pipeline: "7" },
      },
    ],
  },
];
const storage: Storage = {
  buckets: [
    { name: "mlflow", size_bytes: 30 * 2 ** 30 },
    { name: "platform", size_bytes: 10 * 2 ** 30 },
  ],
  capacity_bytes: 100 * 2 ** 30,
};
const modelCache: ModelCache = {
  entries: [
    {
      kind: "base_model",
      reference: "hf:Qwen/Qwen3-0.6B@abc123",
      size_bytes: 3 * 2 ** 30,
      last_used: "2026-10-01T08:30:00+00:00",
    },
    {
      kind: "benchmark",
      reference: "lm_eval:gsm8k",
      size_bytes: 3 * 2 ** 20,
      last_used: "2026-10-02T09:00:00+00:00",
    },
  ],
  capacity_bytes: 200 * 2 ** 30,
};
const checkpoints: Checkpoint[] = [
  { pipeline_id: 7, pipeline_name: "qwen-sft", phase_index: 1, size_bytes: 3 * 2 ** 30 },
];
const routes: Parameters<typeof fakeApi>[0] = {
  "GET /checkpoints": [200, checkpoints],
  "GET /datasets": [200, datasets],
  "GET /models": [200, models],
  "GET /storage": [200, storage],
  "GET /cache": [200, modelCache],
  "GET /settings": [200, { gpu_count: 1 }],
};

test("the page shows object store usage per bucket", async () => {
  fakeApi({ ...routes });
  renderApp("/storage");

  expect(await screen.findByText("40.0 GB of 100.0 GB used")).toBeInTheDocument();
  const usage = screen.getByRole("meter");
  expect(usage).toHaveAttribute("value", "40");
  expect(screen.getByText("mlflow").closest("li")).toHaveTextContent("30.0 GB");
});

test("Datasets and Registered Models are listed with versions, sizes and lineage", async () => {
  fakeApi({ ...routes });
  renderApp("/storage");

  const dataset = (await screen.findByText("chat@1")).closest("tr")!;
  for (const text of ["messages", "2.0 KB"]) {
    expect(within(dataset).getByText(text)).toBeInTheDocument();
  }
  const model = screen.getByText("qwen-sft@3").closest("tr")!;
  for (const text of ["adapter", "hf:Qwen/Qwen3@abc", "#7", "5.0 MB"]) {
    expect(within(model).getByText(text)).toBeInTheDocument();
  }
});

test("kept Checkpoints are listed with their Pipeline, Phase and size, and deletable", async () => {
  let current = checkpoints;
  const calls = fakeApi({
    ...routes,
    "GET /checkpoints": () => [200, current],
    "DELETE /checkpoints/7/1": () => {
      current = [];
      return [204, null];
    },
  });
  renderApp("/storage");

  const label = "qwen-sft (#7), Phase 2";
  const row = (await screen.findByText(label)).closest("tr")!;
  expect(within(row).getByText("3.0 GB")).toBeInTheDocument();
  await userEvent.click(within(row).getByRole("button", { name: "Delete" }));
  await userEvent.click(within(row).getByRole("button", { name: `Delete ${label}` }));

  expect(await screen.findByText(`Deleted ${label}`)).toBeInTheDocument();
  expect(screen.queryByText(label)).not.toBeInTheDocument();
  expect(calls.map((call) => call.route)).toContain("DELETE /checkpoints/7/1");
});

test("a Dataset Version downloads from its presigned URL", async () => {
  const click = vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(() => {});
  fakeApi({
    ...routes,
    "GET /datasets/chat/versions/1/download": [200, { url: "https://objects.test/chat?sig=x" }],
  });
  renderApp("/storage");

  const row = (await screen.findByText("chat@1")).closest("tr")!;
  await userEvent.click(within(row).getByRole("button", { name: /download/i }));

  expect(click).toHaveBeenCalled();
  expect(click.mock.contexts[0]).toHaveAttribute("href", "https://objects.test/chat?sig=x");
  click.mockRestore();
});

test("delete asks once more, then deletes and reloads the list", async () => {
  let current = models;
  const calls = fakeApi({
    ...routes,
    "GET /models": () => [200, current],
    "DELETE /models/qwen-sft/versions/3": () => {
      current = [];
      return [204, null];
    },
  });
  renderApp("/storage");

  const row = (await screen.findByText("qwen-sft@3")).closest("tr")!;
  await userEvent.click(within(row).getByRole("button", { name: "Delete" }));
  await userEvent.click(within(row).getByRole("button", { name: "Delete qwen-sft@3" }));

  expect(await screen.findByText("Deleted qwen-sft@3")).toBeInTheDocument();
  expect(screen.queryByText("qwen-sft@3")).not.toBeInTheDocument();
  expect(calls.map((call) => call.route)).toContain("DELETE /models/qwen-sft/versions/3");
});

test("a refused delete shows which Pipeline uses the version", async () => {
  const reason = "chat@1 is used by Pipeline qwen-sft (#7)";
  fakeApi({ ...routes, "DELETE /datasets/chat/versions/1": [409, { detail: reason }] });
  renderApp("/storage");

  const row = (await screen.findByText("chat@1")).closest("tr")!;
  await userEvent.click(within(row).getByRole("button", { name: "Delete" }));
  await userEvent.click(within(row).getByRole("button", { name: "Delete chat@1" }));

  expect(await screen.findByText(reason)).toBeInTheDocument();
  expect(screen.getByText("chat@1")).toBeInTheDocument();
});

test("the Model Cache lists Base Models and benchmarks with size and last use", async () => {
  fakeApi({ ...routes });
  renderApp("/storage");

  const row = (await screen.findByText("hf:Qwen/Qwen3-0.6B@abc123")).closest("tr")!;
  for (const text of ["Base Model", "2026-10-01 08:30", "3.0 GB"]) {
    expect(within(row).getByText(text)).toBeInTheDocument();
  }
  const benchmark = screen.getByText("lm_eval:gsm8k").closest("tr")!;
  for (const text of ["Benchmark", "2026-10-02 09:00", "3.0 MB"]) {
    expect(within(benchmark).getByText(text)).toBeInTheDocument();
  }
  expect(screen.getByText("3.0 GB of 200.0 GB used")).toBeInTheDocument();
});

test("freeing a cached Base Model a Pipeline uses shows the reason", async () => {
  const reason = "hf:Qwen/Qwen3-0.6B@abc123 is used by Pipeline qwen-sft (#7)";
  const calls = fakeApi({
    ...routes,
    "DELETE /cache?reference=hf%3AQwen%2FQwen3-0.6B%40abc123": [409, { detail: reason }],
  });
  renderApp("/storage");

  const row = (await screen.findByText("hf:Qwen/Qwen3-0.6B@abc123")).closest("tr")!;
  await userEvent.click(within(row).getByRole("button", { name: "Delete" }));
  await userEvent.click(
    within(row).getByRole("button", { name: "Delete hf:Qwen/Qwen3-0.6B@abc123" }),
  );

  expect(await screen.findByText(reason)).toBeInTheDocument();
  expect(calls.map((call) => call.route)).toContain(
    "DELETE /cache?reference=hf%3AQwen%2FQwen3-0.6B%40abc123",
  );
});

test("the sidebar links the Storage page", async () => {
  fakeApi({ ...routes, "GET /pipelines": [200, []] });
  renderApp("/");

  const sidebar = screen.getByRole("complementary");
  expect(within(sidebar).getByRole("link", { name: /storage/i })).toHaveAttribute(
    "href",
    "/storage",
  );
});

function directoryFile(path: string, content: string): File {
  const file = new File([content], path.split("/").pop()!);
  Object.defineProperty(file, "webkitRelativePath", { value: `my-model/${path}` });
  return file;
}

test("a model directory uploads in parts straight to the object store, then registers", async () => {
  const started = {
    id: "abc",
    part_size_bytes: 4,
    files: [
      { path: "config.json", part_urls: ["https://objects.test/c1"] },
      {
        path: "model.safetensors",
        part_urls: ["https://objects.test/m1", "https://objects.test/m2"],
      },
    ],
  };
  const calls = fakeApi({
    ...routes,
    "POST /models/uploads": [201, started],
    "PUT https://objects.test/c1": [200, null],
    "PUT https://objects.test/m1": [200, null],
    "PUT https://objects.test/m2": [200, null],
    "POST /models/uploads/abc/complete": [201, { name: "my-model", version: 1 }],
  });
  renderApp("/storage");

  await userEvent.type(await screen.findByLabelText("Name"), "my-model");
  await userEvent.upload(screen.getByLabelText("Model directory"), [
    directoryFile("config.json", "{}"),
    directoryFile("model.safetensors", "weights"),
    directoryFile(".git/HEAD", "ref"),
  ]);
  await userEvent.click(screen.getByRole("button", { name: /upload/i }));

  expect(await screen.findByText("Uploaded my-model@1")).toBeInTheDocument();
  const start = calls.find((call) => call.route === "POST /models/uploads")!;
  expect(start.body).toEqual({
    name: "my-model",
    files: [
      { path: "config.json", size_bytes: 2 },
      { path: "model.safetensors", size_bytes: 7 },
    ],
    base: null,
    tool_parser: null,
  });
  const parts = calls.filter((call) => call.route.startsWith("PUT "));
  expect(parts.map((call) => [call.route, call.body.size])).toEqual([
    ["PUT https://objects.test/c1", 2],
    ["PUT https://objects.test/m1", 4],
    ["PUT https://objects.test/m2", 3],
  ]);
});

test("a refused model upload shows the reason", async () => {
  const reason = "Rejected my-model: config.json is missing";
  fakeApi({ ...routes, "POST /models/uploads": [422, { detail: reason }] });
  renderApp("/storage");

  await userEvent.type(await screen.findByLabelText("Name"), "my-model");
  await userEvent.upload(screen.getByLabelText("Model directory"), [
    directoryFile("model.safetensors", "weights"),
  ]);
  await userEvent.click(screen.getByRole("button", { name: /upload/i }));

  expect(await screen.findByText(reason)).toBeInTheDocument();
});

test("a Model Version's download lists a link per file", async () => {
  const files = [
    { path: "adapter_config.json", size_bytes: 300, url: "https://objects.test/a?sig=x" },
    { path: "adapter_model.safetensors", size_bytes: 2048, url: "https://objects.test/m?sig=x" },
  ];
  fakeApi({ ...routes, "GET /models/qwen-sft/versions/3/files": [200, { files }] });
  renderApp("/storage");

  const row = (await screen.findByText("qwen-sft@3")).closest("tr")!;
  await userEvent.click(within(row).getByRole("button", { name: /download/i }));

  const link = await screen.findByRole("link", { name: "adapter_model.safetensors" });
  expect(link).toHaveAttribute("href", "https://objects.test/m?sig=x");
  expect(screen.getByRole("link", { name: "adapter_config.json" })).toBeInTheDocument();
});
