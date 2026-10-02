import { screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { fakeApi, renderApp } from "../testApi";
import type { Dataset, RegisteredModel, Storage } from "./storage";

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
const routes: Parameters<typeof fakeApi>[0] = {
  "GET /datasets": [200, datasets],
  "GET /models": [200, models],
  "GET /storage": [200, storage],
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

test("the sidebar links the Storage page", async () => {
  fakeApi({ ...routes, "GET /pipelines": [200, []] });
  renderApp("/");

  const sidebar = screen.getByRole("complementary");
  expect(within(sidebar).getByRole("link", { name: /storage/i })).toHaveAttribute(
    "href",
    "/storage",
  );
});
