import { screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { fakeApi, renderApp } from "../testApi";
import type {
  Endpoint,
  EndpointHistory,
  EndpointRequests,
  EndpointStats,
  Histogram,
  RequestDetail,
  RequestTrace,
} from "./endpoint";

const ttftHistogram: Histogram = { p50: 0.05, p95: 0.2, mean: 0.1, sum: 1, count: 10 };
const value = (key: string, label: string, explanation: string, reading: number) => ({
  key,
  label,
  explanation,
  value: reading,
});

const stats: EndpointStats = {
  read_at: "2026-10-02T09:00:00Z",
  sections: [
    {
      title: "Requests",
      values: [
        value("running", "Running", "Requests the GPU is generating tokens for.", 2),
        value("waiting", "Waiting", "Requests queued.", 0),
      ],
    },
    {
      title: "KV cache",
      values: [
        value("kv_cache_usage", "KV cache usage", "How full it is; not the hit rate.", 0.42),
        value("prefix_cache_hit_rate", "Prefix-cache hit rate", "Share found.", 0.25),
      ],
    },
    {
      title: "Latency",
      values: [
        {
          key: "time_to_first_token",
          label: "Time to first token",
          explanation: "Seconds to the first token.",
          histogram: ttftHistogram,
        },
      ],
    },
  ],
  metrics: [
    {
      name: "vllm:request_prompt_tokens",
      type: "histogram",
      help: "Number of prefill tokens processed.",
      histogram: { p50: 12, p95: 30, mean: 14.5, sum: 145, count: 10 },
    },
  ],
};

// Two Server Samples 10 s apart; the second is the first idle one, without latencies.
const history: EndpointHistory = {
  start: 1000,
  end: 1900,
  values: {
    running: [
      [1800, 2],
      [1810, 0],
    ],
    generation_tokens_per_second: [
      [1800, 80],
      [1810, 0],
    ],
    kv_cache_usage: [
      [1800, 0.5],
      [1810, 0],
    ],
    acceptance_rate: [],
  },
  latencies: {
    time_to_first_token: { p50: [[1800, 0.3]], p95: [[1800, 0.48]], mean: [[1800, 0.3]] },
  },
  feedback: { up: [[1500, 3]], down: [[1500, 1]] },
  judge_score: [[1500, 3.5]],
};

const routes: Record<string, [number, unknown]> = {
  "GET /endpoints/chat/stats": [200, stats],
  "GET /endpoints/chat/history?span=live": [200, history],
  "GET /endpoints/chat/requests?order=time": [200, { requests: [], next_page_token: null }],
};

const chart = (title: string) => screen.getByRole("heading", { name: title }).closest("figure")!;

test("the live values show the KV cache bar, labelled values and every metric", async () => {
  fakeApi({ ...routes });
  renderApp("/serving/chat");

  const bar = await screen.findByRole("meter", { name: "KV cache usage" });
  expect(bar).toHaveAttribute("value", "42");
  expect(bar.closest("section")).toHaveTextContent("How full it is; not the hit rate.");

  const latency = screen.getByRole("heading", { name: "Latency" }).closest("section")!;
  const ttft = within(latency).getByText("Time to first token").parentElement!;
  expect(ttft).toHaveTextContent("p50 50 ms · p95 200 ms · mean 100 ms");
  expect(
    screen.getByText("Prefix-cache hit rate", { selector: "dt" }).parentElement!,
  ).toHaveTextContent("25.0 %");

  const metric = screen.getByText("vllm:request_prompt_tokens").closest("tr")!;
  expect(metric).toHaveTextContent("p50 12 · p95 30 · mean 14.5");
});

test("the charts show the stored history, in a Requests and a Server area", async () => {
  fakeApi({ ...routes });
  renderApp("/serving/chat");
  await screen.findByRole("heading", { name: "Running/waiting" });

  const area = (title: string) => chart(title).closest("section")!;
  expect(within(area("Tokens/s")).getByRole("heading", { level: 2 })).toHaveTextContent("Requests");
  expect(within(area("KV-cache usage")).getByRole("heading", { level: 2 })).toHaveTextContent(
    "Server",
  );
  // The last sample is idle: tokens/s dropped to 0, and TTFT has no point there.
  expect(within(chart("Tokens/s")).getByText("Generated").closest("div")).toHaveTextContent(
    "0.0/s",
  );
  expect(within(chart("Time to first token")).getByText("p50").closest("div")).toHaveTextContent(
    "—",
  );
  expect(within(chart("KV-cache usage")).getByText("Usage").closest("div")).toHaveTextContent(
    "0.0 %",
  );
  // Without speculative decoding, its chart has no points and isn't shown.
  expect(screen.queryByText("Speculative acceptance rate")).not.toBeInTheDocument();
});

test("the timespan picker asks for that span's history", async () => {
  const calls = fakeApi({ ...routes, "GET /endpoints/chat/history?span=7d": [200, history] });
  renderApp("/serving/chat");
  await screen.findByRole("heading", { name: "Running/waiting" });
  expect(screen.getByRole("button", { name: "Live" })).toHaveAttribute("aria-pressed", "true");

  await userEvent.click(screen.getByRole("button", { name: "7d" }));

  expect(screen.getByRole("button", { name: "7d" })).toHaveAttribute("aria-pressed", "true");
  expect(calls.map((call) => call.route)).toContain("GET /endpoints/chat/history?span=7d");
});

test("a stopped Endpoint keeps its charts and says live values need it running", async () => {
  fakeApi({
    ...routes,
    "GET /endpoints/chat/stats": [409, { detail: "Endpoint chat is stopped" }],
  });
  renderApp("/serving/chat");

  expect(
    await screen.findByText("Live values appear while the Endpoint runs."),
  ).toBeInTheDocument();
  expect(await screen.findByRole("heading", { name: "Running/waiting" })).toBeInTheDocument();
  expect(screen.queryByRole("meter")).not.toBeInTheDocument();
});

test("an unreachable Endpoint says why it has no live values", async () => {
  fakeApi({
    ...routes,
    "GET /endpoints/chat/stats": [502, { detail: "Endpoint chat did not answer" }],
  });
  renderApp("/serving/chat");

  expect(await screen.findByText("Endpoint chat did not answer")).toBeInTheDocument();
});

test("the stats page shows the running Endpoint's URL and key to copy", async () => {
  const user = userEvent.setup();
  const url = "https://platform.test/serving/0123456789abcdef0123456789abcdef/v1";
  const chat = { name: "chat", status: "running", url, key: "key-of-chat" } as Endpoint;
  const stopped = { ...chat, status: "stopped", url: "https://platform.test/serving/old/v1" };
  fakeApi({ ...routes, "GET /endpoints": [200, [chat, stopped]] });
  renderApp("/serving/chat");

  const here = `${window.location.origin}/serving/0123456789abcdef0123456789abcdef/v1`;
  expect(await screen.findByText(here)).toBeInTheDocument();
  await user.click(screen.getByRole("button", { name: "Copy URL" }));
  expect(await navigator.clipboard.readText()).toBe(here);
  expect(screen.getByText("key-of-chat")).toBeInTheDocument();
  await user.click(screen.getByRole("button", { name: "Copy Endpoint Key" }));
  expect(await navigator.clipboard.readText()).toBe("key-of-chat");
});

const row = (id: string, overrides: Partial<RequestTrace> = {}): RequestTrace => ({
  trace_id: id,
  arrived_at: "2026-10-02T09:00:00Z",
  latency: 1.5,
  end_to_end: 1.4,
  time_to_first_token: 0.25,
  queue_time: 0.125,
  prefill_time: 0.0625,
  decode_time: 1,
  prompt_tokens: 12,
  completion_tokens: 40,
  temperature: null,
  top_p: null,
  max_tokens: null,
  source: "api",
  status: "ok",
  rating: null,
  judge_score: null,
  prompt_preview: "What's the weather in Rome?",
  mlflow_url: `/mlflow/#/experiments/7/traces/${id}`,
  ...overrides,
});
const page = (rows: RequestTrace[], next: string | null = null): EndpointRequests => ({
  requests: rows,
  next_page_token: next,
});
const table = () => screen.getByRole("heading", { name: "All requests" }).closest("section")!;

test("the requests list shows every request with its source, status, timings and feedback", async () => {
  fakeApi({
    ...routes,
    "GET /endpoints/chat/requests?order=time": [
      200,
      page([row("tr-1", { source: "chat", rating: "up", judge_score: 4 }), row("tr-2")]),
    ],
  });
  renderApp("/serving/chat");

  const found = (await screen.findAllByText("What's the weather in Rome?"))[0].closest("tr")!;
  for (const text of ["chat", "ok", "1.50 s", "250 ms", "12 → 40", "4"]) {
    expect(within(found).getByText(text)).toBeInTheDocument();
  }
  expect(within(found).getByLabelText("up")).toBeInTheDocument();
});

test("filters and the sort toggle ask the API again, and load more appends the next page", async () => {
  const user = userEvent.setup();
  const calls = fakeApi({
    ...routes,
    "GET /endpoints/chat/requests?order=time": [200, page([row("tr-1")], "50")],
    "GET /endpoints/chat/requests?order=time&page_token=50": [200, page([row("tr-2")])],
    "GET /endpoints/chat/requests?source=chat&order=time": [200, page([])],
    "GET /endpoints/chat/requests?source=chat&rating=none&order=time": [200, page([])],
    "GET /endpoints/chat/requests?source=chat&rating=none&order=latency": [200, page([])],
  });
  renderApp("/serving/chat");

  await user.click(await within(table()).findByRole("button", { name: "Load more" }));
  expect(await within(table()).findAllByRole("row")).toHaveLength(3);

  await user.selectOptions(screen.getByLabelText("Source"), "chat");
  await user.selectOptions(screen.getByLabelText("Feedback"), "none");
  await user.click(screen.getByRole("button", { name: "Newest first" }));

  expect(await screen.findByText("No requests match.")).toBeInTheDocument();
  expect(calls.map((call) => call.route)).toContain(
    "GET /endpoints/chat/requests?source=chat&rating=none&order=latency",
  );
});

test("a clicked request opens a drawer with its chat, reasoning, answer and judge score", async () => {
  const user = userEvent.setup();
  const detail: RequestDetail = {
    ...row("tr-1", { rating: "down", judge_score: 2 }),
    messages: [
      { role: "system", content: "Be brief." },
      { role: "user", content: "What's the weather in Rome?" },
    ],
    tools: null,
    reasoning: "Look it up.",
    answer: "Sunny.",
    tool_calls: null,
    error: null,
    judge_reason: "Made up.",
  };
  fakeApi({
    ...routes,
    "GET /endpoints/chat/requests?order=time": [200, page([row("tr-1")])],
    "GET /endpoints/chat/requests/tr-1": [200, detail],
  });
  renderApp("/serving/chat");

  await user.click(await within(table()).findByText("What's the weather in Rome?"));

  const drawer = await screen.findByRole("complementary", { name: "Request" });
  expect(await within(drawer).findByText("Be brief.")).toBeInTheDocument();
  expect(within(drawer).getByText("Reasoning")).toBeInTheDocument();
  expect(within(drawer).getByText("Sunny.")).toBeInTheDocument();
  expect(within(drawer).getByText(/Made up\./)).toBeInTheDocument();
  expect(within(drawer).getByRole("link", { name: /Open in MLflow/ })).toHaveAttribute(
    "href",
    "/mlflow/#/experiments/7/traces/tr-1",
  );
  await user.click(within(drawer).getByRole("button", { name: "Close" }));
  expect(screen.queryByRole("complementary", { name: "Request" })).not.toBeInTheDocument();
});

test("the feedback chart shows the votes per bucket", async () => {
  fakeApi({ ...routes });
  renderApp("/serving/chat");

  const votes = await screen.findByRole("heading", { name: "👍/👎" });
  expect(within(votes.closest("figure")!).getByText("👍").closest("div")).toHaveTextContent("3");
});

test("the average judge score chart shows the scores per bucket", async () => {
  fakeApi({ ...routes });
  renderApp("/serving/chat");

  const scores = await screen.findByRole("heading", { name: "Average judge score" });
  expect(within(scores.closest("figure")!).getByText("Score").closest("div")).toHaveTextContent(
    "3.5 / 5",
  );
});
