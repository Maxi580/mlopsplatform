import { act, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { ENDPOINT_STATS_REFRESH_MS } from "../config";
import { fakeApi, renderApp } from "../testApi";
import type { Endpoint, EndpointStats, Histogram } from "./endpoint";

const ttftHistogram = (sum: number, count: number): Histogram => ({
  p50: 0.05,
  p95: 0.2,
  mean: sum / count,
  sum,
  count,
});
const value = (key: string, label: string, explanation: string, reading: number) => ({
  key,
  label,
  explanation,
  value: reading,
});

function reading(at: string, tokens: number, ttft: Histogram): EndpointStats {
  return {
    read_at: at,
    sections: [
      {
        title: "Requests",
        values: [
          value("running", "Running", "Requests the GPU is generating tokens for.", 2),
          value("waiting", "Waiting", "Requests queued.", 0),
        ],
      },
      {
        title: "Tokens",
        values: [
          value("prompt_tokens", "Prompt tokens", "Tokens of all prompts.", tokens * 2),
          value("generation_tokens", "Generated tokens", "Tokens written.", tokens),
        ],
      },
      {
        title: "KV cache",
        values: [
          value("kv_cache_usage", "KV cache usage", "How full it is; not the hit rate.", 0.42),
          value("prefix_cache_hit_rate", "Prefix-cache hit rate", "Share found.", 0.25),
          value("prefix_cache_queries", "Prefix-cache lookups", "Looked up.", tokens * 2),
          value("prefix_cache_hits", "Prefix-cache hits", "Found.", tokens),
        ],
      },
      {
        title: "Latency",
        values: [
          {
            key: "time_to_first_token",
            label: "Time to first token",
            explanation: "Seconds to the first token.",
            histogram: ttft,
          },
        ],
      },
    ],
    metrics: [
      {
        name: "vllm:num_requests_running",
        type: "gauge",
        help: "Number of requests in model execution batches.",
        value: 2,
      },
      {
        name: "vllm:request_prompt_tokens",
        type: "histogram",
        help: "Number of prefill tokens processed.",
        histogram: { p50: 12, p95: 30, mean: 14.5, sum: 145, count: 10 },
      },
    ],
  };
}

const first = reading("2026-10-02T09:00:00Z", 1000, ttftHistogram(1, 10));
const second = reading("2026-10-02T09:00:05Z", 1400, ttftHistogram(1.6, 12));

test("the stats page shows the KV cache bar, labelled values and every metric", async () => {
  fakeApi({ "GET /endpoints/chat/stats": [200, first] });
  renderApp("/serving/chat");

  const bar = await screen.findByRole("meter", { name: "KV cache usage" });
  expect(bar).toHaveAttribute("value", "42");
  expect(bar.closest("section")).toHaveTextContent("How full it is; not the hit rate.");

  const latency = screen.getByRole("heading", { name: "Latency" }).closest("section")!;
  const ttft = within(latency).getByText("Time to first token").parentElement!;
  expect(ttft).toHaveTextContent("p50 50 ms · p95 200 ms · mean 100 ms");
  expect(ttft).toHaveTextContent("Seconds to the first token.");
  expect(
    screen.getByText("Prefix-cache hit rate", { selector: "dt" }).parentElement!,
  ).toHaveTextContent("25.0 %");

  const metric = screen.getByText("vllm:request_prompt_tokens").closest("tr")!;
  expect(metric).toHaveTextContent("p50 12 · p95 30 · mean 14.5");
  expect(within(metric).getByText("Number of prefill tokens processed.")).toBeInTheDocument();
});

test("the charts start empty and show the current value once two readings arrived", async () => {
  vi.useFakeTimers({ shouldAdvanceTime: true });
  const readings = [first, second];
  fakeApi({ "GET /endpoints/chat/stats": () => [200, readings.shift() ?? second] });
  renderApp("/serving/chat");
  const tokens = (await screen.findByText("Tokens/s")).closest("figure")!;
  expect(within(tokens).getByText("Generated").closest("div")).toHaveTextContent("—");

  await act(() => vi.advanceTimersByTimeAsync(ENDPOINT_STATS_REFRESH_MS));

  expect(within(tokens).getByText("Generated").closest("div")).toHaveTextContent("80.0/s");
  const latencyChart = screen.getByText("Latency (mean per interval)").closest("figure")!;
  // (1.6 - 1) seconds over (12 - 10) requests.
  expect(within(latencyChart).getByText("Time to first token").closest("div")).toHaveTextContent(
    "300 ms",
  );
  const hitRate = screen.getByText("Prefix-cache hit rate", { selector: "h3" }).closest("figure")!;
  expect(hitRate).toHaveTextContent("50.0 %");
  vi.useRealTimers();
});

test("an Endpoint without stats says why", async () => {
  fakeApi({
    "GET /endpoints/chat/stats": [409, { detail: "Endpoint chat is pending" }],
  });
  renderApp("/serving/chat");

  expect(await screen.findByText("Endpoint chat is pending")).toBeInTheDocument();
});

test("the stats page shows the running Endpoint's URL to copy", async () => {
  const user = userEvent.setup();
  const url = "https://platform.test/serving/0123456789abcdef0123456789abcdef/v1";
  const chat = { name: "chat", status: "running", url } as Endpoint;
  const stopped = { ...chat, status: "stopped", url: "https://platform.test/serving/old/v1" };
  fakeApi({
    "GET /endpoints/chat/stats": [200, first],
    "GET /endpoints": [200, [chat, stopped]],
  });
  renderApp("/serving/chat");

  expect(await screen.findByText(url)).toBeInTheDocument();
  await user.click(screen.getByRole("button", { name: "Copy URL" }));
  expect(await navigator.clipboard.readText()).toBe(url);
});
