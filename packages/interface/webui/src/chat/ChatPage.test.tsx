import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import type { Endpoint } from "../serving/endpoint";
import { renderApp } from "../testApi";

const ENDPOINT_PATH = "/serving/0123456789abcdef0123456789abcdef/v1";
const ENDPOINT_URL = `https://platform.test${ENDPOINT_PATH}`;
const endpoint = (name: string, status = "running", judge = false): Endpoint => ({
  name,
  owner: "shared",
  model: "hf:Qwen/Qwen3-0.6B@abc",
  status,
  url: ENDPOINT_URL,
  key: `key-of-${name}`,
  created_at: "2026-10-02T08:30:00+00:00",
  stats: null,
  judge,
  judging: { judged: false, every: 10, max_per_hour: 60 },
});
const ENDPOINTS = [
  endpoint("chat"),
  endpoint("old", "stopped"),
  endpoint("judge", "running", true),
  endpoint("other"),
];

const chunk = (delta: object) => `data: ${JSON.stringify({ choices: [{ index: 0, delta }] })}\n\n`;

/** Answers the Web UI's requests: the Endpoint list, and the chat streamed in the parts given;
 * records every call. */
function fakeEndpoint(parts: string[]) {
  const calls: { url: string; init: RequestInit }[] = [];
  vi.stubGlobal("fetch", async (url: string, init: RequestInit) => {
    calls.push({ url, init });
    if (url === "/endpoints") return new Response(JSON.stringify(ENDPOINTS));
    if (url === `${ENDPOINT_PATH}/chat/completions`) {
      const encoder = new TextEncoder();
      const stream = new ReadableStream({
        start(controller) {
          for (const part of parts) controller.enqueue(encoder.encode(part));
          controller.close();
        },
      });
      return new Response(stream, { headers: { "x-mlp-trace-id": "tr-1" } });
    }
    if (url === `${ENDPOINT_PATH}/feedback`) return new Response(init.body as string);
    return new Response(JSON.stringify({ detail: "Not Found" }), { status: 404 });
  });
  return calls;
}

async function pick(user: ReturnType<typeof userEvent.setup>, name: string) {
  const picker = await screen.findByRole("combobox", { name: "Endpoint" });
  await screen.findByRole("option", { name });
  await user.selectOptions(picker, name);
}

test("the picker offers only running Endpoints, not the Judge", async () => {
  fakeEndpoint([]);
  renderApp("/chat");

  await screen.findByRole("option", { name: "chat" });
  const options = screen.getAllByRole("option").map((option) => option.textContent);
  expect(options).toEqual(["Pick a running Endpoint", "chat", "other"]);
});

test("an answer streams into its message, with its reasoning folded above it", async () => {
  const user = userEvent.setup();
  const calls = fakeEndpoint([
    chunk({ role: "assistant", content: "" }),
    chunk({ reasoning_content: "Greet " }) + chunk({ reasoning: "back." }),
    chunk({ content: "Hel" }),
    `${chunk({ content: "lo!" })}data: [DONE]\n\n`,
  ]);
  renderApp("/chat");
  await pick(user, "chat");

  await user.type(screen.getByRole("textbox", { name: "Message" }), "Say hi.");
  await user.click(screen.getByRole("button", { name: /Send/ }));

  const chat = screen.getByRole("region", { name: "Chat" });
  expect(await within(chat).findByText("Hello!")).toBeInTheDocument();
  expect(within(chat).getByText("Reasoning")).toBeInTheDocument();
  expect(within(chat).getByText("Greet back.")).toBeInTheDocument();
  const sent = calls.find((call) => call.url === `${ENDPOINT_PATH}/chat/completions`)!;
  const headers = sent.init.headers as Record<string, string>;
  expect(headers.authorization).toBe("Bearer key-of-chat");
  expect(headers["x-mlp-source"]).toBe("chat");
  expect(JSON.parse(sent.init.body as string)).toEqual({
    model: "chat",
    stream: true,
    messages: [{ role: "user", content: "Say hi." }],
    temperature: 0.7,
    max_tokens: 1024,
  });
});

test("a failed send puts the message back, so the next send leaves no empty answer", async () => {
  const user = userEvent.setup();
  const calls = fakeEndpoint([chunk({ content: "Hi!" })]);
  const answering = globalThis.fetch;
  vi.stubGlobal("fetch", async (url: string, init: RequestInit) => {
    if (url !== `${ENDPOINT_PATH}/chat/completions`) return answering(url, init);
    vi.stubGlobal("fetch", answering);
    throw new TypeError("Failed to fetch");
  });
  renderApp("/chat");
  await pick(user, "chat");
  const message = screen.getByRole("textbox", { name: "Message" });
  await user.type(message, "Say hi.");
  await user.click(screen.getByRole("button", { name: /Send/ }));

  expect(await screen.findByText("Failed to fetch")).toBeInTheDocument();
  expect(message).toHaveValue("Say hi.");
  const chat = screen.getByRole("region", { name: "Chat" });
  expect(within(chat).queryByText("assistant")).not.toBeInTheDocument();

  await user.click(screen.getByRole("button", { name: /Send/ }));
  await within(chat).findByText("Hi!");
  const sent = calls.find((call) => call.url === `${ENDPOINT_PATH}/chat/completions`)!;
  expect(JSON.parse(sent.init.body as string).messages).toEqual([
    { role: "user", content: "Say hi." },
  ]);
});

test("a thumb posts Feedback for the answer's trace and shows the choice", async () => {
  const user = userEvent.setup();
  const calls = fakeEndpoint([chunk({ content: "Hi!" })]);
  renderApp("/chat");
  await pick(user, "chat");
  await user.type(screen.getByRole("textbox", { name: "Message" }), "Say hi.");
  await user.click(screen.getByRole("button", { name: /Send/ }));

  const good = await screen.findByRole("button", { name: "Good answer" });
  await waitFor(() => expect(good).toBeEnabled());
  await user.click(good);

  await waitFor(() => expect(good).toHaveAttribute("aria-pressed", "true"));
  const feedback = calls.find((call) => call.url === `${ENDPOINT_PATH}/feedback`)!;
  expect(JSON.parse(feedback.init.body as string)).toEqual({ trace_id: "tr-1", rating: "up" });
  expect((feedback.init.headers as Record<string, string>).authorization).toBe(
    "Bearer key-of-chat",
  );
});

test("a thumb can be changed, and a Feedback that can't be sent says so", async () => {
  const user = userEvent.setup();
  const calls = fakeEndpoint([chunk({ content: "Hi!" })]);
  renderApp("/chat");
  await pick(user, "chat");
  await user.type(screen.getByRole("textbox", { name: "Message" }), "Say hi.");
  await user.click(screen.getByRole("button", { name: /Send/ }));
  const good = await screen.findByRole("button", { name: "Good answer" });
  const bad = screen.getByRole("button", { name: "Bad answer" });
  await waitFor(() => expect(good).toBeEnabled());

  await user.click(good);
  await waitFor(() => expect(good).toHaveAttribute("aria-pressed", "true"));
  await user.click(bad);
  await waitFor(() => expect(bad).toHaveAttribute("aria-pressed", "true"));
  expect(good).toHaveAttribute("aria-pressed", "false");
  const sent = calls.filter((call) => call.url === `${ENDPOINT_PATH}/feedback`);
  expect(JSON.parse(sent[1].init.body as string).rating).toBe("down");

  vi.stubGlobal("fetch", async () => {
    throw new TypeError("Failed to fetch");
  });
  await user.click(good);
  expect(await screen.findByText("Failed to fetch")).toBeInTheDocument();
  expect(good).toHaveAttribute("aria-pressed", "false");
});

test("a new chat or another Endpoint clears the history", async () => {
  const user = userEvent.setup();
  fakeEndpoint([chunk({ content: "Hi!" })]);
  renderApp("/chat");
  await pick(user, "chat");
  await user.type(screen.getByRole("textbox", { name: "Message" }), "Say hi.");
  await user.click(screen.getByRole("button", { name: /Send/ }));
  await screen.findByText("Hi!");

  await user.selectOptions(screen.getByRole("combobox", { name: "Endpoint" }), "other");

  expect(screen.queryByText("Hi!")).not.toBeInTheDocument();
});
