import { CircleAlert, LoaderCircle, Plus, Send, ThumbsDown, ThumbsUp } from "lucide-react";
import { useState } from "react";
import { errorMessage, useApi } from "../api";
import { ENDPOINTS } from "../apiPaths";
import {
  CHAT_DEFAULTS,
  CHAT_SOURCE,
  ENDPOINT_LIST_REFRESH_MS,
  SOURCE_HEADER,
  TRACE_ID_HEADER,
} from "../config";
import type { Endpoint } from "../serving/endpoint";

type Rating = "up" | "down";
type Turn = {
  role: "user" | "assistant";
  content: string;
  reasoning?: string;
  // The answer's Request Trace, which Feedback refers to.
  traceId?: string;
  rating?: Rating;
};
type Settings = { system: string; temperature: string; max_tokens: string };

/** A basic chat with a running Endpoint, through its URL and key, so it is traced like any
 * client's; the history lives in this page only. */
export default function ChatPage() {
  const endpoints = useApi<Endpoint[]>(ENDPOINTS, ENDPOINT_LIST_REFRESH_MS).data;
  const running = (endpoints ?? []).filter((found) => found.status === "running" && !found.judge);
  const [name, setName] = useState("");
  const [settings, setSettings] = useState<Settings>(CHAT_DEFAULTS);
  const [turns, setTurns] = useState<Turn[]>([]);
  const [draft, setDraft] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string>();
  const endpoint = running.find((found) => found.name === name);

  async function send() {
    if (!endpoint || !draft.trim()) return;
    const history: Turn[] = [...turns, { role: "user", content: draft }];
    setTurns([...history, { role: "assistant", content: "" }]);
    setDraft("");
    setBusy(true);
    setError(undefined);
    try {
      await streamAnswer(endpoint, settings, history, (answer) => setTurns([...history, answer]));
    } catch (failure) {
      setError(errorMessage(failure));
    }
    setBusy(false);
  }

  async function rate(index: number, rating: Rating) {
    const turn = turns[index];
    if (!endpoint || !turn.traceId) return;
    const response = await fetch(`${endpoint.url}/feedback`, {
      method: "POST",
      headers: { authorization: `Bearer ${endpoint.key}`, "content-type": "application/json" },
      body: JSON.stringify({ trace_id: turn.traceId, rating }),
    });
    if (!response.ok) return setError(`The Feedback was refused (${response.status})`);
    setTurns(turns.map((found, at) => (at === index ? { ...found, rating } : found)));
  }

  return (
    <>
      <header className="page-header">
        <div>
          <h1>Chat</h1>
          <p className="muted">
            Try a running Endpoint; the chat goes through its URL, so it is traced with source chat
            and judged like any request, and is gone once you leave.
          </p>
        </div>
      </header>

      <div className="chat-controls">
        <label className="request-filter">
          <span>Endpoint</span>
          <select
            value={name}
            onChange={(event) => {
              setName(event.target.value);
              setTurns([]);
            }}
          >
            <option value="">Pick a running Endpoint</option>
            {running.map((found) => (
              <option key={found.name}>{found.name}</option>
            ))}
          </select>
        </label>
        <button type="button" className="button ghost small" onClick={() => setTurns([])}>
          <Plus size={14} /> New chat
        </button>
      </div>
      <details className="judge-advanced">
        <summary>Settings</summary>
        <div className="chat-settings">
          <label className="request-filter">
            <span>System prompt</span>
            <textarea
              rows={2}
              value={settings.system}
              onChange={(event) => setSettings({ ...settings, system: event.target.value })}
            />
          </label>
          <label className="request-filter">
            <span>Temperature</span>
            <input
              type="number"
              step="0.1"
              min={0}
              value={settings.temperature}
              onChange={(event) => setSettings({ ...settings, temperature: event.target.value })}
            />
          </label>
          <label className="request-filter">
            <span>Max tokens</span>
            <input
              type="number"
              min={1}
              value={settings.max_tokens}
              onChange={(event) => setSettings({ ...settings, max_tokens: event.target.value })}
            />
          </label>
        </div>
      </details>

      <section className="chat-turns" aria-label="Chat">
        {turns.map((turn, index) => (
          // Turns have no IDs, and their order never changes.
          // biome-ignore lint/suspicious/noArrayIndexKey: see above.
          <div key={index} className={`chat-message ${turn.role}`}>
            <span className="muted">{turn.role}</span>
            {turn.reasoning && (
              <details className="reasoning">
                <summary>Reasoning</summary>
                <pre>{turn.reasoning}</pre>
              </details>
            )}
            <ChatText text={turn.content} />
            {turn.traceId && (
              <span className="chat-rating">
                {(["up", "down"] as const).map((rating) => (
                  <button
                    key={rating}
                    type="button"
                    className={`icon-button ${turn.rating === rating ? "chosen" : ""}`}
                    aria-label={rating === "up" ? "Good answer" : "Bad answer"}
                    aria-pressed={turn.rating === rating}
                    onClick={() => rate(index, rating)}
                  >
                    {rating === "up" ? <ThumbsUp size={14} /> : <ThumbsDown size={14} />}
                  </button>
                ))}
              </span>
            )}
          </div>
        ))}
      </section>
      {error && (
        <p className="banner danger">
          <CircleAlert size={16} /> {error}
        </p>
      )}
      <form
        className="chat-input"
        onSubmit={(event) => {
          event.preventDefault();
          send();
        }}
      >
        <textarea
          aria-label="Message"
          rows={3}
          value={draft}
          disabled={!endpoint}
          onChange={(event) => setDraft(event.target.value)}
        />
        <button type="submit" className="button primary" disabled={!endpoint || busy}>
          {busy ? <LoaderCircle className="spin" size={16} /> : <Send size={16} />} Send
        </button>
      </form>
    </>
  );
}

/** Streams the Endpoint's answer to the history, handing each update of it to `onAnswer`. */
async function streamAnswer(
  endpoint: Endpoint,
  settings: Settings,
  history: Turn[],
  onAnswer: (answer: Turn) => void,
) {
  // 1. The request, with the settings that are set.
  const system = settings.system.trim() ? [{ role: "system", content: settings.system }] : [];
  const response = await fetch(`${endpoint.url}/chat/completions`, {
    method: "POST",
    headers: {
      authorization: `Bearer ${endpoint.key}`,
      "content-type": "application/json",
      [SOURCE_HEADER]: CHAT_SOURCE,
    },
    body: JSON.stringify({
      model: endpoint.name,
      stream: true,
      messages: [...system, ...history.map(({ role, content }) => ({ role, content }))],
      ...(settings.temperature && { temperature: Number(settings.temperature) }),
      ...(settings.max_tokens && { max_tokens: Number(settings.max_tokens) }),
    }),
  });
  if (!response.ok || !response.body) {
    throw new Error(`The Endpoint answered ${response.status}: ${await response.text()}`);
  }

  // 2. Its server-sent events, each adding to the answer and its reasoning.
  const answer: Turn = { role: "assistant", content: "", reasoning: "" };
  answer.traceId = response.headers.get(TRACE_ID_HEADER) ?? undefined;
  const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
  let pending = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    pending += value;
    const lines = pending.split("\n");
    pending = lines.pop() ?? "";
    for (const line of lines) {
      const data = line.startsWith("data:") ? line.slice("data:".length).trim() : "";
      if (!data || data === "[DONE]") continue;
      const delta = JSON.parse(data).choices?.[0]?.delta ?? {};
      answer.content += delta.content ?? "";
      // vLLM names it `reasoning_content`, newer ones `reasoning`.
      answer.reasoning += delta.reasoning_content ?? delta.reasoning ?? "";
    }
    onAnswer({ ...answer });
  }
}

// Plain text, with ``` fences shown as code blocks.
function ChatText({ text }: { text: string }) {
  return (
    <>
      {text.split("```").map((part, index) =>
        index % 2 ? (
          // biome-ignore lint/suspicious/noArrayIndexKey: the parts of one text, in order.
          <pre key={index} className="chat-code">
            <code>{part.replace(/^\w*\n/, "")}</code>
          </pre>
        ) : (
          // biome-ignore lint/suspicious/noArrayIndexKey: the parts of one text, in order.
          <pre key={index}>{part}</pre>
        ),
      )}
    </>
  );
}
