import { CircleAlert, ExternalLink, LoaderCircle, ThumbsDown, ThumbsUp, X } from "lucide-react";
import { useEffect, useState } from "react";
import { callApi, errorMessage, useApi } from "../api";
import { endpointRequest, endpointRequests } from "../apiPaths";
import {
  ENDPOINT_HISTORY_RETENTION_DAYS,
  JUDGE_SCORES,
  REQUEST_SOURCES,
  REQUEST_STATUSES,
} from "../config";
import {
  type EndpointRequests,
  formatSeconds,
  type Message,
  type RequestDetail,
  type RequestTrace,
} from "./endpoint";

type Filters = Record<string, string>;
type Order = "time" | "latency";

/** Every request to the Endpoint, filtered, newest or slowest first, 50 at a time; a click on
 * one opens its full content in a drawer. */
export default function RequestsList({ name }: { name: string }) {
  const [filters, setFilters] = useState<Filters>({});
  const [order, setOrder] = useState<Order>("time");
  const [rows, setRows] = useState<RequestTrace[]>();
  const [next, setNext] = useState<string | null>(null);
  const [error, setError] = useState<string>();
  const [open, setOpen] = useState<string>();

  async function load(pageToken: string | null) {
    try {
      const page = await callApi<EndpointRequests>(
        endpointRequests(name, query({ ...filters, order }, pageToken)),
      );
      setRows((shown) => [...(pageToken ? (shown ?? []) : []), ...page.requests]);
      setNext(page.next_page_token);
      setError(undefined);
    } catch (failure) {
      setError(errorMessage(failure));
    }
  }
  // biome-ignore lint/correctness/useExhaustiveDependencies: a new filter or order starts over.
  useEffect(() => {
    setRows(undefined);
    load(null);
  }, [name, filters, order]);

  const filter = (key: string) => (value: string) => setFilters({ ...filters, [key]: value });
  return (
    <section className="storage-section">
      <h2>All requests</h2>
      <p className="muted">
        Every request through the Endpoint's URL, with its full content, kept{" "}
        {ENDPOINT_HISTORY_RETENTION_DAYS} days.
      </p>
      <div className="request-filters">
        <FilterInput label="From" type="datetime-local" onChange={filter("from")} />
        <FilterInput label="To" type="datetime-local" onChange={filter("to")} />
        <FilterSelect label="Source" choices={REQUEST_SOURCES} onChange={filter("source")} />
        <FilterSelect label="Status" choices={REQUEST_STATUSES} onChange={filter("status")} />
        <FilterSelect
          label="Feedback"
          choices={["up", "down", "none"]}
          onChange={filter("rating")}
        />
        <FilterInput label="Score from" type="number" onChange={filter("score_min")} />
        <FilterInput label="Score to" type="number" onChange={filter("score_max")} />
        <FilterInput label="Min latency (ms)" type="number" onChange={filter("min_latency_ms")} />
        <button
          type="button"
          className="button small ghost"
          onClick={() => setOrder(order === "time" ? "latency" : "time")}
        >
          {order === "time" ? "Newest first" : "Slowest first"}
        </button>
      </div>
      {error && (
        <p className="banner danger">
          <CircleAlert size={16} /> {error}
        </p>
      )}
      {!rows && !error && <LoaderCircle className="spin" size={16} />}
      {rows?.length === 0 && <p className="muted">No requests match.</p>}
      {!!rows?.length && (
        <div className="card table-card">
          <table className="requests-table">
            <thead>
              <tr>
                {["Time", "Source", "Status", "Latency", "TTFT", "Tokens", "👍/👎", "Score"].map(
                  (column) => (
                    <th key={column}>{column}</th>
                  ),
                )}
                <th>Prompt</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => (
                <tr key={row.trace_id} className="clickable" onClick={() => setOpen(row.trace_id)}>
                  <td title={row.arrived_at}>{arrivedText(row.arrived_at)}</td>
                  <td>{row.source ?? "—"}</td>
                  <td>{row.status ?? "—"}</td>
                  <td>{formatSeconds(row.latency)}</td>
                  <td>{formatSeconds(row.time_to_first_token)}</td>
                  <td>{tokensText(row)}</td>
                  <td>
                    <Rating rating={row.rating} />
                  </td>
                  <td>{row.judge_score ?? "—"}</td>
                  <td className="cut-off">{row.prompt_preview ?? "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      {next && (
        <button type="button" className="button ghost small" onClick={() => load(next)}>
          Load more
        </button>
      )}
      {open && <RequestDrawer name={name} traceId={open} onClose={() => setOpen(undefined)} />}
    </section>
  );
}

// The set filters as the API's query; a picked local time is sent in UTC.
function query(filters: Filters, pageToken: string | null): string {
  const set = Object.entries(filters).filter(([, value]) => value !== "");
  const params = new URLSearchParams(
    set.map(([key, value]) => [
      key,
      key === "from" || key === "to" ? new Date(value).toISOString() : value,
    ]),
  );
  if (pageToken) params.set("page_token", pageToken);
  return params.toString();
}

function FilterInput({
  label,
  type,
  onChange,
}: {
  label: string;
  type: string;
  onChange: (value: string) => void;
}) {
  const [min, max] = label.startsWith("Score") ? JUDGE_SCORES : [0, undefined];
  return (
    <label className="request-filter">
      <span>{label}</span>
      <input type={type} min={min} max={max} onChange={(event) => onChange(event.target.value)} />
    </label>
  );
}

function FilterSelect({
  label,
  choices,
  onChange,
}: {
  label: string;
  choices: string[];
  onChange: (value: string) => void;
}) {
  return (
    <label className="request-filter">
      <span>{label}</span>
      <select onChange={(event) => onChange(event.target.value)}>
        <option value="">any</option>
        {choices.map((choice) => (
          <option key={choice}>{choice}</option>
        ))}
      </select>
    </label>
  );
}

function Rating({ rating }: { rating: string | null }) {
  if (rating === "up") return <ThumbsUp size={14} aria-label="up" />;
  if (rating === "down") return <ThumbsDown size={14} aria-label="down" />;
  return <>—</>;
}

/** The request's messages as a chat, its reasoning folded, the answer and tool calls, then its
 * timings, Feedback and Judge Score. */
function RequestDrawer({
  name,
  traceId,
  onClose,
}: {
  name: string;
  traceId: string;
  onClose: () => void;
}) {
  const { data, error } = useApi<RequestDetail>(endpointRequest(name, traceId));
  return (
    <aside className="drawer" aria-label="Request">
      <header className="drawer-header">
        <h2>Request</h2>
        <button type="button" className="icon-button" aria-label="Close" onClick={onClose}>
          <X size={16} />
        </button>
      </header>
      {error && <p className="banner danger">{error.message}</p>}
      {!data && !error && <LoaderCircle className="spin" size={16} />}
      {data && (
        <>
          {data.messages.map((message, index) => (
            // Messages have no IDs, and their order never changes.
            // biome-ignore lint/suspicious/noArrayIndexKey: see above.
            <ChatMessage key={index} message={message} />
          ))}
          {data.reasoning && (
            <details className="reasoning">
              <summary>Reasoning</summary>
              <pre>{data.reasoning}</pre>
            </details>
          )}
          <ChatMessage message={{ role: "assistant", content: data.answer ?? "" }} />
          {!!data.tool_calls?.length && <pre>{JSON.stringify(data.tool_calls, null, 2)}</pre>}
          {data.error && <p className="banner danger">{data.error}</p>}
          <dl className="drawer-facts">
            <dt>Latency</dt>
            <dd>{formatSeconds(data.latency)}</dd>
            <dt>Queue · prefill · decode</dt>
            <dd>
              {[data.queue_time, data.prefill_time, data.decode_time]
                .map(formatSeconds)
                .join(" · ")}
            </dd>
            <dt>Tokens</dt>
            <dd>{tokensText(data)}</dd>
            <dt>Feedback</dt>
            <dd>
              <Rating rating={data.rating} />
            </dd>
            <dt>Judge Score</dt>
            <dd>
              {data.judge_score ?? "—"}
              {data.judge_reason && <span className="muted"> · {data.judge_reason}</span>}
            </dd>
          </dl>
          <a href={data.mlflow_url} target="_blank" rel="noreferrer">
            Open in MLflow <ExternalLink size={14} />
          </a>
        </>
      )}
    </aside>
  );
}

function ChatMessage({ message }: { message: Message }) {
  const content =
    typeof message.content === "string" ? message.content : JSON.stringify(message.content);
  return (
    <div className={`chat-message ${message.role}`}>
      <span className="muted">{message.role}</span>
      <pre>{content}</pre>
      {!!message.tool_calls?.length && <pre>{JSON.stringify(message.tool_calls, null, 2)}</pre>}
    </div>
  );
}

function tokensText(row: RequestTrace): string {
  if (row.prompt_tokens == null) return "—";
  return `${row.prompt_tokens} → ${row.completion_tokens ?? "—"}`;
}

function arrivedText(arrivedAt: string): string {
  return new Date(arrivedAt).toLocaleString("en", {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}
