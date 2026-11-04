import { Gavel } from "lucide-react";
import { useState } from "react";
import { callApi, errorMessage } from "../api";
import { endpointJudging } from "../apiPaths";
import type { Endpoint, Judging } from "./endpoint";

/** An Endpoint's opt-in to judging: on or off, every Nth answer, at most so many an hour. */
export default function JudgingMenu({
  endpoint,
  onSaved,
}: {
  endpoint: Endpoint;
  onSaved: (text: string, failed: boolean) => void;
}) {
  const [judging, setJudging] = useState<Judging>(endpoint.judging);
  const [open, setOpen] = useState(false);
  const number = (key: "every" | "max_per_hour") => (event: { target: { value: string } }) =>
    setJudging({ ...judging, [key]: Number(event.target.value) });

  async function save() {
    try {
      await callApi(endpointJudging(endpoint.name), judging, "PUT");
      const text = judging.judged
        ? `The Judge scores 1 in ${judging.every} answers of ${endpoint.name}`
        : `${endpoint.name} is no longer judged`;
      onSaved(text, false);
    } catch (failure) {
      onSaved(errorMessage(failure), true);
    }
  }

  return (
    // Built only while open, so the row's buttons stay its own; named, as only one fits the spot.
    <details
      name="judging-menu"
      className="judging-menu"
      onToggle={(event) => setOpen(event.currentTarget.open)}
    >
      <summary className={`button ghost small ${endpoint.judging.judged ? "judged" : ""}`}>
        <Gavel size={14} /> Judge
      </summary>
      {open && (
        <div className="card judging-popover">
          <label>
            <input
              type="checkbox"
              checked={judging.judged}
              onChange={(event) => setJudging({ ...judging, judged: event.target.checked })}
            />{" "}
            Judge {endpoint.name}'s answers
          </label>
          <label>
            Every Nth answer
            <input type="number" min={1} value={judging.every} onChange={number("every")} />
          </label>
          <label>
            At most per hour
            <input
              type="number"
              min={1}
              value={judging.max_per_hour}
              onChange={number("max_per_hour")}
            />
          </label>
          <button type="button" className="button primary small" onClick={save}>
            Save
          </button>
        </div>
      )}
    </details>
  );
}
