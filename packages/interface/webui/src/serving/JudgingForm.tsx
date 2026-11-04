import { type ComponentProps, useState } from "react";
import { callApi, errorMessage } from "../api";
import { endpointJudging } from "../apiPaths";
import type { Endpoint, Judging } from "./endpoint";

/** An Endpoint's opt-in to judging; built only while open, so closing discards its edits. */
export default function JudgingForm({
  endpoint,
  onSaved,
  onClose,
}: {
  endpoint: Endpoint;
  onSaved: (text: string, failed: boolean) => Promise<void>;
  onClose: () => void;
}) {
  const [judging, setJudging] = useState<Judging>(endpoint.judging);
  const onNumberChange =
    (key: "every" | "max_per_hour") => (event: { target: { value: string } }) =>
      setJudging({ ...judging, [key]: Number(event.target.value) });

  async function save() {
    try {
      await callApi(endpointJudging(endpoint.name), judging, "PUT");
      const text = judging.judged
        ? `The Judge scores 1 in ${judging.every} answers of ${endpoint.name}`
        : `${endpoint.name} is no longer judged`;
      // Closes once the list has what was saved, as reopening starts from the list.
      await onSaved(text, false);
      onClose();
    } catch (failure) {
      await onSaved(errorMessage(failure), true);
    }
  }

  return (
    <div className="judging-form">
      <h3>Judge Configuration:</h3>
      <JudgingSetting
        title="Judge this Endpoint"
        explanation="The Judge scores its answers from 1 to 5."
        type="checkbox"
        checked={judging.judged}
        onChange={(event) => setJudging({ ...judging, judged: event.target.checked })}
      />
      <JudgingSetting
        title="Every Nth answer"
        explanation="Scores 1 answer in every N, counted from when judging was turned on."
        type="number"
        min={1}
        value={judging.every}
        onChange={onNumberChange("every")}
      />
      <JudgingSetting
        title="At most per hour"
        explanation="Answers over this limit wait and are scored in a later hour."
        type="number"
        min={1}
        value={judging.max_per_hour}
        onChange={onNumberChange("max_per_hour")}
      />
      <button type="button" className="button primary small judging-save" onClick={save}>
        Save
      </button>
    </div>
  );
}

function JudgingSetting({
  title,
  explanation,
  ...input
}: { title: string; explanation: string } & ComponentProps<"input">) {
  return (
    <label className="judging-setting">
      <span>
        <strong>{title}</strong>
        <span className="muted">{explanation}</span>
      </span>
      <input {...input} />
    </label>
  );
}
