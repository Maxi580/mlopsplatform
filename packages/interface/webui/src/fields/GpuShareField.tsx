import { useApi } from "../api";
import { SCHEMA } from "../apiPaths";
import GpuShareNote from "./GpuShareNote";
import NumberInput from "./NumberInput";

type Schema = {
  properties?: Record<string, { description?: string }>;
  $defs?: { ServingOptions?: { properties?: Record<string, { description?: string }> } };
};

/** A new GPU Share for a Restart or Resume: empty keeps `stored`; says what it means below it. */
export default function GpuShareField({
  id,
  label,
  name,
  stored,
  value,
  onChange,
}: {
  id: string;
  label: string;
  // What the API calls it, which also says what holds the share.
  name: "gpu_share" | "gpu_memory_utilization";
  stored: number;
  value: string;
  onChange: (value: string) => void;
}) {
  const schema = useApi<Schema>(SCHEMA).data;
  const hint =
    name === "gpu_share"
      ? schema?.properties?.gpu_share?.description
      : schema?.$defs?.ServingOptions?.properties?.gpu_memory_utilization?.description;
  return (
    <div className="field">
      <label className="field-label" htmlFor={id}>
        {label} <code>{name}</code>
      </label>
      <NumberInput
        id={id}
        label={label}
        value={value}
        from={String(stored)}
        placeholder={String(stored)}
        bounds={{ exclusiveMinimum: 0, maximum: 1 }}
        aria-describedby={`${id}-info`}
        onChange={onChange}
      />
      <GpuShareNote id={`${id}-info`} hint={hint} />
    </div>
  );
}
