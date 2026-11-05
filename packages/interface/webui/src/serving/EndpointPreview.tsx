import { PREVIEW_ENDPOINT } from "../apiPaths";
import { formatShare } from "../fields/numbers";
import { usePreview } from "../usePreview";

// What a start would pin for the model and options.
type Pinned = { gpu_memory_utilization: number; max_model_len: number | null };

/** The GPU Share and length the spec would run on; nothing until it names a servable model. */
export default function EndpointPreview({ spec }: { spec?: Record<string, unknown> }) {
  const pinned = usePreview<Pinned>(PREVIEW_ENDPOINT, spec);
  return pinned ? <p className="muted">{pinnedSummary(pinned)}</p> : null;
}

function pinnedSummary(pinned: Pinned): string {
  return [
    `GPU Share ${formatShare(pinned.gpu_memory_utilization)}`,
    ...(pinned.max_model_len ? [`up to ${pinned.max_model_len} tokens`] : []),
  ].join(", ");
}
