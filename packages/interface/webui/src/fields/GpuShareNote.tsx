import { useApi } from "../api";
import { GPU_RESERVED } from "../apiPaths";
import { formatShare } from "./numbers";

// What `GET /gpus/reserved` answers.
type GpuReserved = { reserved_percent: number | null; recommended_max: number };

/** Below a GPU Share field: what it means, and how full the fullest machine is if that's known. */
export default function GpuShareNote({ id, hint }: { id: string; hint?: string }) {
  const reserved = useApi<GpuReserved>(GPU_RESERVED).data;
  return (
    <>
      {hint && (
        <p className="muted field-note" id={id}>
          {hint}
        </p>
      )}
      {reserved && reserved.reserved_percent !== null && (
        <p className="muted field-note">
          {`Currently reserved: ${reserved.reserved_percent}% of this GPU · Recommended max: ${formatShare(reserved.recommended_max)}`}
        </p>
      )}
    </>
  );
}
