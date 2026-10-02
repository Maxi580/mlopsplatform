import { Ban, CircleCheck, CircleX, Clock, Hourglass, LoaderCircle } from "lucide-react";

const ICONS: Record<string, typeof Clock> = {
  pending: Clock,
  running: LoaderCircle,
  "waiting for GPU": Hourglass,
  succeeded: CircleCheck,
  failed: CircleX,
  cancelled: Ban,
};

export default function StatusBadge({ status }: { status: string }) {
  const Icon = ICONS[status] ?? Clock;
  return (
    <span className={`badge status-${status.replaceAll(" ", "-")}`}>
      <Icon size={14} className={status === "running" ? "spin" : undefined} />
      {status}
    </span>
  );
}
