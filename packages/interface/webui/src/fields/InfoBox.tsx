import { Info } from "lucide-react";
import type { ReactNode } from "react";

/** An ⓘ that shows `text` on hover and on keyboard focus, floating so nothing shifts. */
export default function InfoBox({
  id,
  text,
  label = "About",
}: {
  id: string;
  text: ReactNode;
  label?: string;
}) {
  return (
    <span className="info">
      <button type="button" className="info-button" aria-label={label} aria-describedby={id}>
        <Info size={14} />
      </button>
      <span role="tooltip" id={id} className="info-text">
        {text}
      </span>
    </span>
  );
}
