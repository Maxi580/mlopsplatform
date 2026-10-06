import { Info } from "lucide-react";

/** An ⓘ that shows `text` on hover and on keyboard focus, floating so nothing shifts. */
export default function InfoBox({ id, text }: { id: string; text: string }) {
  return (
    <span className="info">
      <button type="button" className="info-button" aria-label="About" aria-describedby={id}>
        <Info size={14} />
      </button>
      <span role="tooltip" id={id} className="info-text">
        {text}
      </span>
    </span>
  );
}
