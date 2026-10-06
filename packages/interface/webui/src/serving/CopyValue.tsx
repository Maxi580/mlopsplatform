import { Check, Copy } from "lucide-react";
import { useState } from "react";

/** The value in monospace, with a button copying it to the clipboard. */
export default function CopyValue({ label, value }: { label: string; value: string }) {
  const [copied, setCopied] = useState(false);
  return (
    <span className="copy-value">
      <span className="mono">{value}</span>
      <button
        type="button"
        className="icon-button"
        aria-label={`Copy ${label}`}
        title={`Copy ${label}`}
        onClick={async () => {
          await navigator.clipboard.writeText(value);
          setCopied(true);
        }}
      >
        {copied ? <Check size={14} /> : <Copy size={14} />}
      </button>
    </span>
  );
}
