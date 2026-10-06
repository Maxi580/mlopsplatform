import { Check, Copy } from "lucide-react";
import { useState } from "react";

/** The value in monospace, unless hidden, with a button copying it to the clipboard. */
export default function CopyValue({
  label,
  value,
  hidden = false,
}: {
  label: string;
  value: string;
  hidden?: boolean;
}) {
  const [copied, setCopied] = useState(false);
  return (
    <span className="copy-value">
      {!hidden && (
        <span className="mono" title={value}>
          {value}
        </span>
      )}
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
