import { ChevronDown, ChevronUp } from "lucide-react";
import type { InputHTMLAttributes } from "react";
import { type Bounds, stepNumber } from "./numbers";

type Props = Omit<InputHTMLAttributes<HTMLInputElement>, "onChange" | "value"> & {
  value: string;
  // What an empty field steps from, e.g. its default.
  from?: string;
  bounds?: Bounds;
  integer?: boolean;
  label: string;
  onChange: (value: string) => void;
};

/** A number field: typed freely, or stepped by ▲/▼ and the arrow keys in its smallest place shown. */
export default function NumberInput({
  value,
  from,
  bounds,
  integer,
  label,
  onChange,
  ...input
}: Props) {
  const step = (by: 1 | -1) => onChange(stepNumber(value || from || "0", by, bounds, integer));

  return (
    <div className="number-input">
      <input
        {...input}
        value={value}
        inputMode="decimal"
        autoComplete="off"
        onChange={(event) => onChange(event.target.value)}
        onKeyDown={(event) => {
          if (event.key !== "ArrowUp" && event.key !== "ArrowDown") return;
          event.preventDefault();
          step(event.key === "ArrowUp" ? 1 : -1);
        }}
      />
      <span className="steppers">
        {/* The arrow keys step from the keyboard, so tabbing skips these. */}
        <button
          type="button"
          tabIndex={-1}
          disabled={input.disabled}
          aria-label={`Increase ${label}`}
          onClick={() => step(1)}
        >
          <ChevronUp size={12} />
        </button>
        <button
          type="button"
          tabIndex={-1}
          disabled={input.disabled}
          aria-label={`Decrease ${label}`}
          onClick={() => step(-1)}
        >
          <ChevronDown size={12} />
        </button>
      </span>
    </div>
  );
}
