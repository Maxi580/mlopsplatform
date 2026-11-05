import { Eye, EyeOff } from "lucide-react";
import { useState } from "react";

/** A Secret's value, hidden until asked; it travels beside a Pipeline Request, never inside it. */
export default function SecretInput(props: {
  slot: string;
  label: string;
  value: string;
  onChange: (value: string) => void;
}) {
  const [visible, setVisible] = useState(false);
  return (
    <div className="field">
      <label className="field-label" htmlFor={props.slot}>
        {props.label} <code>{props.slot}</code>
      </label>
      <div className="input-with-button">
        <input
          id={props.slot}
          type={visible ? "text" : "password"}
          value={props.value}
          autoComplete="off"
          placeholder="optional for public models"
          onChange={(event) => props.onChange(event.target.value)}
        />
        <button
          type="button"
          className="icon-button"
          aria-label={visible ? "Hide" : "Show"}
          onClick={() => setVisible(!visible)}
        >
          {visible ? <EyeOff size={16} /> : <Eye size={16} />}
        </button>
      </div>
    </div>
  );
}
