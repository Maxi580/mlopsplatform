import { X } from "lucide-react";
import { type ReactNode, useState } from "react";
import { matchedParts, matchesWords } from "./fuzzy";

export type Choice = { value: string; label: string; detail?: string; badge?: ReactNode };
// `searched` choices were already found for what is typed, e.g. by the Hub, so they aren't
// narrowed again; an `action` is offered under the choices, e.g. an upload.
export type ChoiceGroup = { label: string; choices: Choice[]; searched?: boolean };
export type ComboboxAction = { label: string; run: () => void };

type Props = {
  id: string;
  label: string;
  value: string;
  groups: ChoiceGroup[];
  placeholder?: string;
  invalid?: boolean;
  describedBy?: string;
  action?: ComboboxAction;
  onChange: (value: string) => void;
  // What the user types, e.g. to search the Hub with.
  onType?: (typed: string) => void;
};

/** A text field with a list of grouped choices, narrowed by every word typed; picking a choice
 * writes its value, the keyboard alone can pick one, and × empties the field. */
export default function Combobox({
  id,
  label,
  value,
  groups,
  placeholder,
  invalid,
  describedBy,
  action,
  onChange,
  onType,
}: Props) {
  const [open, setOpen] = useState(false);
  // What the user typed since the last pick; the list narrows by it.
  const [typed, setTyped] = useState("");
  const [active, setActive] = useState(0);
  const shown = groups
    .map((group) => ({
      ...group,
      choices: group.searched
        ? group.choices
        : group.choices.filter((choice) => matchesWords(`${choice.label} ${choice.value}`, typed)),
    }))
    .filter((group) => group.choices.length);
  const choices = shown.flatMap((group) => group.choices);
  const listbox = `${id}-choices`;
  const optionId = (index: number) => `${id}-choice-${index}`;

  function type(text: string) {
    onChange(text);
    setTyped(text);
    setActive(0);
    setOpen(true);
    onType?.(text);
  }

  function pick(choice: Choice) {
    onChange(choice.value);
    setTyped("");
    setOpen(false);
  }

  return (
    <div className="combobox">
      <input
        id={id}
        role="combobox"
        value={value}
        placeholder={placeholder}
        autoComplete="off"
        aria-invalid={invalid}
        aria-describedby={describedBy}
        aria-expanded={open}
        aria-controls={listbox}
        aria-autocomplete="list"
        aria-activedescendant={open && choices.length ? optionId(active) : undefined}
        onChange={(event) => type(event.target.value)}
        onFocus={() => setOpen(true)}
        onBlur={() => setOpen(false)}
        onKeyDown={(event) => {
          if (event.key === "ArrowDown" || event.key === "ArrowUp") {
            event.preventDefault();
            setOpen(true);
            const by = event.key === "ArrowDown" ? 1 : -1;
            setActive((active + by + choices.length) % Math.max(choices.length, 1));
          } else if (event.key === "Enter" && open && choices[active]) {
            event.preventDefault();
            pick(choices[active]);
          } else if (event.key === "Escape") {
            setOpen(false);
          }
        }}
      />
      {value && (
        <button
          type="button"
          className="combobox-clear"
          aria-label={`Clear ${label}`}
          onClick={() => type("")}
        >
          <X size={14} />
        </button>
      )}
      {open && (shown.length > 0 || action) && (
        <div className="combobox-list">
          <div id={listbox} role="listbox" aria-label={label}>
            {shown.map((group) => (
              // biome-ignore lint/a11y/useSemanticElements: a listbox groups its options with role="group", not a fieldset.
              <div key={group.label} role="group" aria-label={group.label}>
                <div className="combobox-group" aria-hidden="true">
                  {group.label}
                </div>
                {group.choices.map((choice) => {
                  const index = choices.indexOf(choice);
                  return (
                    <div
                      key={choice.value}
                      id={optionId(index)}
                      role="option"
                      // Focus stays in the field, which points at the active option.
                      tabIndex={-1}
                      aria-selected={index === active}
                      className="combobox-choice"
                      // Picked before the field loses focus and closes the list.
                      onMouseDown={(event) => {
                        event.preventDefault();
                        pick(choice);
                      }}
                      onMouseEnter={() => setActive(index)}
                    >
                      <span className="choice-label">
                        {matchedParts(choice.label, typed).map((part, at) =>
                          part.matched ? (
                            // biome-ignore lint/suspicious/noArrayIndexKey: parts have no identity.
                            <mark key={at}>{part.text}</mark>
                          ) : (
                            part.text
                          ),
                        )}
                      </span>
                      {choice.badge}
                      {choice.detail && <span className="muted">{choice.detail}</span>}
                    </div>
                  );
                })}
              </div>
            ))}
          </div>
          {action && (
            <button
              type="button"
              className="combobox-action"
              onMouseDown={(event) => {
                event.preventDefault();
                setOpen(false);
                action.run();
              }}
            >
              {action.label}
            </button>
          )}
        </div>
      )}
    </div>
  );
}
