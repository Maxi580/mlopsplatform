// A number field's bounds, as the schema states them (pydantic's gt, ge, lt and le).
export type Bounds = {
  minimum?: number;
  maximum?: number;
  exclusiveMinimum?: number;
  exclusiveMaximum?: number;
};

/** A number as a form shows it: below 0.001 in scientific notation, e.g. `2e-4`. */
export function formatNumber(value: number): string {
  return value !== 0 && Math.abs(value) < 0.001 ? value.toExponential() : String(value);
}

/** The text one step up or down: its smallest decimal place shown changes by 1 (0.05 → 0.06,
 * 2e-4 → 3e-4), and down from a leading 1 into the next place (0.01 → 0.009), within the bounds. */
export function stepNumber(text: string, by: 1 | -1, bounds: Bounds = {}, integer = false): string {
  // 1. The text as digits times a power of ten, e.g. 2.5e-4 as 25 × 10^-5.
  const match = text.trim().match(/^(-?)(\d*)(?:\.(\d*))?(?:e([+-]?\d+))?$/i);
  if (!match || !(match[2] || match[3])) return text;
  const [, sign, whole, fraction = "", exponent = "0"] = match;
  let digits = Number(`${sign}${whole}${fraction}`);
  let power = Number(exponent) - fraction.length;

  // 2. One step in that place; towards zero from a single 1, a step in the place below.
  if (Math.abs(digits) === 1 && Math.sign(digits) !== by && !(integer && power <= 0)) {
    digits = 9 * digits;
    power -= 1;
  } else {
    digits += by;
  }
  const value = Number(`${digits}e${power}`);

  // 3. Within the bounds: an exclusive one keeps the value, an inclusive one is where it stops.
  const { minimum, maximum, exclusiveMinimum, exclusiveMaximum } = bounds;
  if (exclusiveMinimum !== undefined && value <= exclusiveMinimum) return text;
  if (exclusiveMaximum !== undefined && value >= exclusiveMaximum) return text;
  if (minimum !== undefined && value < minimum) return formatNumber(minimum);
  if (maximum !== undefined && value > maximum) return formatNumber(maximum);
  return power >= 0 || Math.abs(value) < 0.001 ? formatNumber(value) : value.toFixed(-power);
}
