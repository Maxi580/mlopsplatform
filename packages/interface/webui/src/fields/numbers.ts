/** A number as a form shows it: below 0.001 in scientific notation, e.g. `2e-4`. */
export function formatNumber(value: number): string {
  return value !== 0 && Math.abs(value) < 0.001 ? value.toExponential() : String(value);
}
