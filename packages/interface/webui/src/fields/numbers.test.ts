import { formatNumber, stepNumber } from "./numbers";

test.each([
  ["0.05", 1, "0.06"],
  ["2e-4", 1, "3e-4"],
  ["16", 1, "17"],
  ["3", 1, "4"],
  ["0.06", -1, "0.05"],
  ["2.5e-4", 1, "2.6e-4"],
  ["9e-5", 1, "1e-4"],
])("a step changes the smallest decimal place shown by 1: %s by %i is %s", (text, by, stepped) => {
  expect(stepNumber(text, by as 1 | -1)).toBe(stepped);
});

test.each([
  ["0.01", "0.009"],
  ["1e-4", "9e-5"],
  ["0.1", "0.09"],
  ["1", "0.9"],
])("stepping down from a leading 1 moves into the next place down: %s is %s", (text, stepped) => {
  expect(stepNumber(text, -1)).toBe(stepped);
});

test("an integer steps by whole numbers, also down from 1", () => {
  expect(stepNumber("1", -1, {}, true)).toBe("0");
  expect(stepNumber("10", -1, {}, true)).toBe("9");
});

test("the schema's bounds are never crossed", () => {
  // gt 0: a step that would reach the bound keeps the value.
  expect(stepNumber("1", -1, { exclusiveMinimum: 0 }, true)).toBe("1");
  // ge 0 and le 1: a step stops at the bound.
  expect(stepNumber("0", -1, { minimum: 0 })).toBe("0");
  expect(stepNumber("1", 1, { maximum: 1 })).toBe("1");
  expect(stepNumber("0.95", 1, { maximum: 1 })).toBe("0.96");
  expect(stepNumber("0.9", 1, { exclusiveMaximum: 1 })).toBe("0.9");
});

test("text that is no number is left as it is", () => {
  expect(stepNumber("long", 1)).toBe("long");
});

test("values below 0.001 show in scientific notation", () => {
  expect(formatNumber(0.0002)).toBe("2e-4");
  expect(formatNumber(0.002)).toBe("0.002");
  expect(formatNumber(0)).toBe("0");
});
