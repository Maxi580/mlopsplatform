import { matchedParts, matchesWords } from "./fuzzy";

test("every typed word must appear in the name, in any order and case", () => {
  expect(matchesWords("Qwen/Qwen2.5-7B-Instruct", "qwen 7b")).toBe(true);
  expect(matchesWords("Qwen/Qwen2.5-7B-Instruct", "7b qwen")).toBe(true);
  expect(matchesWords("meta-llama/Llama-3.1-8B", "qwen 8b")).toBe(false);
  expect(matchesWords("anything", "  ")).toBe(true);
});

test("the parts of a name that match a typed word are marked", () => {
  expect(matchedParts("Qwen2.5-7B", "7b qwen")).toEqual([
    { text: "Qwen", matched: true },
    { text: "2.5-", matched: false },
    { text: "7B", matched: true },
  ]);
});
