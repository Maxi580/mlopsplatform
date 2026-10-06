/** Whether every word typed appears in the text, in any order and case. */
export function matchesWords(text: string, typed: string): boolean {
  const lower = text.toLowerCase();
  return wordsOf(typed).every((word) => lower.includes(word));
}

/** The text in parts, each marked if it is where a typed word appears, for highlighting. */
export function matchedParts(text: string, typed: string): { text: string; matched: boolean }[] {
  const lower = text.toLowerCase();
  const marked = new Array<boolean>(text.length).fill(false);
  for (const word of wordsOf(typed)) {
    for (let at = lower.indexOf(word); at >= 0; at = lower.indexOf(word, at + 1)) {
      marked.fill(true, at, at + word.length);
    }
  }
  const parts: { text: string; matched: boolean }[] = [];
  for (const [index, character] of [...text].entries()) {
    const last = parts.at(-1);
    if (last && last.matched === marked[index]) last.text += character;
    else parts.push({ text: character, matched: marked[index] });
  }
  return parts;
}

function wordsOf(typed: string): string[] {
  return typed.toLowerCase().split(/\s+/).filter(Boolean);
}
