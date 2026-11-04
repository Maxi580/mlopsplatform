import { useEffect, useState } from "react";
import { callApi } from "./api";
import { PREVIEW_DELAY_MS } from "./config";

/** The API's answer to the body, asked anew after each change; undefined if none or refused. */
export function usePreview<Answer>(path: string, body: unknown): Answer | undefined {
  const [preview, setPreview] = useState<Answer>();
  const sent = body === undefined ? undefined : JSON.stringify(body);
  useEffect(() => {
    if (sent === undefined) {
      setPreview(undefined);
      return;
    }
    let current = true;
    const timer = setTimeout(() => {
      callApi<Answer>(path, JSON.parse(sent)).then(
        (answer) => current && setPreview(answer),
        () => current && setPreview(undefined),
      );
    }, PREVIEW_DELAY_MS);
    return () => {
      current = false;
      clearTimeout(timer);
    };
  }, [path, sent]);
  return preview;
}
