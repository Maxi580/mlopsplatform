import { useCallback, useEffect, useState } from "react";
import { LOGIN } from "./apiPaths";

// Fired on any 401, so the app can send the user to the login page.
export const LOGGED_OUT_EVENT = "mlp:logged-out";

export class ApiError extends Error {
  constructor(
    readonly status: number,
    readonly detail: unknown,
  ) {
    super(typeof detail === "string" ? detail : `The API answered ${status}`);
  }
}

/** The API's JSON answer; ApiError with FastAPI's `detail` when it refused. A file is sent as
 * it is, anything else as JSON. */
export async function callApi<T>(
  path: string,
  body?: unknown,
  method = body === undefined ? "GET" : "POST",
  headers: Record<string, string> = {},
): Promise<T> {
  const json = body !== undefined && !(body instanceof Blob);
  const response = await fetch(path, {
    method,
    headers: json ? { ...headers, "content-type": "application/json" } : headers,
    body: json ? JSON.stringify(body) : (body as Blob | undefined),
    credentials: "same-origin",
  });
  if (response.status === 401 && path !== LOGIN) window.dispatchEvent(new Event(LOGGED_OUT_EVENT));
  const data = await response.json().catch(() => null);
  if (!response.ok) throw new ApiError(response.status, data?.detail ?? response.statusText);
  return data as T;
}

export function errorMessage(failure: unknown): string {
  return failure instanceof Error ? failure.message : String(failure);
}

/** The answer to a GET, reloaded every `refreshMs` if given. */
export function useApi<T>(path: string, refreshMs?: number) {
  const [data, setData] = useState<T>();
  const [error, setError] = useState<ApiError>();
  const [loadedAt, setLoadedAt] = useState<Date>();
  const reload = useCallback(
    () =>
      callApi<T>(path).then(
        (answer) => {
          setData(answer);
          setError(undefined);
          setLoadedAt(new Date());
        },
        (failure) => setError(failure),
      ),
    [path],
  );
  useEffect(() => {
    reload();
    if (!refreshMs) return;
    const timer = setInterval(reload, refreshMs);
    return () => clearInterval(timer);
  }, [reload, refreshMs]);
  return { data, error, loadedAt, reload };
}
