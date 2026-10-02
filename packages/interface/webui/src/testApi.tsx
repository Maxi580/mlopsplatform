import { render } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import App from "./App";

type Answer = [status: number, body: unknown];
type Handler = (body: any) => Answer;

/** Answers `fetch` from `routes` ("POST /pipelines" -> answer) and records every call. */
export function fakeApi(routes: Record<string, Handler | Answer>) {
  const calls: { route: string; body: any }[] = [];
  vi.stubGlobal("fetch", async (path: string, init: RequestInit) => {
    const route = `${init.method} ${path}`;
    // JSON for the API; anything else, such as a part of an uploaded file, as it is.
    const body = typeof init.body === "string" ? JSON.parse(init.body) : init.body;
    calls.push({ route, body });
    const handler = routes[route] ?? [404, { detail: "Not Found" }];
    const [status, answer] = typeof handler === "function" ? handler(body) : handler;
    return new Response(status === 204 ? null : JSON.stringify(answer), { status });
  });
  return calls;
}

export function renderApp(path: string) {
  return render(
    <MemoryRouter initialEntries={[path]}>
      <App />
    </MemoryRouter>,
  );
}
