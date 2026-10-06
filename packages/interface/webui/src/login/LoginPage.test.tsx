import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { fakeApi, renderApp } from "../testApi";

test("logging in goes to the Pipelines page", async () => {
  const calls = fakeApi({
    "POST /auth/login": [200, { token: "t" }],
    "GET /pipelines": [200, []],
    "GET /settings": [200, { gpu_count: 1 }],
  });
  renderApp("/login");

  await userEvent.type(screen.getByLabelText("Password"), "correct horse");
  await userEvent.click(screen.getByRole("button", { name: /log in/i }));

  expect(await screen.findByRole("heading", { name: "Pipelines" })).toBeInTheDocument();
  expect(calls[0]).toEqual({ route: "POST /auth/login", body: { password: "correct horse" } });
});

test("a wrong password is shown on the login page", async () => {
  fakeApi({ "POST /auth/login": [401, { detail: "Wrong password" }] });
  renderApp("/login");

  await userEvent.type(screen.getByLabelText("Password"), "nope");
  await userEvent.click(screen.getByRole("button", { name: /log in/i }));

  expect(await screen.findByRole("alert")).toHaveTextContent("Wrong password");
});

test("an expired session sends the user to the login page", async () => {
  const notLoggedIn: [number, unknown] = [401, { detail: "Not logged in" }];
  fakeApi({ "GET /pipelines": notLoggedIn, "GET /settings": notLoggedIn });
  renderApp("/");

  expect(await screen.findByRole("heading", { name: "Welcome back" })).toBeInTheDocument();
});

// Answers 401 to every listed GET until the login succeeded.
function platform(loggedIn = false) {
  const page: [number, unknown] = [200, []];
  const routes = {
    "GET /pipelines": page,
    "GET /settings": [200, { gpu_count: 1 }] as [number, unknown],
    "GET /auth/verify": [200, { name: "shared" }] as [number, unknown],
    "GET /storage": [200, { buckets: [] }] as [number, unknown],
    "GET /datasets": page,
    "GET /models": page,
    "GET /checkpoints": page,
    "GET /cache": [200, { entries: [] }] as [number, unknown],
  };
  const guarded = Object.fromEntries(
    Object.entries(routes).map(([route, answer]) => [
      route,
      () => (loggedIn ? answer : ([401, { detail: "Not logged in" }] as [number, unknown])),
    ]),
  );
  return fakeApi({
    ...guarded,
    "POST /auth/login": () => {
      loggedIn = true;
      return [200, { token: "t" }];
    },
    "POST /auth/logout": () => {
      loggedIn = false;
      return [200, {}];
    },
  });
}

async function logIn() {
  await userEvent.type(await screen.findByLabelText("Password"), "correct horse");
  await userEvent.click(screen.getByRole("button", { name: /log in/i }));
}

test("an unknown URL lands on Pipelines", async () => {
  platform(true);
  renderApp("/asdgf");

  expect(await screen.findByRole("heading", { name: "Pipelines" })).toBeInTheDocument();
});

test("logged out, an unknown URL lands on login, then Pipelines", async () => {
  platform();
  renderApp("/asdgf");

  await logIn();

  expect(await screen.findByRole("heading", { name: "Pipelines" })).toBeInTheDocument();
});

test("logged out, a real page lands on login, then back on that page", async () => {
  platform();
  renderApp("/storage");

  await logIn();

  expect(await screen.findByRole("heading", { name: "Storage" })).toBeInTheDocument();
});

test("the sidebar shows the user, and Log out ends the session", async () => {
  const calls = platform(true);
  renderApp("/");

  expect(await screen.findByText("shared")).toBeInTheDocument();
  await userEvent.click(screen.getByRole("button", { name: /log out/i }));

  expect(await screen.findByRole("heading", { name: "Welcome back" })).toBeInTheDocument();
  expect(calls.map((call) => call.route)).toContain("POST /auth/logout");
});
