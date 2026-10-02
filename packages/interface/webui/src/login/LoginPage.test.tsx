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
