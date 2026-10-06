import { LoaderCircle, LogIn } from "lucide-react";
import { type FormEvent, useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { callApi, errorMessage } from "../api";
import { LOGIN } from "../apiPaths";
import Logo from "../shell/Logo";

export default function LoginPage() {
  const navigate = useNavigate();
  const from: string = useLocation().state?.from ?? "/";
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  // The API sets the session cookie, which also opens the KFP and MLflow UIs for 12 hours.
  async function logIn(event: FormEvent) {
    event.preventDefault();
    setBusy(true);
    try {
      await callApi(LOGIN, { password });
      navigate(from, { replace: true });
    } catch (failure) {
      setError(errorMessage(failure));
      setBusy(false);
    }
  }

  return (
    <main className="login">
      <form className="login-card" onSubmit={logIn}>
        <Logo />
        <h1>Welcome back</h1>
        <p className="muted">Log in with the team's shared password.</p>
        <label className="field">
          <span className="field-label">Password</span>
          <input
            type="password"
            autoFocus
            required
            value={password}
            onChange={(event) => setPassword(event.target.value)}
            aria-invalid={!!error}
          />
        </label>
        {error && (
          <p className="error" role="alert">
            {error}
          </p>
        )}
        <button type="submit" className="button primary wide" disabled={busy}>
          {busy ? <LoaderCircle className="spin" size={16} /> : <LogIn size={16} />}
          Log in
        </button>
      </form>
    </main>
  );
}
