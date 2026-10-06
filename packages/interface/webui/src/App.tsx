import { useEffect, useRef } from "react";
import { Navigate, Route, Routes, useLocation, useNavigate } from "react-router-dom";
import { LOGGED_OUT_EVENT } from "./api";
import LoginPage from "./login/LoginPage";
import NewPipelinePage from "./pipelines/NewPipelinePage";
import PipelinesPage from "./pipelines/PipelinesPage";
import EndpointStatsPage from "./serving/EndpointStatsPage";
import ServingPage from "./serving/ServingPage";
import Shell from "./shell/Shell";
import StoragePage from "./storage/StoragePage";

export default function App() {
  const navigate = useNavigate();
  // Login returns to the page the session expired on.
  const location = useLocation();
  const current = useRef(location);
  current.current = location;
  useEffect(() => {
    const toLogin = () => {
      const { pathname, search } = current.current;
      if (pathname !== "/login") navigate("/login", { state: { from: pathname + search } });
    };
    window.addEventListener(LOGGED_OUT_EVENT, toLogin);
    return () => window.removeEventListener(LOGGED_OUT_EVENT, toLogin);
  }, [navigate]);

  return (
    <Routes>
      <Route path="/login" element={<LoginPage />} />
      <Route element={<Shell />}>
        <Route index element={<PipelinesPage />} />
        <Route path="/pipelines/new" element={<NewPipelinePage />} />
        <Route path="/storage" element={<StoragePage />} />
        <Route path="/serving" element={<ServingPage />} />
        <Route path="/serving/:name" element={<EndpointStatsPage />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Route>
    </Routes>
  );
}
