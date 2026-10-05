import { useEffect } from "react";
import { Route, Routes, useNavigate } from "react-router-dom";
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
  useEffect(() => {
    const toLogin = () => navigate("/login");
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
      </Route>
    </Routes>
  );
}
