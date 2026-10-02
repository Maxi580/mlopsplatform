import { ChartLine, Cpu, ExternalLink, HardDrive, Plus, Server, Workflow } from "lucide-react";
import { NavLink, Outlet } from "react-router-dom";
import { useApi } from "../api";
import { SETTINGS } from "../apiPaths";
import { KUBEFLOW_UI_URL, MLFLOW_UI_URL } from "../config";
import Logo from "./Logo";

export default function Shell() {
  const settings = useApi<{ gpu_count: number }>(SETTINGS).data;

  return (
    <div className="shell">
      <aside className="sidebar">
        <Logo />
        <nav>
          <NavLink to="/" end>
            <Workflow size={18} /> Pipelines
          </NavLink>
          <NavLink to="/pipelines/new">
            <Plus size={18} /> New Pipeline
          </NavLink>
          <NavLink to="/storage">
            <HardDrive size={18} /> Storage
          </NavLink>
          <NavLink to="/serving">
            <Server size={18} /> Serving
          </NavLink>
          <p className="nav-heading">Tools</p>
          <a href={KUBEFLOW_UI_URL} target="_blank" rel="noreferrer">
            <Workflow size={18} /> Kubeflow <ExternalLink className="trailing" size={14} />
          </a>
          <a href={MLFLOW_UI_URL} target="_blank" rel="noreferrer">
            <ChartLine size={18} /> MLflow <ExternalLink className="trailing" size={14} />
          </a>
        </nav>
        {settings && (
          <div className="gpu-card" title="Stages and Endpoints wait while every GPU is busy">
            <Cpu size={18} />
            <div>
              <strong>
                {settings.gpu_count} GPU{settings.gpu_count === 1 ? "" : "s"}
              </strong>
              <span>on this platform</span>
            </div>
          </div>
        )}
      </aside>
      <main className="content">
        <Outlet />
      </main>
    </div>
  );
}
