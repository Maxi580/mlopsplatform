import { CircleAlert, Download, HardDrive } from "lucide-react";
import { type ReactNode, useState } from "react";
import { callApi, errorMessage, useApi } from "../api";
import {
  DATASETS,
  datasetDownload,
  datasetVersion,
  MODELS,
  modelVersion,
  modelVersionFiles,
  STORAGE,
} from "../apiPaths";
import { BUCKET_CONTENTS, STORAGE_WARNING_PERCENT } from "../config";
import ModelUploadForm from "./ModelUploadForm";
import type { Dataset, ModelVersionFile, RegisteredModel, Storage } from "./storage";

export default function StoragePage() {
  const datasets = useApi<Dataset[]>(DATASETS);
  const models = useApi<RegisteredModel[]>(MODELS);
  const storage = useApi<Storage>(STORAGE);
  const [notice, setNotice] = useState<{ text: string; failed: boolean }>();
  const [modelFiles, setModelFiles] = useState<{ label: string; files: ModelVersionFile[] }>();
  const error = datasets.error ?? models.error ?? storage.error;

  function reload() {
    datasets.reload();
    models.reload();
    storage.reload();
  }

  async function remove(path: string, label: string) {
    try {
      await callApi(path, undefined, "DELETE");
      setNotice({ text: `Deleted ${label}`, failed: false });
    } catch (failure) {
      setNotice({ text: errorMessage(failure), failed: true });
    }
    reload();
  }

  // A Model Version has many files, so they are offered as links rather than saved at once.
  async function showModelFiles(name: string, version: number) {
    try {
      const { files } = await callApi<{ files: ModelVersionFile[] }>(
        modelVersionFiles(name, version),
      );
      setModelFiles({ label: `${name}@${version}`, files });
    } catch (failure) {
      setNotice({ text: errorMessage(failure), failed: true });
    }
  }

  async function download(name: string, version: number) {
    try {
      const { url } = await callApi<{ url: string }>(datasetDownload(name, version));
      // A link click, so the browser saves the file under a readable name.
      const link = document.createElement("a");
      link.href = url;
      link.download = `${name}-${version}.jsonl`;
      link.click();
    } catch (failure) {
      setNotice({ text: errorMessage(failure), failed: true });
    }
  }

  return (
    <>
      <header className="page-header">
        <div>
          <h1>Storage</h1>
          <p className="muted">
            Datasets, Registered Models, model uploads and how full the object store is.
          </p>
        </div>
      </header>

      {notice && (
        <p className={`banner ${notice.failed ? "danger" : "success"}`}>
          {notice.failed && <CircleAlert size={16} />} {notice.text}
        </p>
      )}
      {error && (
        <p className="banner danger">
          <CircleAlert size={16} /> {error.message}
        </p>
      )}

      {storage.data && <UsageCard storage={storage.data} />}

      <VersionTable
        title="Datasets"
        columns={["Dataset Version", "Row format", "Size"]}
        empty="No Datasets yet; upload one with `mlp datasets upload`."
        onDelete={remove}
        rows={datasets.data?.flatMap((dataset) =>
          dataset.versions.map((v) => ({
            label: `${dataset.name}@${v.version}`,
            cells: [v.row_format, formatBytes(v.size_bytes)],
            path: datasetVersion(dataset.name, v.version),
            onDownload: () => download(dataset.name, v.version),
          })),
        )}
      />

      <VersionTable
        title="Registered Models"
        columns={["Model Version", "Weights", "Base Model", "Pipeline", "Size"]}
        empty="No Registered Models yet; every finetune Phase registers one."
        onDelete={remove}
        rows={models.data?.flatMap((model) =>
          model.versions.map((v) => ({
            label: `${model.name}@${v.version}`,
            cells: [
              v.tags.weights && (
                <span key="weights" className="chip">
                  {v.tags.weights}
                </span>
              ),
              <span key="base_model" className="mono">
                {v.tags.base_model}
              </span>,
              v.tags.pipeline && `#${v.tags.pipeline}`,
              formatBytes(v.size_bytes),
            ],
            path: modelVersion(model.name, v.version),
            onDownload: () => showModelFiles(model.name, v.version),
          })),
        )}
      />
      {modelFiles && <ModelFilesCard {...modelFiles} />}

      <section className="storage-section">
        <h2>Upload a model</h2>
        <p className="muted">
          Full weights (config.json, tokenizer and *.safetensors), or an Adapter with its base.
        </p>
        <ModelUploadForm
          onDone={(text, failed) => {
            setNotice({ text, failed });
            reload();
          }}
        />
      </section>
    </>
  );
}

function ModelFilesCard({ label, files }: { label: string; files: ModelVersionFile[] }) {
  return (
    <section className="card model-files">
      <strong>Files of {label}</strong>
      <ul className="buckets">
        {files.map((file) => (
          <li key={file.path}>
            <a className="mono" href={file.url} download={file.path.split("/").pop()}>
              {file.path}
            </a>
            <span />
            <span>{formatBytes(file.size_bytes)}</span>
          </li>
        ))}
      </ul>
    </section>
  );
}

function UsageCard({ storage }: { storage: Storage }) {
  const used = storage.buckets.reduce((total, bucket) => total + bucket.size_bytes, 0);
  const percent = Math.round((used / storage.capacity_bytes) * 100);

  return (
    <section className="card usage">
      <div className="usage-heading">
        <HardDrive size={18} />
        <strong>{`${formatBytes(used)} of ${formatBytes(storage.capacity_bytes)} used`}</strong>
        <span className="muted">{percent}%</span>
      </div>
      <meter
        aria-label="Object store usage"
        min={0}
        max={100}
        high={STORAGE_WARNING_PERCENT}
        value={percent}
      />
      <ul className="buckets">
        {storage.buckets.map((bucket) => (
          <li key={bucket.name}>
            <span className="mono">{bucket.name}</span>
            <span className="muted">{BUCKET_CONTENTS[bucket.name]}</span>
            <span>{formatBytes(bucket.size_bytes)}</span>
          </li>
        ))}
      </ul>
    </section>
  );
}

type VersionRowData = {
  label: string;
  cells: ReactNode[];
  path: string;
  onDownload?: () => void;
};

function VersionTable({
  title,
  columns,
  empty,
  rows,
  onDelete,
}: {
  title: string;
  columns: string[];
  empty: string;
  rows?: VersionRowData[];
  onDelete: (path: string, label: string) => void;
}) {
  return (
    <section className="storage-section">
      <h2>{title}</h2>
      {rows?.length === 0 && <p className="muted">{empty}</p>}
      {!!rows?.length && (
        <div className="card table-card">
          <table>
            <thead>
              <tr>
                {columns.map((column) => (
                  <th key={column}>{column}</th>
                ))}
                <th aria-label="Actions" />
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => (
                <VersionRow key={row.label} row={row} onDelete={onDelete} />
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}

function VersionRow({
  row,
  onDelete,
}: {
  row: VersionRowData;
  onDelete: (path: string, label: string) => void;
}) {
  // Deleting asks once more in place, rather than in a browser dialog.
  const [confirming, setConfirming] = useState(false);

  return (
    <tr>
      <td className="pipeline-name">{row.label}</td>
      {row.cells.map((cell, index) => (
        // biome-ignore lint/suspicious/noArrayIndexKey: the columns are fixed per table.
        <td key={index}>{cell}</td>
      ))}
      <td className="actions">
        {row.onDownload && (
          <button type="button" className="button ghost small" onClick={row.onDownload}>
            <Download size={14} /> Download
          </button>
        )}
        {confirming ? (
          <>
            <button
              type="button"
              className="button danger small"
              onClick={() => {
                setConfirming(false);
                onDelete(row.path, row.label);
              }}
            >
              Delete {row.label}
            </button>
            <button
              type="button"
              className="button ghost small"
              onClick={() => setConfirming(false)}
            >
              Keep
            </button>
          </>
        ) : (
          <button type="button" className="button ghost small" onClick={() => setConfirming(true)}>
            Delete
          </button>
        )}
      </td>
    </tr>
  );
}

function formatBytes(bytes: number): string {
  const units = ["B", "KB", "MB", "GB", "TB"];
  let index = 0;
  while (bytes >= 1024 && index < units.length - 1) {
    bytes /= 1024;
    index++;
  }
  return index === 0 ? `${bytes} B` : `${bytes.toFixed(1)} ${units[index]}`;
}
