import { LoaderCircle, Upload } from "lucide-react";
import { type FormEvent, useState } from "react";
import { createPortal } from "react-dom";
import { callApi, errorMessage } from "../api";
import { datasetVersions } from "../apiPaths";
import type { Dataset } from "./storage";

type Props = {
  datasets: Dataset[];
  // The row formats the field reads, which the API checks the file against; any when empty.
  rowFormats?: string[];
  onUploaded: (name: string) => void;
  onClose: () => void;
};

/** Uploads a JSONL file as a Dataset: a new one, or under an existing name its next Version. */
export default function DatasetUploadDialog({
  datasets,
  rowFormats = [],
  onUploaded,
  onClose,
}: Props) {
  const [file, setFile] = useState<File>();
  const [name, setName] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const chosen = name.trim();
  const replaces = datasets.some((dataset) => dataset.name === chosen);

  async function upload(event: FormEvent) {
    event.preventDefault();
    // React carries events out of a portal to the page's form, which would submit too.
    event.stopPropagation();
    if (!file) return;
    setBusy(true);
    setError("");
    const query = rowFormats.length ? `?row_formats=${rowFormats.join(",")}` : "";
    try {
      await callApi(`${datasetVersions(chosen)}${query}`, file);
      onUploaded(chosen);
    } catch (failure) {
      setError(errorMessage(failure));
      setBusy(false);
    }
  }

  // Outside any page form, as forms don't nest.
  return createPortal(
    <div className="dialog-backdrop">
      <form
        className="dialog card"
        role="dialog"
        aria-modal="true"
        aria-labelledby="dataset-upload-title"
        onSubmit={upload}
        onKeyDown={(event) => event.key === "Escape" && onClose()}
      >
        <h2 id="dataset-upload-title">Upload a Dataset</h2>
        <label className="field">
          <span className="field-label">JSONL file</span>
          <input
            type="file"
            accept=".jsonl,.json"
            onChange={(event) => {
              const picked = event.target.files?.[0];
              setFile(picked);
              if (picked) setName(datasetName(picked.name));
            }}
          />
        </label>
        <label className="field">
          <span className="field-label">Name</span>
          <input
            value={name}
            required
            pattern="[A-Za-z0-9][A-Za-z0-9_.\-]*"
            autoComplete="off"
            onChange={(event) => setName(event.target.value)}
          />
        </label>
        {replaces && (
          <p className="muted">
            Replaces <strong>{chosen}</strong> (the old data stays with Pipelines that used it)
          </p>
        )}
        {error && (
          <p className="field-error" role="alert">
            {error}
          </p>
        )}
        <div className="dialog-actions">
          <button type="button" className="button ghost" onClick={onClose}>
            Cancel
          </button>
          <button type="submit" className="button primary" disabled={busy || !file}>
            {busy ? <LoaderCircle className="spin" size={16} /> : <Upload size={16} />}
            Upload
          </button>
        </div>
      </form>
    </div>,
    document.body,
  );
}

// A file's name as a Dataset name: without its extension, other characters as dashes.
function datasetName(fileName: string): string {
  return fileName
    .replace(/\.jsonl?$/i, "")
    .replace(/[^A-Za-z0-9_.-]+/g, "-")
    .replace(/^[^A-Za-z0-9]+/, "");
}
