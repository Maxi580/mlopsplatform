import { Upload } from "lucide-react";
import { type FormEvent, useState } from "react";
import { errorMessage } from "../api";
import { uploadModel } from "./modelUpload";

// Browsers pick a whole directory with this attribute, which React's types don't know.
const pickDirectory = { webkitdirectory: "" } as object;

/** Uploads a model directory as the next version of a Registered Model, then reports it. */
export default function ModelUploadForm({
  onDone,
}: {
  onDone: (text: string, failed: boolean) => void;
}) {
  const [name, setName] = useState("");
  const [base, setBase] = useState("");
  const [toolParser, setToolParser] = useState("");
  const [files, setFiles] = useState<File[]>([]);
  const [progress, setProgress] = useState<string>();

  async function submit(event: FormEvent) {
    event.preventDefault();
    try {
      const uploaded = await uploadModel(name, base, toolParser, files, setProgress);
      onDone(`Uploaded ${uploaded.name}@${uploaded.version}`, false);
    } catch (failure) {
      onDone(errorMessage(failure), true);
    }
    setProgress(undefined);
  }

  return (
    <form className="card upload-form" onSubmit={submit}>
      <div className="field-grid">
        <div className="field">
          <label className="field-label" htmlFor="upload-name">
            Name
          </label>
          <input
            id="upload-name"
            required
            placeholder="my-model"
            value={name}
            onChange={(event) => setName(event.target.value)}
          />
        </div>
        <div className="field">
          <label className="field-label" htmlFor="upload-base">
            Base <code>only for an Adapter</code>
          </label>
          <input
            id="upload-base"
            placeholder="hf:org/name or model:name@version"
            value={base}
            onChange={(event) => setBase(event.target.value)}
          />
        </div>
        <div className="field">
          <label className="field-label" htmlFor="upload-tool-parser">
            Tool parser <code>only for a model_type the platform doesn't know</code>
          </label>
          <input
            id="upload-tool-parser"
            placeholder="e.g. hermes"
            value={toolParser}
            onChange={(event) => setToolParser(event.target.value)}
          />
        </div>
        <div className="field">
          <label className="field-label" htmlFor="upload-directory">
            Model directory
          </label>
          <input
            id="upload-directory"
            type="file"
            multiple
            {...pickDirectory}
            onChange={(event) => setFiles([...(event.target.files ?? [])])}
          />
        </div>
      </div>
      <div className="upload-actions">
        <button
          type="submit"
          className="button"
          disabled={progress !== undefined || files.length === 0}
        >
          <Upload size={14} /> Upload
        </button>
        {progress && <span className="muted">{progress}</span>}
      </div>
    </form>
  );
}
