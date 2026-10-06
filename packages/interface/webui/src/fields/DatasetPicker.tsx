import { useState } from "react";
import DatasetUploadDialog from "../storage/DatasetUploadDialog";
import Combobox, { type ChoiceGroup } from "./Combobox";
import type { Catalog } from "./catalog";

type Props = {
  id: string;
  label: string;
  value: string;
  // The References the field takes: `dataset:`, and `@distill` while `distill` is on.
  references: string[];
  // The row formats the field reads; the Datasets offered and uploads are held to them.
  rowFormats: string[];
  catalog: Catalog;
  invalid?: boolean;
  describedBy?: string;
  onChange: (value: string) => void;
};

/** A Dataset field's picker: this Pipeline's Distillation Dataset and the Datasets by name,
 * each meaning its latest Version, and an upload of a new one. */
export default function DatasetPicker({ references, rowFormats, catalog, ...props }: Props) {
  const [uploading, setUploading] = useState(false);
  const groups: ChoiceGroup[] = [];
  if (references.some((reference) => reference.startsWith("@")))
    groups.push({
      label: "This Pipeline",
      choices: references
        .filter((reference) => reference.startsWith("@"))
        .map((output) => ({ value: output, label: "Distillation Dataset from this Pipeline" })),
    });
  groups.push({
    label: "Datasets",
    choices: catalog.datasets.flatMap((dataset) => {
      const latest = dataset.versions.at(-1);
      const readable = !rowFormats.length || rowFormats.includes(latest?.row_format ?? "");
      return latest && readable
        ? [{ value: `dataset:${dataset.name}`, label: dataset.name, detail: latest.row_format }]
        : [];
    }),
  });

  return (
    <>
      <Combobox
        {...props}
        groups={groups}
        placeholder="Search Datasets…"
        action={{ label: "Upload…", run: () => setUploading(true) }}
      />
      {uploading && (
        <DatasetUploadDialog
          datasets={catalog.datasets}
          rowFormats={rowFormats}
          onClose={() => setUploading(false)}
          onUploaded={(name) => {
            setUploading(false);
            props.onChange(`dataset:${name}`);
            catalog.reloadDatasets?.();
          }}
        />
      )}
    </>
  );
}
