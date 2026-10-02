import { callApi } from "../api";
import { MODEL_UPLOADS, modelUploadComplete } from "../apiPaths";
import type { NewModelVersion, StartedModelUpload } from "./storage";

/** The new Model Version, once the directory's files are uploaded in parts and checked. */
export async function uploadModel(
  name: string,
  base: string,
  toolParser: string,
  directory: File[],
  onProgress: (text: string) => void,
): Promise<NewModelVersion> {
  // 1. The files by their path inside the directory, hidden ones such as .git left out.
  const files = new Map(
    directory
      .map((file) => [pathInDirectory(file), file] as const)
      .filter(([path]) => !path.split("/").some((part) => part.startsWith("."))),
  );

  // 2. Started; the API checks the file names and the base before anything travels.
  const started = await callApi<StartedModelUpload>(MODEL_UPLOADS, {
    name,
    files: [...files].map(([path, file]) => ({ path, size_bytes: file.size })),
    base: base || null,
    tool_parser: toolParser || null,
  });

  // 3. Each part straight to the object store, through its presigned URL.
  const size = started.part_size_bytes;
  for (const { path, part_urls } of started.files) {
    const file = files.get(path)!;
    for (const [index, url] of part_urls.entries()) {
      onProgress(`Uploading ${path} (part ${index + 1} of ${part_urls.length})`);
      const part = file.slice(index * size, (index + 1) * size);
      const response = await fetch(url, { method: "PUT", body: part });
      if (!response.ok) throw new Error(`Uploading ${path} failed (${response.status})`);
    }
  }

  // 4. The checks, then the new version.
  onProgress("Checking the files");
  return callApi<NewModelVersion>(modelUploadComplete(started.id), undefined, "POST");
}

// A picked directory names each file "<directory>/<path>".
function pathInDirectory(file: File): string {
  const path = file.webkitRelativePath;
  return path ? path.slice(path.indexOf("/") + 1) : file.name;
}
