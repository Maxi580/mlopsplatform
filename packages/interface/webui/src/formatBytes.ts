export function formatBytes(bytes: number): string {
  const units = ["B", "KB", "MB", "GB", "TB"];
  let index = 0;
  while (bytes >= 1024 && index < units.length - 1) {
    bytes /= 1024;
    index++;
  }
  return index === 0 ? `${bytes} B` : `${bytes.toFixed(1)} ${units[index]}`;
}
