import { FileText } from "lucide-react";

/** The table row under a failed Pipeline or Endpoint, saying why, with its full log if it has one. */
export default function FailureRow({
  failure,
  columns,
  logUrl,
}: {
  failure: string;
  columns: number;
  logUrl?: string;
}) {
  return (
    <tr className="failure-row">
      <td colSpan={columns}>
        <pre className="failure">{failure}</pre>
        {logUrl && (
          <a className="link-button" href={logUrl} target="_blank" rel="noreferrer">
            <FileText size={14} /> Full log
          </a>
        )}
      </td>
    </tr>
  );
}
