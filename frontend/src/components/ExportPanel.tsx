import { useEffect, useState } from "react";
import { api } from "../api";
import type { ExportResponse, Run } from "../types";

/** "Take it home": render the run's GIF on the server, show it with a QR code to the phone page. */
export function ExportPanel({ run }: { run: Run }) {
  const [result, setResult] = useState<ExportResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    setResult(null);
    setError(null);
  }, [run.run_id]);

  async function make() {
    setBusy(true);
    setError(null);
    try {
      setResult(await api.exportRun(run));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  async function copy(url: string) {
    try {
      await navigator.clipboard.writeText(url);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1500);
    } catch {
      // clipboard blocked (http on a phone): the link is still visible to copy by hand
    }
  }

  const shareId = result?.share_url.split("/").pop();
  const isGif = result?.gif_url.endsWith(".gif");
  const host = result ? new URL(result.share_url).hostname : "";
  const reach = /^(localhost|127\.|\[::1\])/.test(host)
    ? "This link only opens on this computer. Download the GIF instead, or run make tunnel for a public link."
    : /^(10\.|192\.168\.|172\.(1[6-9]|2\d|3[01])\.)/.test(host)
      ? "Your phone must be on the same Wi-Fi as this computer to open it."
      : null;

  return (
    <div className="export">
      {!result && (
        <button className="primary" onClick={make} disabled={busy}>
          {busy ? "Making your animation…" : "Download latest run"}
        </button>
      )}
      {error && <p className="error">{error}</p>}
      {result && shareId && (
        <div className="export-result">
          <img className="export-gif" src={result.gif_url} alt="animation of the latest run" />
          <div className="export-side">
            <img className="export-qr" src={result.qr_png_url} alt="QR code to the download page" />
            <p className="muted">Scan with your phone to save the {isGif ? "GIF" : "image"}.</p>
            {reach && <p className="muted note">{reach}</p>}
            <a className="share-link" href={result.share_url} target="_blank" rel="noreferrer">
              {result.share_url}
            </a>
            <div className="row wrap">
              <button className="chip" onClick={() => copy(result.share_url)}>{copied ? "Copied" : "Copy link"}</button>
              <a className="chip" href={`/api/share/${shareId}/download`} download>
                Download {isGif ? "GIF" : "PNG"}
              </a>
              <a className="chip" href={`${result.qr_png_url}?download=1`} download>Download QR</a>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
