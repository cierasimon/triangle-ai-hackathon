import { useState } from "react";
import { api } from "./api";
import { Formula } from "./components/Formula";
import { RunCanvas } from "./components/RunCanvas";
import type { Run, UploadResponse } from "./types";

const fmt = (v: number | null, digits = 3) => (v === null ? "—" : v.toFixed(digits));

export function App() {
  const [upload, setUpload] = useState<UploadResponse | null>(null);
  const [run, setRun] = useState<Run | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function onFile(file: File | undefined) {
    if (!file) return;
    setBusy(true);
    setError(null);
    setRun(null);
    try {
      const up = await api.upload(file);
      setUpload(up);
      setRun(await api.run({ image_id: up.image_id }));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  const rec = upload?.recognition;
  const v = run?.verification;
  const p = run?.scene.params;

  return (
    <main>
      <header>
        <h1>PaperPhysics</h1>
        <label className="upload">
          {busy ? "Working…" : "Take / choose a photo"}
          <input
            type="file"
            accept="image/*"
            capture="environment"
            disabled={busy}
            onChange={(e) => onFile(e.target.files?.[0])}
          />
        </label>
      </header>
      {error && <p className="error">{error}</p>}

      {upload && (
        <div className="grid">
          <section>
            <h2>Recognized</h2>
            <img src={upload.overlay_png_url} alt="photo with recognized outlines" />
            {rec && rec.warnings.length > 0 && (
              <ul className="warnings">
                {rec.warnings.map((w) => <li key={w}>{w}</li>)}
              </ul>
            )}
            <p className="muted">
              {rec?.lines.map((l) => l.role).join(" + ")}
              {rec?.ball ? " + ball" : " · no ball"} · paper {rec?.paper_found ? "found" : "not found"}
            </p>
          </section>

          {run && v && p && (
            <section>
              <h2>Run {run.index}</h2>
              <RunCanvas run={run} />
              <table>
                <tbody>
                  {(["angle_deg", "mass_kg", "gravity", "friction", "motion"] as const).map((k) => (
                    <tr key={k}>
                      <th>{k}</th>
                      <td>{typeof p[k] === "number" ? +(p[k] as number).toFixed(2) : p[k]}</td>
                      <td><span className="badge">{p.sources[k] ?? "default"}</span></td>
                    </tr>
                  ))}
                </tbody>
              </table>
              {p.defaults_used.length > 0 && (
                <p className="muted">
                  Defaults used: {p.defaults_used.filter((k) => k !== "ball").join(", ")}
                  {p.defaults_used.includes("ball") && " · no ball was drawn, so one starts near the top"}
                </p>
              )}
              <table>
                <thead>
                  <tr><th></th><th>predicted</th><th>simulated</th></tr>
                </thead>
                <tbody>
                  <tr><th>time (s)</th><td>{fmt(v.predicted_time_s)}</td><td>{fmt(v.simulated_time_s)}</td></tr>
                  <tr><th>speed (m/s)</th><td>{fmt(v.predicted_speed_m_s)}</td><td>{fmt(v.simulated_speed_m_s)}</td></tr>
                  <tr className={v.ok ? "" : "bad"}>
                    <th>error</th><td colSpan={2}>{v.error_pct === null ? "—" : `${v.error_pct.toFixed(2)} %`}</td>
                  </tr>
                </tbody>
              </table>
              <Formula latex={v.formula_latex} />
              {v.note && <p className="muted">{v.note}</p>}
            </section>
          )}
        </div>
      )}
    </main>
  );
}
