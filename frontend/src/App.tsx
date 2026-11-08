import { useMemo, useRef, useState } from "react";
import { api } from "./api";
import { Compare } from "./components/Compare";
import { Controls } from "./components/Controls";
import { ExportPanel } from "./components/ExportPanel";
import { Formula } from "./components/Formula";
import { RecognitionDetails } from "./components/RecognitionDetails";
import { SceneCanvas } from "./components/SceneCanvas";
import { TaDrawer } from "./components/TaDrawer";
import { duration } from "./geometry";
import { predict } from "./physics";
import type { ParamsOverride, Run, UploadResponse } from "./types";
import { usePlayback } from "./usePlayback";

const fmt = (v: number | null, digits = 3) => (v === null ? "—" : v.toFixed(digits));
const RERUN_DELAY_MS = 150;
const PARAM_LABELS: Record<string, string> = {
  angle_deg: "θ", mass_kg: "m", gravity: "g", friction: "μ",
};

/** The params the learner has set themselves on a run (e.g. one a quiz created). */
function overrideFrom(run: Run): ParamsOverride {
  const p = run.scene.params;
  const out: Record<string, unknown> = {};
  for (const [k, src] of Object.entries(p.sources)) if (src === "user") out[k] = p[k as keyof typeof p];
  return out as ParamsOverride;
}

function changeSummary(run: Run): string {
  const parts = Object.entries(run.changed).map(
    ([k, [, b]]) => `${PARAM_LABELS[k] ?? k} ${+b.toFixed(2)}`,
  );
  return parts.length ? parts.join(", ") : run.index === 1 ? "as drawn" : "";
}

export function App() {
  const [upload, setUpload] = useState<UploadResponse | null>(null);
  const [runs, setRuns] = useState<Run[]>([]);
  const [currentId, setCurrentId] = useState<string | null>(null);
  const [override, setOverride] = useState<ParamsOverride>({});
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [dragAngle, setDragAngle] = useState<number | null>(null);
  const [snap, setSnap] = useState(false);
  const [useGemini, setUseGemini] = useState(true);
  const releaseTimer = useRef<number | undefined>(undefined);

  const run = runs.find((r) => r.run_id === currentId) ?? null;
  const { t, playing, play, reset } = usePlayback(run ? duration(run.frames) : 0, run?.run_id ?? "");

  async function onFile(file: File | undefined) {
    if (!file) return;
    setBusy(true);
    setError(null);
    setRuns([]);
    setCurrentId(null);
    setOverride({});
    setUpload(null);
    try {
      const up = await api.upload(file, useGemini ? "hybrid" : "opencv");
      setUpload(up);
      if (up.recognition.ball && up.recognition.lines.some(line => line.role === "ramp")) {
        await requestRun(up.image_id, {}, "change");
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  async function requestRun(imageId: string, next: ParamsOverride, reason: "change" | "drag") {
    const r = await api.run({ image_id: imageId, params_override: next, reason });
    setOverride(next);
    setRuns((prev) => [...prev, r]);
    setCurrentId(r.run_id);
    return r;
  }

  async function change(next: ParamsOverride, reason: "change" | "drag" = "change") {
    if (!upload) return;
    setBusy(true);
    setError(null);
    try {
      await requestRun(upload.image_id, next, reason);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
      setDragAngle(null);
    }
  }

  function onDrag(angle: number) {
    window.clearTimeout(releaseTimer.current);
    setDragAngle(angle);
  }

  function onDragEnd(angle: number) {
    window.clearTimeout(releaseTimer.current);
    releaseTimer.current = window.setTimeout(
      () => change({ ...override, angle_deg: angle }, "drag"),
      RERUN_DELAY_MS,
    );
  }

  const rec = upload?.recognition;
  const v = run?.verification;
  const p = run?.scene.params;
  const drawnAngle = runs[0]?.scene.params.angle_deg ?? null;
  const live = useMemo(
    () => (run && p && dragAngle != null
      ? predict(dragAngle, run.scene.ramp_length_m, p.gravity, p.friction, p.motion)
      : null),
    [run, p, dragAngle],
  );
  const comparePair = useMemo<[string, string] | null>(() => {
    if (runs.length < 2 || !run) return null;
    const i = runs.findIndex((r) => r.run_id === run.run_id);
    return [runs[Math.max(0, i - 1)].run_id, run.run_id];
  }, [runs, run]);

  return (
    <main>
      <header>
        <h1>PaperPhysics</h1>
        <label className="upload">
          {busy && !run ? "Working…" : "Take / choose a photo"}
          <input
            type="file"
            accept="image/*"
            capture="environment"
            disabled={busy}
            onChange={(e) => onFile(e.target.files?.[0])}
          />
        </label>
      </header>
      <label className="muted check">
        <input type="checkbox" checked={useGemini} disabled={busy}
          onChange={e => setUseGemini(e.target.checked)} /> Use Gemini + OpenCV recognition
      </label>
      <p className="muted">When enabled, uploads are sent to Google Cloud and may incur API charges.
        Turn off for local OpenCV recognition. This choice applies to the next upload.</p>
      {busy && !run && <p role="status">{useGemini ? "Recognizing with Gemini and OpenCV…" : "Recognizing locally…"}</p>}
      {error && <p className="error">{error}</p>}

      {upload && (
        <div className="grid">
          <section>
            <h2>Recognized</h2>
            <img src={upload.overlay_png_url} alt="photo with recognized outlines" />
            {rec && <RecognitionDetails recognition={rec} />}
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
              <div className="row wrap">
                <h2>Run {run.index}</h2>
                {busy && <span className="muted">running…</span>}
              </div>
              <SceneCanvas
                run={run}
                t={t}
                onDrag={busy ? undefined : onDrag}
                onDragEnd={onDragEnd}
                dragAngle={dragAngle}
                originalAngle={drawnAngle}
                snap={snap}
              />
              <div className="row wrap">
                <button onClick={play} disabled={playing || dragAngle != null}>Play</button>
                <button onClick={reset}>Reset</button>
                <label className="muted check">
                  <input type="checkbox" checked={snap} onChange={(e) => setSnap(e.target.checked)} /> snap to 5°
                </label>
                {dragAngle == null && <span className="muted">drag the red handle to tilt the ramp</span>}
              </div>
              {live && dragAngle != null && (
                <p className="live">
                  θ = {dragAngle.toFixed(1)}° ·{" "}
                  {live.will_move
                    ? `predicted t = ${fmt(live.time_s, 2)} s · v = ${fmt(live.speed_m_s, 2)} m/s`
                    : `the ball will not slide at this angle (μ = ${p.friction} ≥ tan θ = ${Math.tan((dragAngle * Math.PI) / 180).toFixed(2)})`}
                </p>
              )}

              <div className="history">
                {runs.map((r) => (
                  <button key={r.run_id} className={r.run_id === run.run_id ? "chip active" : "chip"}
                          onClick={() => setCurrentId(r.run_id)}>
                    Run {r.index}
                    <span className="chip-note">{changeSummary(r)}</span>
                  </button>
                ))}
              </div>

              <Controls
                params={p}
                busy={busy}
                angleOverridden={override.angle_deg !== undefined}
                onChange={(patch) => change({ ...override, ...patch })}
                onResetAngle={() => {
                  const rest = { ...override };
                  delete rest.angle_deg;
                  change(rest);
                }}
                onResetAll={() => change({})}
              />

              <table>
                <tbody>
                  {(["angle_deg", "mass_kg", "gravity", "friction", "motion"] as const).map((k) => (
                    <tr key={k}>
                      <th>{k}</th>
                      <td>{typeof p[k] === "number" ? +(p[k] as number).toFixed(2) : p[k]}</td>
                      <td><span className={`badge ${p.sources[k] ?? "default"}`}>{p.sources[k] ?? "default"}</span></td>
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
              <ExportPanel run={run} />
            </section>
          )}
        </div>
      )}

      {comparePair && <Compare runs={runs} defaultPair={comparePair} />}

      <TaDrawer
        runs={runs}
        busy={busy}
        onApply={(patch) => {
          // the backend sends unset fields as null; only the set ones are the change
          const set = Object.fromEntries(Object.entries(patch).filter(([, v]) => v != null));
          change({ ...override, ...set });
        }}
        onRun={(r) => {
          setRuns((prev) => [...prev, r]);
          setCurrentId(r.run_id);
          setOverride(overrideFrom(r));
        }}
      />
    </main>
  );
}
