import { useMemo, useRef, useState } from "react";
import { api } from "./api";
import { Controls, letter, type BodyControl } from "./components/Controls";
import { Equations } from "./components/Equations";
import { ExportPanel } from "./components/ExportPanel";
import { RecognitionDetails } from "./components/RecognitionDetails";
import { VelocityGraph } from "./components/VelocityGraph";
import { SceneCanvas } from "./components/SceneCanvas";
import { TaPanel } from "./components/TaPanel";
import { duration, rampOf } from "./geometry";
import { predict } from "./physics";
import type { Params, ParamsOverride, Run, UploadResponse } from "./types";
import { usePlayback } from "./usePlayback";

const fmt = (v: number | null, digits = 2) => (v === null ? "—" : v.toFixed(digits));
const RERUN_DELAY_MS = 300;
const title = (s: string) => s[0].toUpperCase() + s.slice(1);

export function App() {
  const [upload, setUpload] = useState<UploadResponse | null>(null);
  const [runs, setRuns] = useState<Run[]>([]);
  const [currentId, setCurrentId] = useState<string | null>(null);
  const [override, setOverride] = useState<ParamsOverride>({});
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [useGemini, setUseGemini] = useState(true);
  // changes being made right now (sliders, drags): shown at once, simulated after a pause
  const [draft, setDraft] = useState<ParamsOverride>({});
  const releaseTimer = useRef<number | undefined>(undefined);
  const pending = useRef<ParamsOverride>({});

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
      setDraft({});
    }
  }

  /** Preview a change right away, then rerun the simulation once the learner pauses. */
  function preview(patch: ParamsOverride, reason: "change" | "drag") {
    setDraft((d) => ({ ...d, ...patch }));
    pending.current = { ...pending.current, ...patch };
    window.clearTimeout(releaseTimer.current);
    releaseTimer.current = window.setTimeout(() => {
      const next = { ...override, ...pending.current };
      pending.current = {};
      change(next, reason);
    }, RERUN_DELAY_MS);
  }

  const p = run?.scene.params;
  const v = run?.verification;
  const drawnAngle = runs[0]?.scene.params.angle_deg ?? null;
  const fullLength = run ? rampOf(run.scene).fullLength : 1;
  const angle = draft.angle_deg ?? p?.angle_deg ?? 0;
  const length = draft.ramp_full_m ?? fullLength;
  const mass = draft.mass_kg ?? p?.mass_kg ?? 1;
  const friction = draft.friction ?? p?.friction ?? 0;
  const gravity = draft.gravity ?? p?.gravity ?? 9.81;
  const v0 = draft.initial_speed_m_s ?? p?.initial_speed_m_s ?? 0;
  // every object, A first: initial speed (signed), shown live while a slider moves
  const otherSpeeds = (run?.scene.others ?? []).map((o, i) => draft.other_speeds_m_s?.[i] ?? o.speed_m_s);
  const bodies: BodyControl[] = run ? [
    { name: letter(0), speed: v0, surface: "ramp", tag: tag("initial_speed_m_s") },
    ...run.scene.others.map((o, i) => ({
      name: letter(i + 1), speed: otherSpeeds[i], surface: o.surface,
      tag: draft.other_speeds_m_s?.[i] != null || override.other_speeds_m_s?.[i] != null ? "changed"
        : o.speed_m_s !== 0 ? "from drawing" : "default",
    })),
  ] : [];
  function setSpeed(body: number, v: number) {
    if (body === 0) return preview({ initial_speed_m_s: v }, "change");
    preview({ other_speeds_m_s: otherSpeeds.map((s, i) => (i === body - 1 ? v : s)) }, "change");
  }
  function tag(key: keyof ParamsOverride & keyof Params): string {
    if (draft[key] !== undefined) return "changed";
    const src = p?.sources[key] ?? "default";
    return src === "user" ? "changed" : src === "default" ? "default" : "from drawing";
  }
  const other = run?.scene.others[0];
  const otherMass = other ? draft.other_mass_kg ?? other.mass_kg : null;
  const changing = Object.keys(draft).length > 0;
  const travel = draft.start_fraction !== undefined
    ? draft.start_fraction * length
    : run ? (run.scene.ramp_length_m * length) / fullLength : 0;
  const live = useMemo(() => {
    const touched = ["angle_deg", "ramp_full_m", "start_fraction", "friction", "gravity", "initial_speed_m_s"]
      .some((k) => draft[k as keyof ParamsOverride] !== undefined);
    if (!run || !p || !touched) return null;
    return predict(angle, travel, gravity, friction, p.motion, v0, length - travel);
  }, [run, p, draft, angle, travel, length, gravity, friction, v0]);
  const time = live ? live.time_s : v?.simulated_time_s ?? null;
  const speed = live ? live.speed_m_s : v?.simulated_speed_m_s ?? null;
  const hit = changing ? null : run?.collision;

  return (
    <div className="page">
    <main>
      <header>
        <h1>PaperPhysics</h1>
        <label className="upload">
          {busy && !run ? "Working…" : upload ? "New photo" : "Take / choose a photo"}
          <input
            type="file"
            accept="image/*"
            capture="environment"
            disabled={busy}
            onChange={(e) => onFile(e.target.files?.[0])}
          />
        </label>
      </header>
      <label className="muted check" title="Applies to the next upload. Turn off for local OpenCV recognition.">
        <input type="checkbox" checked={useGemini} disabled={busy}
          onChange={e => setUseGemini(e.target.checked)} /> Use Gemini + OpenCV recognition
        <span className="notice">· uploads are sent to Google Cloud and may incur API charges</span>
      </label>
      {busy && !run && <p role="status">{useGemini ? "Recognizing with Gemini and OpenCV…" : "Recognizing locally…"}</p>}
      {error && <p className="error">{error}</p>}
      {!upload && !busy && (
        <p className="hint">Draw a ramp and a ball on paper, then take a photo of it.</p>
      )}

      {upload && !run && !busy && (
        <section className="panel recognized">
          <h2>Recognized</h2>
          <img src={upload.overlay_png_url} alt="photo with recognized outlines" />
          <RecognitionDetails recognition={upload.recognition} />
          {upload.recognition.warnings.length > 0 && (
            <ul className="warnings">
              {upload.recognition.warnings.map((w) => <li key={w}>{w}</li>)}
            </ul>
          )}
        </section>
      )}

      {run && p && upload && (
        <div className="lab">
          <section className="stage">
            <SceneCanvas
              run={run}
              t={t}
              onDrag={busy ? undefined : (a, l) => setDraft((d) => ({ ...d, angle_deg: a, ramp_full_m: l }))}
              onDragEnd={(a, l) => preview({ angle_deg: a, ramp_full_m: l }, "drag")}
              dragAngle={draft.angle_deg}
              dragLength={draft.ramp_full_m}
              originalAngle={drawnAngle}
              mass={mass}
              otherMass={otherMass ?? undefined}
              speeds={bodies.map((b) => b.speed)}
              onBallDrag={busy ? undefined : (f) => setDraft((d) => ({ ...d, start_fraction: f }))}
              onBallDragEnd={(f) => preview({ start_fraction: f }, "change")}
              dragStart={draft.start_fraction}
              onOtherDrag={busy || !other ? undefined
                : (o) => setDraft((d) => ({ ...d, other_surface: o.surface, other_at: o.at }))}
              onOtherDragEnd={(o) => preview({ other_surface: o.surface, other_at: o.at }, "change")}
              dragOther={draft.other_surface && draft.other_at !== undefined
                ? { surface: draft.other_surface, at: draft.other_at } : null}
            />
            <div className="below">
            <VelocityGraph run={run} t={t} />
            <Equations
              run={run}
              angle={angle}
              mass={mass}
              otherMass={otherMass}
              friction={friction}
              gravity={gravity}
              v0={v0}
              travel={travel}
              time={time}
              speed={speed}
              changing={changing}
            />
            </div>
          </section>

          <aside className="side">
            <div className="panel">
              <h2>{other ? "Ramp & objects" : `Ramp & ${run.scene.ball_shape}`}</h2>
              <Controls
                angle={angle}
                length={length}
                mass={mass}
                otherMass={otherMass}
                friction={friction}
                gravity={gravity}
                bodies={bodies}
                tags={{
                  angle_deg: tag("angle_deg"), mass_kg: tag("mass_kg"), friction: tag("friction"),
                  gravity: tag("gravity"), initial_speed_m_s: tag("initial_speed_m_s"),
                }}
                disabled={busy}
                onAngle={(a) => preview({ angle_deg: a }, "change")}
                onLength={(l) => preview({ ramp_full_m: l }, "change")}
                onMass={(m) => preview({ mass_kg: m }, "change")}
                onOtherMass={(m) => preview({ other_mass_kg: m }, "change")}
                onFriction={(mu) => preview({ friction: mu }, "change")}
                onGravity={(g) => preview({ gravity: g }, "change")}
                onSpeed={setSpeed}
              />
            </div>
            <div className="panel">
              <h2>Result</h2>
              <div className="stat"><span>Time to bottom</span><strong>{fmt(time)} s</strong></div>
              <div className="stat"><span>Speed at bottom</span><strong>{fmt(speed)} m/s</strong></div>
              {hit && (
                <>
                  <h3>Collision at {fmt(hit.t)} s</h3>
                  <div className="stat"><span>{title(run.scene.ball_shape)} 1</span><strong>{fmt(hit.v1_before)} → {fmt(hit.v1_after)} m/s</strong></div>
                  <div className="stat"><span>{title(run.scene.others[hit.other]?.shape ?? "ball")} 2</span><strong>{fmt(hit.v2_before)} → {fmt(hit.v2_after)} m/s</strong></div>
                </>
              )}
            </div>
            <div className="buttons">
              <button className="wide" onClick={play} disabled={playing || changing}>Play</button>
              <button onClick={reset}>Reset</button>
              <button disabled={busy} onClick={() => change({})}>Back to drawing</button>
            </div>
            <ExportPanel run={run} />
            <details className="photo">
              <summary>Your photo</summary>
              <img src={upload.overlay_png_url} alt="photo with recognized outlines" />
              <RecognitionDetails recognition={upload.recognition} />
              {upload.recognition.warnings.length > 0 && (
                <ul className="warnings">
                  {upload.recognition.warnings.map((w) => <li key={w}>{w}</li>)}
                </ul>
              )}
            </details>
          </aside>
        </div>
      )}
    </main>
    <TaPanel runs={runs} />
    </div>
  );
}
