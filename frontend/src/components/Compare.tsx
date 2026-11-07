import { useEffect, useState } from "react";
import { api } from "../api";
import { duration } from "../geometry";
import type { Run, WhyCard } from "../types";
import { usePlayback } from "../usePlayback";
import { Formula } from "./Formula";
import { SceneCanvas } from "./SceneCanvas";

const fmt = (v: number | null | undefined, d = 3) => (v == null ? "—" : v.toFixed(d));
const timeOf = (r: Run) => r.verification.simulated_time_s ?? r.verification.predicted_time_s;
const speedOf = (r: Run) => r.verification.simulated_speed_m_s ?? r.verification.predicted_speed_m_s;

function ratio(a: number | null, b: number | null) {
  return a && b ? `${(b / a).toFixed(2)}×` : "—";
}

/** Two runs side by side, playing in sync, with the why-card for the difference. */
export function Compare({ runs, defaultPair }: { runs: Run[]; defaultPair: [string, string] }) {
  const [pair, setPair] = useState(defaultPair);
  const [card, setCard] = useState<WhyCard | null>(null);
  useEffect(() => {
    setPair(defaultPair); // follow the newest run
  }, [defaultPair[0], defaultPair[1]]);

  const a = runs.find((r) => r.run_id === pair[0]);
  const b = runs.find((r) => r.run_id === pair[1]);
  const length = Math.max(a ? duration(a.frames) : 0, b ? duration(b.frames) : 0);
  const { t, playing, play, reset } = usePlayback(length, `${pair[0]}-${pair[1]}`);

  useEffect(() => {
    if (!a || !b || a.run_id === b.run_id) {
      setCard(null);
      return;
    }
    let live = true;
    api.whycard({ run_a: a.run_id, run_b: b.run_id }).then((c) => live && setCard(c)).catch(() => setCard(null));
    return () => {
      live = false;
    };
  }, [a?.run_id, b?.run_id]);

  if (!a || !b) return null;
  const select = (i: 0 | 1) => (
    <select value={pair[i]} aria-label={i ? "second run" : "first run"}
            onChange={(e) => setPair(i ? [pair[0], e.target.value] : [e.target.value, pair[1]])}>
      {runs.map((r) => <option key={r.run_id} value={r.run_id}>Run {r.index}</option>)}
    </select>
  );

  return (
    <section className="compare">
      <div className="row wrap">
        <h2>Compare</h2>
        {select(0)}
        <span className="muted">vs</span>
        {select(1)}
        <button onClick={play} disabled={playing}>Play both</button>
        <button onClick={reset}>Reset</button>
      </div>
      <div className="pair">
        <SceneCanvas run={a} t={t} label={`Run ${a.index}`} />
        <SceneCanvas run={b} t={t} label={`Run ${b.index}`} />
      </div>
      <table>
        <thead>
          <tr><th></th><th>Run {a.index}</th><th>Run {b.index}</th><th>ratio</th></tr>
        </thead>
        <tbody>
          <tr><th>time (s)</th><td>{fmt(timeOf(a))}</td><td>{fmt(timeOf(b))}</td><td>{ratio(timeOf(a), timeOf(b))}</td></tr>
          <tr><th>final speed (m/s)</th><td>{fmt(speedOf(a))}</td><td>{fmt(speedOf(b))}</td><td>{ratio(speedOf(a), speedOf(b))}</td></tr>
        </tbody>
      </table>
      {card && (
        <div className={card.cancels ? "whycard cancels" : "whycard"}>
          <h3>{card.cancels ? "It cancels" : "Why?"}</h3>
          <Formula latex={card.latex} />
          <p>{card.sentence}</p>
        </div>
      )}
    </section>
  );
}
