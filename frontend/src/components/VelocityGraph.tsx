import { useEffect, useMemo, useRef, useState } from "react";
import { rampOf, velocityAt } from "../geometry";
import type { Run } from "../types";

interface Props {
  run: Run;
  t: number; // playback time (s): a marker follows it
}

const HEIGHT = 168;
const M = { l: 38, r: 30, t: 14, b: 26 };
const SERIES = ["var(--series-1)", "var(--series-2)", "var(--series-3)", "var(--series-4)"];
const MAX_SERIES = SERIES.length;

/** A "nice" tick step for a span, 1/2/5 × 10^k. */
function niceStep(span: number, target: number) {
  const raw = span / target;
  const pow = 10 ** Math.floor(Math.log10(raw));
  const f = raw / pow;
  return (f < 1.5 ? 1 : f < 3.5 ? 2 : f < 7.5 ? 5 : 10) * pow;
}

interface Series {
  name: string;
  v: number[]; // signed velocity at each sample time
}

/** Velocity against time for every object. Positive is downhill, then away from the ramp along
 * the ground; negative is back up the ramp / toward it. */
function sample(run: Run): { times: number[]; series: Series[] } {
  const base = rampOf(run.scene);
  const away = -base.side; // the ground direction leading away from the ramp
  const times = run.frames.map((f) => f.t);
  const count = Math.min(MAX_SERIES, 1 + run.scene.others.length);
  const series: Series[] = [];
  for (let i = 0; i < count; i++) {
    const v = times.map((t) => {
      const [vx, vy] = velocityAt(run.frames, t, i - 1);
      const speed = Math.hypot(vx, vy);
      const along = Math.abs(vx) > 1e-3 * speed ? vx * away : -vy; // vertical ramp: falling is positive
      return along >= 0 ? speed : -speed;
    });
    series.push({ name: String.fromCharCode(65 + i), v });
  }
  return { times, series };
}

const f2 = (x: number) => (Math.abs(x) < 0.005 ? 0 : x).toFixed(2);

/** Velocity-time graph: one line per object (A, B, …), a marker that follows the playback,
 * a hover readout of every line at that time, and a table view. */
export function VelocityGraph({ run, t }: Props) {
  const box = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(360);
  const [hover, setHover] = useState<number | null>(null);
  const [table, setTable] = useState(false);

  useEffect(() => {
    const el = box.current;
    if (!el) return;
    const ro = new ResizeObserver(() => setWidth(Math.max(240, el.clientWidth)));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const { times, series } = useMemo(() => sample(run), [run]);
  const tMax = times[times.length - 1] || 1;
  const all = series.flatMap((s) => s.v);
  const vMin = Math.min(0, ...all);
  const vMax = Math.max(0.1, ...all);
  const yStep = niceStep(vMax - vMin || 1, 4);
  const yLo = Math.floor(vMin / yStep) * yStep;
  const yHi = Math.ceil(vMax / yStep) * yStep;
  const xStep = niceStep(tMax, 5);

  const w = width - M.l - M.r;
  const h = HEIGHT - M.t - M.b;
  const x = (tt: number) => M.l + (tt / tMax) * w;
  const y = (v: number) => M.t + (1 - (v - yLo) / (yHi - yLo || 1)) * h;

  const yTicks: number[] = [];
  for (let v = yLo; v <= yHi + 1e-9; v += yStep) yTicks.push(Math.round(v * 1e6) / 1e6);
  const xTicks: number[] = [];
  for (let tt = 0; tt <= tMax + 1e-9; tt += xStep) xTicks.push(Math.round(tt * 1e6) / 1e6);

  // nearest sample to a time
  const nearest = (tt: number) => {
    let lo = 0;
    let hi = times.length - 1;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (times[mid] < tt) lo = mid + 1;
      else hi = mid;
    }
    return lo > 0 && tt - times[lo - 1] < times[lo] - tt ? lo - 1 : lo;
  };
  const now = nearest(Math.min(t, tMax));
  const shown = hover ?? null;

  function onMove(e: React.PointerEvent<SVGSVGElement>) {
    const rect = e.currentTarget.getBoundingClientRect();
    const px = ((e.clientX - rect.left) / rect.width) * width;
    setHover(nearest(Math.min(tMax, Math.max(0, ((px - M.l) / w) * tMax))));
  }
  function onKey(e: React.KeyboardEvent<SVGSVGElement>) {
    if (e.key !== "ArrowLeft" && e.key !== "ArrowRight") return;
    e.preventDefault();
    const step = e.key === "ArrowRight" ? 4 : -4;
    setHover((h0) => Math.min(times.length - 1, Math.max(0, (h0 ?? now) + step)));
  }

  const c = run.collision;
  const single = series.length === 1;
  const rows = useMemo(() => {
    const step = Math.max(1, Math.round(times.length / 14));
    return times.map((_, k) => k).filter((k) => k % step === 0 || k === times.length - 1);
  }, [times]);

  return (
    <div className="vgraph" ref={box}>
      <div className="vg-head">
        <h3>{single ? "Velocity of A" : "Velocity"} vs time</h3>
        {!single && (
          <span className="vg-legend">
            {series.map((s, i) => (
              <span key={s.name}><i style={{ background: SERIES[i] }} />{s.name}</span>
            ))}
          </span>
        )}
        <button className="chip ghost" onClick={() => setTable(!table)} aria-pressed={table}>
          {table ? "Graph" : "Table"}
        </button>
      </div>

      {table ? (
        <div className="vg-table" style={{ height: HEIGHT }}>
          <table>
            <thead>
              <tr><th>t (s)</th>{series.map((s) => <th key={s.name}>{s.name} (m/s)</th>)}</tr>
            </thead>
            <tbody>
              {rows.map((k) => (
                <tr key={k}><td>{f2(times[k])}</td>{series.map((s) => <td key={s.name}>{f2(s.v[k])}</td>)}</tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <div className="vg-plot">
          <svg width={width} height={HEIGHT} viewBox={`0 0 ${width} ${HEIGHT}`} tabIndex={0}
               role="img" aria-label="Velocity of each object against time. Use the arrow keys to read values."
               onPointerMove={onMove} onPointerLeave={() => setHover(null)} onKeyDown={onKey}
               onBlur={() => setHover(null)}>
            {yTicks.map((v) => (
              <g key={`y${v}`}>
                <line className={v === 0 ? "vg-zero" : "vg-grid"} x1={M.l} x2={width - M.r} y1={y(v)} y2={y(v)} />
                <text className="vg-tick" x={M.l - 6} y={y(v)} textAnchor="end" dominantBaseline="middle">{+v.toFixed(2)}</text>
              </g>
            ))}
            {xTicks.map((tt) => (
              <text key={`x${tt}`} className="vg-tick" x={x(tt)} y={HEIGHT - 8} textAnchor="middle">{+tt.toFixed(2)}</text>
            ))}
            <text className="vg-axis" x={M.l} y={9}>m/s</text>
            <text className="vg-axis" x={width - M.r} y={HEIGHT - 8} textAnchor="end">s</text>

            {c && (
              <g>
                <line className="vg-hit" x1={x(c.t)} x2={x(c.t)} y1={M.t} y2={M.t + h} />
                <text className="vg-axis" x={x(c.t) + 4} y={M.t + 9}>hit</text>
              </g>
            )}

            {series.map((s, i) => (
              <polyline key={s.name} fill="none" stroke={SERIES[i]} strokeWidth={2} strokeLinejoin="round"
                        strokeLinecap="round"
                        points={s.v.map((v, k) => `${x(times[k]).toFixed(1)},${y(v).toFixed(1)}`).join(" ")} />
            ))}

            {/* playback marker: a hairline and a dot per line */}
            <line className="vg-now" x1={x(times[now])} x2={x(times[now])} y1={M.t} y2={M.t + h} />
            {series.map((s, i) => (
              <circle key={`n${s.name}`} cx={x(times[now])} cy={y(s.v[now])} r={4} fill={SERIES[i]} className="vg-dot" />
            ))}

            {/* direct labels at the line ends (a few lines only) */}
            {!single && series.map((s) => (
              <text key={`l${s.name}`} className="vg-label" x={x(tMax) + 5} y={y(s.v[s.v.length - 1])}
                    dominantBaseline="middle">{s.name}</text>
            ))}

            {shown !== null && (
              <g>
                <line className="vg-cross" x1={x(times[shown])} x2={x(times[shown])} y1={M.t} y2={M.t + h} />
                {series.map((s, i) => (
                  <circle key={`h${s.name}`} cx={x(times[shown])} cy={y(s.v[shown])} r={4.5} fill={SERIES[i]} className="vg-dot" />
                ))}
              </g>
            )}
          </svg>
          {shown !== null && (
            <div className="vg-tip" style={{
              left: Math.min(width - 130, Math.max(M.l, x(times[shown]) + 10)), top: M.t,
            }}>
              <div className="vg-tip-t">t = {f2(times[shown])} s</div>
              {series.map((s, i) => (
                <div key={s.name} className="vg-tip-row">
                  <i style={{ background: SERIES[i] }} /><b>{f2(s.v[shown])} m/s</b><span>{s.name}</span>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
      <p className="vg-note">+ downhill, then away from the ramp · − back up / toward it</p>
    </div>
  );
}
