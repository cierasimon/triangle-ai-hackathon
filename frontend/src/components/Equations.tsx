import katex from "katex";
import { velocityAt } from "../geometry";
import type { Run } from "../types";

interface Props {
  run: Run;
  angle: number; // deg, live
  mass: number; // A
  otherMass: number | null; // B, C, ... (null: nothing else in the picture)
  friction: number;
  gravity: number;
  v0: number; // A's initial speed (+ down the ramp)
  travel: number; // m, from A's start to the bottom of the ramp
  time: number | null; // s to the bottom
  speed: number | null; // m/s at the bottom
  changing: boolean; // a slider is moving: the collision numbers are from the last run
}

const f = (x: number, d = 2) => (Math.abs(x) < 0.005 ? 0 : x).toFixed(d);

function K({ tex }: { tex: string }) {
  return <span className="tex" dangerouslySetInnerHTML={{
    __html: katex.renderToString(tex, { throwOnError: false, displayMode: false }),
  }} />;
}

/** Signed velocities of the two bodies along A's line of motion, just before and after the hit. */
function collisionNumbers(run: Run, mA: number, mB: number) {
  const c = run.collision;
  if (!c) return null;
  const at = (t: number, i: number) => velocityAt(run.frames, Math.max(0, t), i);
  const aIn = at(c.t - 0.03, -1);
  const norm = Math.hypot(aIn[0], aIn[1]);
  if (norm < 1e-6) return null;
  const u = [aIn[0] / norm, aIn[1] / norm];
  const dot = (v: [number, number]) => v[0] * u[0] + v[1] * u[1];
  const sign = (t: number, i: number) => (dot(at(t, i)) >= 0 ? 1 : -1);
  const after = c.t + 0.06;
  const v1 = c.v1_before, v2 = sign(c.t - 0.03, c.other) * c.v2_before;
  const w1 = sign(after, -1) * c.v1_after, w2 = sign(after, c.other) * c.v2_after;
  return {
    v1, v2, w1, w2,
    pBefore: mA * v1 + mB * v2, pAfter: mA * w1 + mB * w2,
    e: Math.abs(v1 - v2) > 1e-6 ? (w2 - w1) / (v1 - v2) : null,
  };
}

/** The physics behind what is on screen, with the live numbers put in. */
export function Equations({ run, angle, mass, otherMass, friction, gravity, v0, travel, time, speed, changing }: Props) {
  const th = (angle * Math.PI) / 180;
  const roll = run.scene.params.motion === "roll";
  const along = mass * gravity * Math.sin(th); // weight component down the slope
  const normal = mass * gravity * Math.cos(th);
  const fr = friction * normal;
  const a = roll ? (5 / 7) * gravity * Math.sin(th) : gravity * (Math.sin(th) - friction * Math.cos(th));
  const holds = !roll && friction >= Math.tan(th) && v0 === 0;

  const mB = otherMass ?? 1;
  const hit = changing ? null : collisionNumbers(run, mass, mB);

  return (
    <div className="equations">
      <section>
        <h3>Newton's laws · along the ramp</h3>
        <p><K tex={String.raw`\sum F = ma`} />{" "}
          {roll ? <K tex={String.raw`\Rightarrow\ a=\tfrac{5}{7}\,g\sin\theta`} />
            : <K tex={String.raw`\Rightarrow\ mg\sin\theta-\mu mg\cos\theta = ma`} />}</p>
        {!roll && <p><K tex={String.raw`a=g(\sin\theta-\mu\cos\theta)=${f(gravity)}\,(\sin ${f(angle, 0)}^\circ-${f(friction)}\cos ${f(angle, 0)}^\circ)=${f(a)}\ \mathrm{m/s^2}`} /></p>}
        {roll && <p><K tex={String.raw`a=\tfrac{5}{7}(${f(gravity)})\sin ${f(angle, 0)}^\circ=${f(a)}\ \mathrm{m/s^2}`} /></p>}
        <p className="small"><K tex={String.raw`mg\sin\theta=${f(along)}\,\mathrm N,\ \ N=mg\cos\theta=${f(normal)}\,\mathrm N${roll ? "" : String.raw`,\ \ f=\mu N=${f(fr)}\,\mathrm N`}`} /></p>
        {holds && <p className="small">μ ≥ tan θ ({f(friction)} ≥ {f(Math.tan(th))}): friction holds the ball; it does not slide.</p>}
      </section>

      <section>
        <h3>Motion · constant acceleration</h3>
        <p><K tex={String.raw`v=v_0+at,\quad s=v_0t+\tfrac12at^2,\quad v^2=v_0^2+2as`} /></p>
        {time !== null && speed !== null ? (
          <p><K tex={String.raw`t=\dfrac{\sqrt{v_0^2+2aL}-v_0}{a}=${f(time)}\ \mathrm s,\quad v=\sqrt{v_0^2+2aL}=${f(speed)}\ \mathrm{m/s}`} /></p>
        ) : (
          <p className="small">The ball does not reach the bottom with these values.</p>
        )}
        <p className="small"><K tex={String.raw`L=${f(travel)}\,\mathrm m,\ \ v_0=${f(v0, 1)}\,\mathrm{m/s}`} /></p>
      </section>

      {otherMass !== null && (
        <section>
          <h3>Collision · momentum</h3>
          <p><K tex={String.raw`m_Av_A+m_Bv_B=m_Av_A'+m_Bv_B'`} /></p>
          {hit ? (
            <>
              <p><K tex={String.raw`${f(mass)}(${f(hit.v1)})+${f(mB)}(${f(hit.v2)})=${f(mass)}(${f(hit.w1)})+${f(mB)}(${f(hit.w2)})`} /></p>
              <p className="small"><K tex={String.raw`p_{\text{before}}=${f(hit.pBefore)},\ p_{\text{after}}=${f(hit.pAfter)}\ \mathrm{kg\,m/s}`} />
                {hit.e !== null && <> · <K tex={String.raw`e=\dfrac{v_B'-v_A'}{v_A-v_B}=${f(hit.e)}`} /></>}</p>
              <p className="small">Along A's line of motion; gravity acts during contact, so p is only approximately conserved.</p>
            </>
          ) : (
            <p className="small">{changing ? "Updating…" : "A and B do not collide in this run."}</p>
          )}
        </section>
      )}
    </div>
  );
}
