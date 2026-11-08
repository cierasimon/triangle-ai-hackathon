import { useEffect, useRef, useState } from "react";
import {
  angleTo, ballStartAt, duration, fractionAt, frameAt, lengthTo, otherRestAt, rampOf, topAt,
  velocityAt, viewOf, withLength,
} from "../geometry";
import type { Point, Run, Shape } from "../types";

interface Props {
  run: Run;
  t: number; // playback time (s)
  /** Enable the handle on the ramp top: it sets the angle and the length. Live, then on release. */
  onDrag?: (angleDeg: number, lengthM: number) => void;
  onDragEnd?: (angleDeg: number, lengthM: number) => void;
  dragAngle?: number | null; // preview angle while changing it
  dragLength?: number | null; // preview whole ramp length while changing it
  originalAngle?: number | null; // faint dashed ramp
  mass?: number; // preview mass for the weight arrow
  otherMass?: number; // preview mass of the other balls
  /** Initial speed of each object, A first (+ down the ramp / right along the ground). Shown as arrows at the start. */
  speeds?: number[];
  /** Enable dragging the ball along the ramp. Called with where it starts (0 = bottom, 1 = top). */
  onBallDrag?: (fraction: number) => void;
  onBallDragEnd?: (fraction: number) => void;
  dragStart?: number | null; // preview ball start while dragging it
  /** Enable dragging the second object along the ramp or the flat ground. */
  onOtherDrag?: (place: OtherPlace) => void;
  onOtherDragEnd?: (place: OtherPlace) => void;
  dragOther?: OtherPlace | null; // preview second-object place while dragging it
}

/** Where the second object rests: as OtherBall.surface and .at. */
export interface OtherPlace {
  surface: "ramp" | "ground";
  at: number;
}

const HANDLE_PX = 9;
const MIN_BALL_PX = 8;
const WALL_PX = 22; // the stop at the end of the ground
const HIT_PX = 28;
const FG = "#ececec";
const MUTED = "#8c8c8c";
const FAINT = "#4a4a4a";
const FONT = "13px system-ui, sans-serif";
const letterOf = (i: number) => String.fromCharCode(65 + i);

function arrow(ctx: CanvasRenderingContext2D, x0: number, y0: number, x1: number, y1: number) {
  const len = Math.hypot(x1 - x0, y1 - y0);
  if (len < 2) return;
  const head = Math.min(10, len * 0.4);
  const a = Math.atan2(y1 - y0, x1 - x0);
  ctx.beginPath();
  ctx.moveTo(x0, y0);
  ctx.lineTo(x1, y1);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(x1 - head * Math.cos(a - 0.45), y1 - head * Math.sin(a - 0.45));
  ctx.lineTo(x1 - head * Math.cos(a + 0.45), y1 - head * Math.sin(a + 0.45));
  ctx.closePath();
  ctx.fill();
}

interface BallState {
  shape: Shape;
  pos: Point; // centre as simulated (for a body of `radius`)
  spin: number;
  vel: Point;
  radius: number; // m, as simulated (half the side for a block)
  drawRadius: number; // m: bigger or smaller while its mass slider is being moved
  mass: number;
}

/** Same material: size grows with the cube root of mass (mirrors scene/builder.py size_factor). */
const sizeRatio = (mass: number, simulated: number) => Math.cbrt(mass / simulated);

// Draws one run (meters, y up) with a fixed view, so dragging the ramp doesn't rescale it.
export function SceneCanvas({
  run, t, onDrag, onDragEnd, dragAngle, dragLength, originalAngle, mass, otherMass, speeds,
  onBallDrag, onBallDragEnd, dragStart, onOtherDrag, onOtherDragEnd, dragOther,
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [size, setSize] = useState({ w: 0, h: 0 });
  const dragging = useRef<"ramp" | "ball" | "other" | null>(null);
  const lastValue = useRef<string | null>(null);

  // redraw whenever the canvas gets laid out or resized (it can be 0×0 on first mount)
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const observer = new ResizeObserver(() => setSize({ w: canvas.clientWidth, h: canvas.clientHeight }));
    observer.observe(canvas);
    return () => observer.disconnect();
  }, []);

  const scene = run.scene;
  const base = rampOf(scene);
  const view = viewOf(base, scene.ground_y_m, scene.others, scene.ground_end_x_m); // from the run, not the preview
  const ramp = dragLength != null ? withLength(base, dragLength) : base;
  const scale = Math.min(size.w / (view.maxX - view.minX), size.h / (view.maxY - view.minY)) || 1;
  const offX = (size.w - (view.maxX - view.minX) * scale) / 2;
  const offY = (size.h - (view.maxY - view.minY) * scale) / 2;
  const px = (x: number) => offX + (x - view.minX) * scale;
  const py = (y: number) => size.h - offY - (y - view.minY) * scale;
  const toMeters = (cx: number, cy: number): Point => [
    (cx - offX) / scale + view.minX,
    (size.h - offY - cy) / scale + view.minY,
  ];

  const params = scene.params;
  const angle = dragAngle ?? params.angle_deg;
  const top = topAt(ramp, angle);
  const previewing = dragAngle != null || dragLength != null || dragStart != null || dragOther != null;

  // balls: at rest at their (maybe dragged) start while anything is being changed, else playing
  const balls: BallState[] = [];
  const f = previewing ? undefined : frameAt(run.frames, t);
  const moving = !previewing && t > 0 && t < duration(run.frames);
  const a = (angle * Math.PI) / 180;
  const tilt = Math.atan2(Math.sin(a), ramp.side * Math.cos(a)); // a block lies flat on the ramp
  const m1 = mass ?? params.mass_kg;
  balls.push({
    shape: scene.ball_shape,
    pos: f ? [f.x, f.y] : ballStartAt(ramp, angle, dragStart != null ? dragStart * ramp.fullLength : ramp.travel),
    spin: f ? f.angle : scene.ball_shape === "block" ? tilt : 0,
    vel: moving ? velocityAt(run.frames, t) : [0, 0],
    radius: ramp.radius,
    drawRadius: ramp.radius * sizeRatio(m1, params.mass_kg),
    mass: m1,
  });
  scene.others.forEach((o, i) => {
    const s = f?.others?.[i];
    const m2 = otherMass ?? o.mass_kg;
    balls.push({
      shape: o.shape,
      pos: s ? [s[0], s[1]] : otherRestAt(ramp, angle, i === 0 && dragOther ? { ...o, ...dragOther } : o),
      spin: s ? s[2] : o.shape === "block" && o.surface === "ramp" ? tilt : 0,
      vel: moving ? velocityAt(run.frames, t, i) : [0, 0],
      radius: o.radius_m,
      drawRadius: o.radius_m * sizeRatio(m2, o.mass_kg),
      mass: m2,
    });
  });
  // at the start (or while something is being changed) show each object's initial velocity
  const atStart = previewing || t <= 0;
  const down: Point = [-ramp.side * Math.cos(a), -Math.sin(a)];
  balls.forEach((b, i) => {
    const v = speeds?.[i] ?? (i === 0 ? params.initial_speed_m_s : scene.others[i - 1].speed_m_s);
    const surface = i === 0 ? "ramp" : i === 1 && dragOther ? dragOther.surface : scene.others[i - 1].surface;
    if (atStart && !moving) b.vel = surface === "ramp" ? [down[0] * v, down[1] * v] : [v, 0];
  });
  const c = run.collision;
  const vMax = Math.max(
    run.verification.simulated_speed_m_s ?? run.verification.predicted_speed_m_s ?? 0,
    c?.v1_before ?? 0, c?.v2_after ?? 0, c?.v1_after ?? 0,
    ...(speeds ?? []).map(Math.abs),
  );

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext("2d");
    if (!canvas || !ctx || !size.w || !size.h) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = size.w * dpr;
    canvas.height = size.h * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, size.w, size.h);
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.font = FONT;
    const [bx, by] = ramp.bottom;
    const unit = base.fullLength * scale; // drawn ramp length in px, sizes the arrows

    // ground, up to the stop where the balls halt
    const end = scene.ground_end_x_m;
    const gy = py(scene.ground_y_m);
    ctx.lineWidth = 2;
    ctx.strokeStyle = MUTED;
    ctx.beginPath();
    ctx.moveTo(px(ramp.side > 0 ? view.maxX : view.minX), gy);
    ctx.lineTo(px(end ?? (ramp.side > 0 ? view.minX : view.maxX)), gy);
    ctx.stroke();
    if (end != null) {
      ctx.lineWidth = 4;
      ctx.strokeStyle = FG;
      ctx.beginPath();
      ctx.moveTo(px(end), gy);
      ctx.lineTo(px(end), gy - WALL_PX);
      ctx.stroke();
    }

    // the ramp before this change, while it is being changed
    if (originalAngle != null && (Math.abs(originalAngle - angle) > 0.05 || dragLength != null)) {
      const o = topAt(base, dragLength != null ? params.angle_deg : originalAngle);
      ctx.save();
      ctx.setLineDash([6, 6]);
      ctx.strokeStyle = FAINT;
      ctx.beginPath();
      ctx.moveTo(px(bx), py(by));
      ctx.lineTo(px(o[0]), py(o[1]));
      ctx.stroke();
      ctx.restore();
    }

    // ramp
    ctx.strokeStyle = FG;
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(px(bx), py(by));
    ctx.lineTo(px(top[0]), py(top[1]));
    ctx.stroke();

    // angle arc at the bottom, between the ground and the ramp
    const arcR = Math.min(56, unit * 0.22);
    const a = (angle * Math.PI) / 180;
    const startA = ramp.side > 0 ? 0 : Math.PI;
    const endA = ramp.side > 0 ? -a : Math.PI + a;
    ctx.strokeStyle = MUTED;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.arc(px(bx), py(by), arcR, Math.min(startA, endA), Math.max(startA, endA));
    ctx.stroke();
    // label just under the ground, past the arc, so it never sits on the ramp
    ctx.fillStyle = FG;
    ctx.textAlign = ramp.side > 0 ? "left" : "right";
    ctx.textBaseline = "middle";
    ctx.fillText(`θ = ${Math.round(angle)}°`, px(bx) + ramp.side * (arcR + 6), py(by) + 14);
    // ramp length, along the middle of the ramp on the outside
    const mid: Point = [(bx + top[0]) / 2, (by + top[1]) / 2];
    ctx.fillStyle = MUTED;
    ctx.textAlign = ramp.side > 0 ? "right" : "left";
    ctx.fillText(`L = ${ramp.fullLength.toFixed(2)} m`, px(mid[0]) - ramp.side * 14, py(mid[1]) - 10);

    // balls, each with its weight and velocity
    const label = (name: string, i: number) => (balls.length > 1 ? `${name}(${letterOf(i)})` : name);
    balls.forEach((b, i) => {
      // never smaller than MIN_BALL_PX on screen; drawn bigger than simulated (tiny on a long
      // ramp, or a mass slider moving), it is lifted so it still sits on the surface
      const r = Math.max(b.drawRadius * scale, MIN_BALL_PX);
      const lift = r - b.radius * scale; // px, out of the surface it rests on
      const onRamp = (b.pos[0] - ramp.bottom[0]) * ramp.side > 0 && b.pos[1] > ramp.bottom[1] + b.radius * 1.01;
      const [nx, ny] = onRamp ? [-ramp.side * Math.sin(a), Math.cos(a)] : [0, 1]; // y up
      const cx = px(b.pos[0]) + nx * lift;
      const cy = py(b.pos[1]) - ny * lift;
      ctx.save();
      ctx.translate(cx, cy);
      ctx.rotate(-b.spin);
      const held = (i === 0 && dragging.current === "ball") || (i === 1 && dragging.current === "other");
      ctx.fillStyle = held ? FG : i === 0 ? "#bdbdbd" : "#6f6f6f";
      if (b.shape === "block") {
        ctx.fillRect(-r, -r, 2 * r, 2 * r);
      } else {
        ctx.beginPath();
        ctx.arc(0, 0, r, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.restore();

      // the object's letter, A for the first, B for the second, ...
      ctx.fillStyle = held || i === 0 ? "#111" : "#f2f2f2";
      ctx.font = `700 ${Math.max(11, Math.min(r * 1.1, 22))}px system-ui, sans-serif`;
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(letterOf(i), cx, cy + 1);
      ctx.font = FONT;
      ctx.fillStyle = FG;

      // weight: straight down, longer for a heavier ball
      const wLen = Math.max(0, Math.min(unit * (0.08 + 0.017 * b.mass), size.h - cy - 24));
      ctx.strokeStyle = FG;
      ctx.fillStyle = FG;
      ctx.lineWidth = 2;
      arrow(ctx, cx, cy, cx, cy + wLen);
      ctx.textAlign = "left";
      ctx.textBaseline = "middle";
      ctx.fillText(`${label("W", i)} = ${(b.mass * params.gravity).toFixed(1)} N`, cx + 8, cy + wLen - 4);

      // velocity: along the motion, scaled to the fastest any ball gets
      const speed = Math.hypot(b.vel[0], b.vel[1]);
      if (speed > 1e-3 && vMax) {
        const vLen = unit * 0.4 * Math.min(1, speed / vMax);
        const ex = cx + (b.vel[0] / speed) * vLen;
        const ey = cy - (b.vel[1] / speed) * vLen;
        arrow(ctx, cx, cy, ex, ey);
        const text = `${label("v", i)} = ${speed.toFixed(2)} m/s`;
        const w = ctx.measureText(text).width;
        const lx = b.vel[0] >= 0 ? ex + 6 : ex - 6 - w; // left edge of the label
        ctx.textAlign = "left";
        ctx.fillText(text, Math.min(Math.max(6, lx), size.w - 6 - w), ey - 10); // stay on the canvas
      }
    });

    // clock, top right
    ctx.fillStyle = MUTED;
    ctx.textAlign = "right";
    ctx.textBaseline = "top";
    ctx.fillText(`t = ${(previewing ? 0 : t).toFixed(2)} s · slow motion`, size.w - 14, 12);

    // drag handle
    if (onDrag) {
      ctx.fillStyle = dragging.current === "ramp" ? FG : "#1c1c1c";
      ctx.strokeStyle = FG;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(px(top[0]), py(top[1]), HANDLE_PX, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
    }
  });

  function local(e: React.PointerEvent<HTMLCanvasElement>): [number, number] {
    const rect = e.currentTarget.getBoundingClientRect();
    return [e.clientX - rect.left, e.clientY - rect.top];
  }

  // the latest dragged value, as numbers, for the release callback
  const dragged = useRef<number[]>([]);
  const draggable = !!(onDrag || onBallDrag || onOtherDrag);
  const grabs = (b: BallState | undefined, cx: number, cy: number) =>
    !!b && Math.hypot(cx - px(b.pos[0]), cy - py(b.pos[1])) <= Math.max(b.drawRadius * scale, MIN_BALL_PX) + 14;

  /** Where the second object would rest for a pointer at p: on the ramp above the foot, else
   * on the flat ground, between the foot and the stop. */
  function placeOther(p: Point): OtherPlace {
    const o = scene.others[0];
    if ((p[0] - ramp.bottom[0]) * ramp.side > 0) {
      return { surface: "ramp", at: Math.round(Math.min(0.98, Math.max(0.02, fractionAt(ramp, angle, p))) * 100) / 100 };
    }
    const away = -ramp.side;
    const end = scene.ground_end_x_m ?? ramp.bottom[0] + away * ramp.fullLength;
    const lo = Math.min(ramp.bottom[0] + away * o.radius_m, end - away * o.radius_m);
    const hi = Math.max(ramp.bottom[0] + away * o.radius_m, end - away * o.radius_m);
    return { surface: "ground", at: Math.round(Math.min(hi, Math.max(lo, p[0])) * 1000) / 1000 };
  }
  const handlers = draggable
    ? {
        onPointerDown(e: React.PointerEvent<HTMLCanvasElement>) {
          const [cx, cy] = local(e);
          if (onDrag && Math.hypot(cx - px(top[0]), cy - py(top[1])) <= HIT_PX) dragging.current = "ramp";
          else if (onBallDrag && grabs(balls[0], cx, cy)) dragging.current = "ball";
          else if (onOtherDrag && grabs(balls[1], cx, cy)) dragging.current = "other";
          else return;
          e.currentTarget.setPointerCapture(e.pointerId);
          e.preventDefault();
        },
        onPointerMove(e: React.PointerEvent<HTMLCanvasElement>) {
          if (!dragging.current) return;
          const p = toMeters(...local(e));
          const place = dragging.current === "other" ? placeOther(p) : null;
          const v = dragging.current === "ramp"
            ? [Math.round(angleTo(ramp, p)), Math.round(lengthTo(ramp, p) * 100) / 100]
            : place
              ? [place.surface === "ramp" ? 1 : 0, place.at]
              : [Math.round(fractionAt(ramp, angle, p) * 100) / 100];
          const key = v.join(",");
          if (key === lastValue.current) return;
          lastValue.current = key;
          dragged.current = v;
          if (dragging.current === "ramp") onDrag?.(v[0], v[1]);
          else if (place) onOtherDrag?.(place);
          else onBallDrag?.(v[0]);
        },
        onPointerUp(e: React.PointerEvent<HTMLCanvasElement>) {
          const kind = dragging.current;
          if (!kind) return;
          dragging.current = null;
          e.currentTarget.releasePointerCapture(e.pointerId);
          const v = dragged.current;
          const moved = lastValue.current != null;
          lastValue.current = null;
          if (!moved) return;
          if (kind === "ramp") onDragEnd?.(v[0], v[1]);
          else if (kind === "other") onOtherDragEnd?.({ surface: v[0] ? "ramp" : "ground", at: v[1] });
          else onBallDragEnd?.(v[0]);
        },
      }
    : {};

  return (
    <canvas
      ref={canvasRef}
      className={draggable ? "scene draggable" : "scene"}
      {...handlers}
      onPointerCancel={handlers.onPointerUp}
    />
  );
}
