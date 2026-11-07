import { useEffect, useRef, useState } from "react";
import { angleTo, ballStartAt, frameAt, rampOf, topAt, viewOf } from "../geometry";
import type { Point, Run } from "../types";

interface Props {
  run: Run;
  t: number; // playback time (s)
  /** Enable the drag handle on the ramp top. Called with live angles, then on release. */
  onDrag?: (angleDeg: number) => void;
  onDragEnd?: (angleDeg: number) => void;
  dragAngle?: number | null; // preview angle while dragging
  originalAngle?: number | null; // faint dashed ramp
  snap?: boolean;
  label?: string;
}

const HANDLE_PX = 11;
const HIT_PX = 28;

// Draws one run (meters, y up) with a fixed view, so dragging the ramp doesn't rescale it.
export function SceneCanvas({ run, t, onDrag, onDragEnd, dragAngle, originalAngle, snap, label }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [size, setSize] = useState({ w: 0, h: 0 });
  const dragging = useRef(false);
  const lastAngle = useRef<number | null>(null);

  // redraw whenever the canvas gets laid out or resized (it can be 0×0 on first mount)
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const observer = new ResizeObserver(() => setSize({ w: canvas.clientWidth, h: canvas.clientHeight }));
    observer.observe(canvas);
    return () => observer.disconnect();
  }, []);

  const ramp = rampOf(run.scene);
  const view = viewOf(ramp, run.scene.ground_y_m);
  const scale = Math.min(size.w / (view.maxX - view.minX), size.h / (view.maxY - view.minY)) || 1;
  const offX = (size.w - (view.maxX - view.minX) * scale) / 2;
  const offY = (size.h - (view.maxY - view.minY) * scale) / 2;
  const px = (x: number) => offX + (x - view.minX) * scale;
  const py = (y: number) => size.h - offY - (y - view.minY) * scale;
  const toMeters = (cx: number, cy: number): Point => [
    (cx - offX) / scale + view.minX,
    (size.h - offY - cy) / scale + view.minY,
  ];

  const angle = dragAngle ?? run.scene.params.angle_deg;
  const top = topAt(ramp, angle);

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
    const [bx, by] = ramp.bottom;

    ctx.lineWidth = 3;
    ctx.strokeStyle = "#2563eb";
    ctx.beginPath();
    ctx.moveTo(px(view.minX), py(run.scene.ground_y_m));
    ctx.lineTo(px(view.maxX), py(run.scene.ground_y_m));
    ctx.stroke();

    if (originalAngle != null && Math.abs(originalAngle - angle) > 0.05) {
      const o = topAt(ramp, originalAngle);
      ctx.save();
      ctx.setLineDash([6, 6]);
      ctx.strokeStyle = "rgba(220, 38, 38, 0.35)";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(px(bx), py(by));
      ctx.lineTo(px(o[0]), py(o[1]));
      ctx.stroke();
      ctx.restore();
    }

    ctx.strokeStyle = "#dc2626";
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(px(bx), py(by));
    ctx.lineTo(px(top[0]), py(top[1]));
    ctx.stroke();

    let ball: Point;
    let spin = 0;
    if (dragAngle != null) {
      ball = ballStartAt(ramp, dragAngle);
    } else {
      const f = frameAt(run.frames, t);
      ball = f ? [f.x, f.y] : ballStartAt(ramp, angle);
      spin = f?.angle ?? 0;
    }
    const r = ramp.radius * scale;
    ctx.save();
    ctx.translate(px(ball[0]), py(ball[1]));
    ctx.rotate(-spin);
    ctx.fillStyle = "#16a34a";
    ctx.beginPath();
    ctx.arc(0, 0, r, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = "#fff";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(0, 0);
    ctx.lineTo(r, 0);
    ctx.stroke();
    ctx.restore();

    if (onDrag) {
      ctx.fillStyle = dragging.current ? "#dc2626" : "#fff";
      ctx.strokeStyle = "#dc2626";
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.arc(px(top[0]), py(top[1]), HANDLE_PX, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
    }
    if (label) {
      ctx.fillStyle = "#6b6b6b";
      ctx.font = "13px system-ui, sans-serif";
      ctx.fillText(label, 8, 18);
    }
  });

  function local(e: React.PointerEvent<HTMLCanvasElement>): [number, number] {
    const rect = e.currentTarget.getBoundingClientRect();
    return [e.clientX - rect.left, e.clientY - rect.top];
  }

  const handlers = onDrag
    ? {
        onPointerDown(e: React.PointerEvent<HTMLCanvasElement>) {
          const [cx, cy] = local(e);
          if (Math.hypot(cx - px(top[0]), cy - py(top[1])) > HIT_PX) return;
          dragging.current = true;
          e.currentTarget.setPointerCapture(e.pointerId);
          e.preventDefault();
        },
        onPointerMove(e: React.PointerEvent<HTMLCanvasElement>) {
          if (!dragging.current) return;
          const a = angleTo(ramp, toMeters(...local(e)), snap);
          if (a !== lastAngle.current) {
            lastAngle.current = a;
            onDrag(a);
          }
        },
        onPointerUp(e: React.PointerEvent<HTMLCanvasElement>) {
          if (!dragging.current) return;
          dragging.current = false;
          e.currentTarget.releasePointerCapture(e.pointerId);
          if (lastAngle.current != null) onDragEnd?.(lastAngle.current);
          lastAngle.current = null;
        },
      }
    : {};

  return (
    <canvas
      ref={canvasRef}
      className={onDrag ? "scene draggable" : "scene"}
      {...handlers}
      onPointerCancel={handlers.onPointerUp}
    />
  );
}
