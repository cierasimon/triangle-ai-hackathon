import { useEffect, useRef, useState } from "react";
import type { Run } from "../types";

// Plays back a Run's trajectory (meters, y up) on a canvas. Physics happens on the backend.
export function RunCanvas({ run }: { run: Run }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [playing, setPlaying] = useState(true);
  const [frameIndex, setFrameIndex] = useState(0);
  const [size, setSize] = useState({ w: 0, h: 0 });

  // redraw whenever the canvas gets laid out or resized (it can be 0×0 on first mount)
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const observer = new ResizeObserver(() =>
      setSize({ w: canvas.clientWidth, h: canvas.clientHeight }),
    );
    observer.observe(canvas);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    setFrameIndex(0);
    setPlaying(true);
  }, [run.run_id]);

  useEffect(() => {
    if (!playing) return;
    let raf = 0;
    const start = performance.now() - (run.frames[frameIndex]?.t ?? 0) * 1000;
    const tick = (now: number) => {
      const t = (now - start) / 1000;
      let i = run.frames.findIndex((f) => f.t >= t);
      if (i === -1) i = run.frames.length - 1;
      setFrameIndex(i);
      if (i < run.frames.length - 1) raf = requestAnimationFrame(tick);
      else setPlaying(false);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
    // frameIndex is only read to resume from the current frame, so it is not a dependency
  }, [playing, run]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const dpr = window.devicePixelRatio || 1;
    const { w: cssW, h: cssH } = size;
    if (!cssW || !cssH) return;
    canvas.width = cssW * dpr;
    canvas.height = cssH * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    const { ramp_top_m: top, ramp_bottom_m: bottom, ground_y_m, ball_radius_m } = run.scene;
    const xs = [top[0], bottom[0]];
    const ys = [top[1], bottom[1], ground_y_m];
    const pad = 0.15;
    const minX = Math.min(...xs) - pad, maxX = Math.max(...xs) + pad;
    const minY = Math.min(...ys) - pad, maxY = Math.max(...ys) + pad;
    const scale = Math.min(cssW / (maxX - minX), cssH / (maxY - minY));
    const offX = (cssW - (maxX - minX) * scale) / 2;
    const offY = (cssH - (maxY - minY) * scale) / 2;
    const px = (x: number) => offX + (x - minX) * scale;
    const py = (y: number) => cssH - offY - (y - minY) * scale;

    ctx.clearRect(0, 0, cssW, cssH);
    ctx.lineCap = "round";
    ctx.lineWidth = 3;
    ctx.strokeStyle = "#2563eb";
    ctx.beginPath();
    ctx.moveTo(px(minX), py(ground_y_m));
    ctx.lineTo(px(maxX), py(ground_y_m));
    ctx.stroke();
    ctx.strokeStyle = "#dc2626";
    ctx.beginPath();
    ctx.moveTo(px(top[0]), py(top[1]));
    ctx.lineTo(px(bottom[0]), py(bottom[1]));
    ctx.stroke();

    const f = run.frames[frameIndex];
    if (f) {
      const r = ball_radius_m * scale;
      ctx.save();
      ctx.translate(px(f.x), py(f.y));
      ctx.rotate(-f.angle);
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
    }
  }, [run, frameIndex, size]);

  const t = run.frames[frameIndex]?.t ?? 0;
  return (
    <div className="run-canvas">
      <canvas ref={canvasRef} />
      <div className="row">
        <button onClick={() => setPlaying(true)} disabled={playing}>
          Play
        </button>
        <button onClick={() => { setPlaying(false); setFrameIndex(0); }}>Reset</button>
        <span className="muted">t = {t.toFixed(2)} s</span>
      </div>
    </div>
  );
}
