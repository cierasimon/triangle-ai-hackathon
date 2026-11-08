import { useCallback, useEffect, useRef, useState } from "react";

/** Real seconds of physics per second of wall clock: the ball moves in slow motion. */
export const PLAYBACK_SPEED = 0.12;
/** Wall-clock pause before the ball starts, so the setup can be read first. */
const HOLD_MS = 700;

/** A clock from 0 to `length` seconds (slowed down); restarts whenever `key` changes. */
export function usePlayback(length: number, key: string) {
  const [t, setT] = useState(0);
  const [playing, setPlaying] = useState(true);
  const tRef = useRef(0);

  useEffect(() => {
    tRef.current = 0;
    setT(0);
    setPlaying(true);
  }, [key]);

  useEffect(() => {
    if (!playing) return;
    let raf = 0;
    const hold = tRef.current === 0 ? HOLD_MS : 0;
    const start = performance.now() + hold - (tRef.current * 1000) / PLAYBACK_SPEED;
    const tick = (now: number) => {
      const next = Math.min(Math.max(0, ((now - start) / 1000) * PLAYBACK_SPEED), length);
      tRef.current = next;
      setT(next);
      if (next < length) raf = requestAnimationFrame(tick);
      else setPlaying(false);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [playing, length, key]);

  const play = useCallback(() => {
    if (tRef.current >= length) tRef.current = 0;
    setPlaying(true);
  }, [length]);
  const reset = useCallback(() => {
    setPlaying(false);
    tRef.current = 0;
    setT(0);
  }, []);

  return { t, playing, play, reset };
}
