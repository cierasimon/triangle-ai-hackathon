import { useCallback, useEffect, useRef, useState } from "react";

/** A clock from 0 to `length` seconds; restarts whenever `key` changes. */
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
    const start = performance.now() - tRef.current * 1000;
    const tick = (now: number) => {
      const next = Math.min((now - start) / 1000, length);
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
