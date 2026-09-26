import { useEffect, useRef } from 'react';
import { needsResync } from '../../lib/match';

interface PovVideoProps {
  src: string;
  mainTime: number;
  isPlaying: boolean;
  label: string;
  className?: string;
}

/** A POV clip rendered frame-aligned with the broadcast video: same fps and frame count, so time maps 1:1. */
export function PovVideo({ src, mainTime, isPlaying, label, className }: PovVideoProps) {
  const ref = useRef<HTMLVideoElement>(null);

  // `src` is a dependency because swapping the source resets the element to frame 0 and pauses it,
  // which has to be corrected even when neither the clock nor the play state moved.
  useEffect(() => {
    const video = ref.current;
    if (!video) return;
    if (needsResync(video.currentTime, mainTime)) {
      video.currentTime = mainTime;
    }
    if (isPlaying && video.paused) {
      video.play().catch(() => {});
    } else if (!isPlaying && !video.paused) {
      video.pause();
    }
  }, [src, mainTime, isPlaying]);

  return <video ref={ref} className={className} src={src} muted playsInline preload="auto" aria-label={label} />;
}
