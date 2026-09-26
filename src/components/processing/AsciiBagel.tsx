import { useEffect, useRef } from 'react';
import { useAccessibility } from '../../contexts/AccessibilityContext';
import { renderBagel } from './bagelTorus';
import './processing.css';

export function AsciiBagel() {
  const ref = useRef<HTMLPreElement>(null);
  // Follow the in-app motion setting, not just the OS media query, so toggling
  // Motion in the UI starts and stops the spin straight away.
  const { settings } = useAccessibility();
  const still = settings.reducedMotion;

  useEffect(() => {
    const node = ref.current;
    if (!node) return;

    if (still) {
      node.textContent = renderBagel(0.6, 1.2);
      return;
    }

    let spinX = 0;
    let spinZ = 0;
    let raf = 0;
    let last = 0;

    const tick = (now: number) => {
      raf = requestAnimationFrame(tick);
      if (now - last < 45) return; // ~22fps reads as animated ASCII rather than video
      last = now;
      spinX += 0.05;
      spinZ += 0.026;
      node.textContent = renderBagel(spinX, spinZ);
    };

    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [still]);

  return <pre className="bagel-ascii" ref={ref} aria-hidden="true" />;
}
