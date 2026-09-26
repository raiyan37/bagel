import { render } from '@testing-library/react';
import { act } from 'react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { AccessibilityProvider } from '../../contexts/AccessibilityContext';
import { AsciiBagel } from './AsciiBagel';
import { renderBagel } from './bagelTorus';

/** Drive requestAnimationFrame by hand so we can watch consecutive frames. */
function useManualRaf() {
  let now = 0;
  const queue: FrameRequestCallback[] = [];
  vi.stubGlobal('requestAnimationFrame', (cb: FrameRequestCallback) => {
    queue.push(cb);
    return queue.length;
  });
  vi.stubGlobal('cancelAnimationFrame', () => {});
  return {
    advance(ms: number) {
      now += ms;
      const due = queue.splice(0, queue.length);
      act(() => due.forEach((cb) => cb(now)));
    },
  };
}

describe('renderBagel', () => {
  it('draws a torus with a visible hole', () => {
    const rows = renderBagel(0.6, 1.2).split('\n');
    expect(rows).toHaveLength(22);
    expect(rows.every((r) => r.length === 44)).toBe(true);
    // Solid shading around a gap: the middle rows must have ink on both sides of a space.
    const holed = rows.filter((r) => /\S\s{2,}\S/.test(r));
    expect(holed.length).toBeGreaterThan(0);
  });

  it('produces a different frame at each rotation, so it actually animates', () => {
    const frames = [0, 1, 2, 3].map((i) => renderBagel(i * 0.05, i * 0.026));
    expect(new Set(frames).size).toBe(frames.length);
  });

  it('is deterministic for a given rotation', () => {
    expect(renderBagel(1.1, 0.4)).toBe(renderBagel(1.1, 0.4));
  });
});

describe('AsciiBagel', () => {
  beforeEach(() => localStorage.clear());
  afterEach(() => localStorage.clear());

  it('redraws the bagel at a new rotation on each animation frame', () => {
    localStorage.setItem('accessibility_settings', JSON.stringify({ highContrast: false, reducedMotion: false }));
    const raf = useManualRaf();
    render(
      <AccessibilityProvider>
        <AsciiBagel />
      </AccessibilityProvider>,
    );
    const pre = document.querySelector('.bagel-ascii') as HTMLPreElement;

    raf.advance(100);
    const first = pre.textContent;
    raf.advance(100);
    const second = pre.textContent;

    expect(first).toBeTruthy();
    expect(second).toBeTruthy();
    expect(second).not.toBe(first);
  });

  it('holds a single frame when the viewer asked for reduced motion', () => {
    localStorage.setItem('accessibility_settings', JSON.stringify({ highContrast: false, reducedMotion: true }));
    const raf = useManualRaf();
    render(
      <AccessibilityProvider>
        <AsciiBagel />
      </AccessibilityProvider>,
    );
    const pre = document.querySelector('.bagel-ascii') as HTMLPreElement;

    const first = pre.textContent;
    raf.advance(500);
    expect(pre.textContent).toBe(first);
    expect(first).toBe(renderBagel(0.6, 1.2));
  });
});
