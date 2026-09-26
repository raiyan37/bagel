import { describe, expect, it } from 'vitest';
import { sampleTracks } from '../test/fixtures';
import {
  assetUrl,
  cardPosition,
  coverTransform,
  frameIndexAt,
  needsResync,
  playerSnapshot,
  videoPercentToScreen,
} from './match';

describe('frameIndexAt', () => {
  it('maps time to a clamped frame index', () => {
    expect(frameIndexAt(0, 25, 250)).toBe(0);
    expect(frameIndexAt(1, 25, 250)).toBe(25);
    expect(frameIndexAt(0.12, 25, 250)).toBe(3);
    expect(frameIndexAt(100, 25, 250)).toBe(249);
    expect(frameIndexAt(-1, 25, 250)).toBe(0);
    expect(frameIndexAt(Number.NaN, 25, 250)).toBe(0);
  });
});

describe('playerSnapshot', () => {
  it('reads one frame and tracks the top speed so far', () => {
    expect(playerSnapshot(sampleTracks, 'near', 2)).toEqual({
      x: 52, y: 61, visible: false, speedKmh: 12, distanceM: 0.3, topSpeedKmh: 14,
    });
    expect(playerSnapshot(sampleTracks, 'far', 99).x).toBe(44);
  });
});

describe('cover geometry', () => {
  it('matches object-fit: cover', () => {
    const t = coverTransform({ width: 1280, height: 720 }, { width: 1000, height: 1000 });
    expect(t.scale).toBeCloseTo(1000 / 720);
    expect(t.offsetX).toBeCloseTo((1000 - 1280 * (1000 / 720)) / 2);
    expect(t.offsetY).toBeCloseTo(0);
    const center = videoPercentToScreen(50, 50, { width: 1280, height: 720 }, { width: 1000, height: 1000 });
    expect(center.left).toBeCloseTo(500);
    expect(center.top).toBeCloseTo(500);
  });

  it('places cards above the anchor and keeps them on screen', () => {
    const container = { width: 1000, height: 600 };
    const card = { width: 180, height: 160 };
    expect(cardPosition({ left: 500, top: 400 }, card, container)).toEqual({ left: 410, top: 228 });
    expect(cardPosition({ left: 10, top: 50 }, card, container)).toEqual({ left: 8, top: 8 });
    expect(cardPosition({ left: 995, top: 400 }, card, container)).toEqual({ left: 812, top: 228 });
  });
});

describe('misc helpers', () => {
  it('builds asset urls and detects drift', () => {
    expect(assetUrl('demo', 'pov_near.mp4')).toBe('/matches/demo/pov_near.mp4');
    expect(needsResync(1.0, 1.1)).toBe(false);
    expect(needsResync(1.0, 1.3)).toBe(true);
  });
});
