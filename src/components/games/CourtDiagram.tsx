import type { CSSProperties } from 'react';
import './games.css';

/*
  Plan view at true ITF proportions (23.77 x 10.97 m doubles, 8.23 m singles,
  service lines 6.40 m from the net), scaled 20x into user units. The three
  marks are the vantage points the product actually offers.
*/
export function CourtDiagram() {
  return (
    <svg
      className="court"
      viewBox="0 0 665 300"
      role="img"
      aria-label="A tennis court from above, marked with the three viewpoints bagel can place you at: behind the near baseline, behind the far baseline, and anywhere else on court."
    >
      <g className="court-lines" fill="none" stroke="currentColor" strokeWidth="2.2" pathLength={1}>
        <rect x="95" y="60" width="475" height="219" />
        <line x1="95" y1="87" x2="570" y2="87" />
        <line x1="95" y1="252" x2="570" y2="252" />
        <line x1="205" y1="87" x2="205" y2="252" />
        <line x1="461" y1="87" x2="461" y2="252" />
        <line x1="205" y1="170" x2="461" y2="170" />
        <line x1="95" y1="163" x2="95" y2="177" />
        <line x1="570" y1="163" x2="570" y2="177" />
      </g>

      <line className="court-net" x1="333" y1="48" x2="333" y2="291" strokeWidth="3" />

      <g className="court-marks">
        <g className="court-mark" style={{ '--i': 0 } as CSSProperties}>
          <circle cx="45" cy="170" r="6" />
          <text x="45" y="196">near eye</text>
        </g>
        <g className="court-mark" style={{ '--i': 1 } as CSSProperties}>
          <circle cx="620" cy="170" r="6" />
          <text x="620" y="196">far eye</text>
        </g>
        <g className="court-mark court-mark--free" style={{ '--i': 2 } as CSSProperties}>
          <circle cx="405" cy="26" r="9" fill="none" strokeWidth="3" />
          <text x="405" y="12">anywhere</text>
        </g>
      </g>
    </svg>
  );
}
