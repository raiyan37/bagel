import type { PlayerInfo } from '../../lib/match';
import './tennis.css';

interface TennisScoreOverlayProps {
  near: PlayerInfo;
  far: PlayerInfo;
  score: string;
  competition: string;
}

function surname(name: string): string {
  const parts = name.trim().split(/\s+/);
  return (parts[parts.length - 1] || name).toUpperCase();
}

export function TennisScoreOverlay({ near, far, score, competition }: TennisScoreOverlayProps) {
  return (
    <div
      className="tennis-score"
      role="status"
      aria-label={`${near.name} versus ${far.name}${score ? `, score ${score}` : ''}`}
    >
      <div className="tennis-score-glass" aria-hidden="true" />
      <div className="tennis-score-content">
        <div className="tennis-score-player">
          <span className="tennis-score-dot" style={{ background: near.color }} aria-hidden="true" />
          <span className="tennis-score-name">{surname(near.name)}</span>
        </div>
        <div className="tennis-score-center">
          <span className="tennis-score-value">{score || 'vs'}</span>
          <span className="tennis-score-label">{competition}</span>
        </div>
        <div className="tennis-score-player">
          <span className="tennis-score-name">{surname(far.name)}</span>
          <span className="tennis-score-dot" style={{ background: far.color }} aria-hidden="true" />
        </div>
      </div>
    </div>
  );
}
