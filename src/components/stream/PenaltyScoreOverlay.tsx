import { useState, useEffect } from 'react';
import './penalty-glass.css';
import manUtdLogo from '../../assets/man_utd.png';
import arsenalLogo from '../../assets/arsenal.png';

interface Team {
  name: string;
  shortName: string;
  flag: string;
  color: string;
}

interface PenaltyStats {
  scored: number;
  taken: number;
  saved: number;
}

interface PenaltyScoreOverlayProps {
  homeTeam: Team;
  awayTeam: Team;
  homePenalties: PenaltyStats;
  awayPenalties: PenaltyStats;
  currentRound: number;
  isHomeTurn: boolean;
  justScored?: 'home' | 'away' | null;
}

export function PenaltyScoreOverlay({
  homeTeam,
  awayTeam,
  homePenalties,
  awayPenalties,
  isHomeTurn,
  justScored,
}: PenaltyScoreOverlayProps) {
  const [mounted, setMounted] = useState(false);
  const [animateScore, setAnimateScore] = useState<'home' | 'away' | null>(null);

  useEffect(() => {
    setMounted(true);
  }, []);

  // Trigger score animation when goal is scored
  useEffect(() => {
    if (justScored) {
      setAnimateScore(justScored);
      // Reset animation after it completes
      const timer = setTimeout(() => setAnimateScore(null), 600);
      return () => clearTimeout(timer);
    }
  }, [justScored]);

  const renderPenaltyDots = (stats: PenaltyStats, team: 'home' | 'away') => {
    const dots = [];
    for (let i = 0; i < 5; i++) {
      let status = 'empty';
      if (i < stats.taken) {
        status = i < stats.scored ? 'scored' : 'missed';
      }
      // Add animation class if this is the newly scored dot
      const isNewlyScored = animateScore === team && status === 'scored' && i === stats.scored - 1;
      dots.push(
        <div
          key={i}
          className={`pk-dot pk-dot--${status} ${isNewlyScored ? 'pk-dot--animate' : ''}`}
          style={{ animationDelay: `${i * 60}ms` }}
        />
      );
    }
    return dots;
  };

  return (
    <div className={`pk-broadcast ${mounted ? 'pk-broadcast--show' : ''}`}>
      {/* Glass background */}
      <div className="pk-broadcast-glass" />
      <div className="pk-broadcast-border" />

      {/* Content */}
      <div className="pk-broadcast-content">
        {/* Home Team (Man Utd) */}
        <div className={`pk-team-section ${isHomeTurn ? 'pk-team-section--active' : ''}`}>
          <img src={manUtdLogo} alt="Man Utd" className="pk-team-logo" />
          <div className="pk-team-info">
            <span className="pk-team-name">{homeTeam.shortName}</span>
            <div className="pk-dots">{renderPenaltyDots(homePenalties, 'home')}</div>
          </div>
        </div>

        {/* Score */}
        <div className="pk-score-section">
          <div className="pk-score-display">
            <span className={`pk-score-number ${animateScore === 'home' ? 'pk-score-number--animate' : ''}`}>
              {homePenalties.scored}
            </span>
            <span className="pk-score-dash">-</span>
            <span className={`pk-score-number ${animateScore === 'away' ? 'pk-score-number--animate' : ''}`}>
              {awayPenalties.scored}
            </span>
          </div>
          <div className="pk-label">PENALTIES</div>
        </div>

        {/* Away Team (Arsenal) */}
        <div className={`pk-team-section ${!isHomeTurn ? 'pk-team-section--active' : ''}`}>
          <div className="pk-team-info pk-team-info--reverse">
            <span className="pk-team-name">{awayTeam.shortName}</span>
            <div className="pk-dots">{renderPenaltyDots(awayPenalties, 'away')}</div>
          </div>
          <img src={arsenalLogo} alt="Arsenal" className="pk-team-logo" />
        </div>
      </div>
    </div>
  );
}
