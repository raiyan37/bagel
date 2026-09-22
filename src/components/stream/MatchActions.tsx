import { LiveIndicator } from '../shared/LiveIndicator';
import { IconButton } from '../shared/IconButton';
import './styles.css';

const StatsIcon = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <line x1="18" y1="20" x2="18" y2="10" />
    <line x1="12" y1="20" x2="12" y2="4" />
    <line x1="6" y1="20" x2="6" y2="14" />
  </svg>
);

const HeartIcon = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 0 0-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 0 0 0-7.78z" />
  </svg>
);

const MoreIcon = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <circle cx="12" cy="12" r="1" />
    <circle cx="19" cy="12" r="1" />
    <circle cx="5" cy="12" r="1" />
  </svg>
);

export function MatchActions() {
  return (
    <div className="match-actions glass glass-noise animate-fade-in-scale">
      <div className="match-live-badge">
        <LiveIndicator size="sm" />
      </div>
      <div className="match-actions-buttons">
        <IconButton variant="ghost" size="sm" title="Statistics">
          <StatsIcon />
        </IconButton>
        <IconButton variant="ghost" size="sm" title="Favorite">
          <HeartIcon />
        </IconButton>
        <IconButton variant="ghost" size="sm" title="More">
          <MoreIcon />
        </IconButton>
      </div>
    </div>
  );
}
