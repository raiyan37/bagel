import { useNavigate } from 'react-router-dom';
import './games.css';

export interface Match {
  id: string;
  near: string;
  far: string;
  status: 'live' | 'upcoming' | 'replay';
  time: string;
}

const statusText: Record<Match['status'], string> = {
  live: 'live',
  replay: 'replay',
  upcoming: 'upcoming',
};

export function MatchRow({ match }: { match: Match }) {
  const navigate = useNavigate();
  const open = () => navigate(`/processing/${match.id}`);
  const label = match.status === 'upcoming' ? match.time || statusText.upcoming : statusText[match.status];

  return (
    <article
      className="match-row"
      role="listitem"
      tabIndex={0}
      aria-label={`${match.near} versus ${match.far}. ${label}. Press Enter to watch.`}
      onClick={open}
      onKeyDown={(e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          open();
        }
      }}
    >
      <h3 className="match-names">
        {match.near}
        <span className="match-v"> v </span>
        {match.far}
      </h3>
      <span className="match-status">
        {match.status === 'live' && <span className="match-dot" aria-hidden="true" />}
        {label}
      </span>
      <span className="baseline match-rule" aria-hidden="true" />
    </article>
  );
}
