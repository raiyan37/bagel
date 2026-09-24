import { GameCard, type Game } from '../components/games/GameCard';
import { AccessibilityToggle } from '../components/shared/AccessibilityToggle';
import { useMatchIndex } from '../hooks/useMatchData';
import type { MatchSummary } from '../lib/match';
import '../components/games/games-glass.css';
import '../components/tennis/tennis.css';

function initials(name: string): string {
  return name
    .split(/\s+/)
    .map((part) => part[0] ?? '')
    .join('')
    .slice(0, 3)
    .toUpperCase();
}

function toGame(match: MatchSummary): Game {
  return {
    id: match.id,
    homeTeam: { name: match.near, shortName: initials(match.near), color: '#3B82F6' },
    awayTeam: { name: match.far, shortName: initials(match.far), color: '#F97316' },
    competition: match.competition,
    competitionShort: match.competition.toUpperCase(),
    time: '',
    status: match.status,
  };
}

export function GamesPage() {
  const { matches, loading, error } = useMatchIndex();

  return (
    <div className="games-page">
      <a href="#main-content" className="skip-link">
        Skip to main content
      </a>

      <header className="games-header" role="banner">
        <div className="games-brand">
          <span className="games-logo" aria-hidden="true">ph</span>
          <h1 className="games-brand-title">Project Horizon · Tennis</h1>
        </div>
        <p className="games-tagline">See the rally through<br />the eyes of those who play it</p>
      </header>

      <main id="main-content" role="main">
        <h2 className="sr-only">Available Matches</h2>
        {loading ? (
          <p className="games-empty" role="status">Loading matches…</p>
        ) : matches.length === 0 ? (
          error !== null ? (
            <p className="games-empty" role="alert">Could not read the match list: {error}</p>
          ) : (
            <p className="games-empty" role="status">
              No processed matches yet. In <code>pipeline\</code> run{' '}
              <code>.venv\Scripts\python -m horizon all path\to\clip.mp4 --match-id demo</code>.
            </p>
          )
        ) : (
          <div className="games-grid" role="list" aria-label="Tennis matches">
            {matches.map((match) => (
              <GameCard key={match.id} game={toGame(match)} />
            ))}
          </div>
        )}
      </main>

      <AccessibilityToggle />
    </div>
  );
}
