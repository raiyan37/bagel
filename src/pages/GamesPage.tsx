import { CourtDiagram } from '../components/games/CourtDiagram';
import { MatchRow, type Match } from '../components/games/MatchRow';
import { AccessibilityToggle } from '../components/shared/AccessibilityToggle';
import { useMatchIndex } from '../hooks/useMatchData';
import type { MatchSummary } from '../lib/match';
import '../components/games/games.css';

function toMatch(summary: MatchSummary): Match {
  return { id: summary.id, near: summary.near, far: summary.far, status: summary.status, time: '' };
}

export function GamesPage() {
  const { matches, loading, error } = useMatchIndex();

  return (
    <div className="home">
      <a href="#main-content" className="skip-link">
        Skip to main content
      </a>

      <div className="home-inner">
        <header className="home-header">
          <span className="wordmark">bagel</span>
          <h1 className="home-title">Watch the point from inside it</h1>
          <p className="home-lede">
            bagel rebuilds a rally in three dimensions from a single broadcast camera. Open a match to
            follow it from either player&rsquo;s eyes, or put the camera wherever you want to stand.
          </p>
        </header>
      </div>

      <div className="home-court">
        <CourtDiagram />
      </div>

      <div className="home-inner">

        <main id="main-content">
          <h2 className="sr-only">Matches</h2>
          {loading ? (
            <p className="home-note" role="status">
              Reading the match list&hellip;
            </p>
          ) : matches.length === 0 ? (
            error !== null ? (
              <p className="home-note" role="alert">
                Could not read the match list: {error}
              </p>
            ) : (
              <p className="home-note" role="status">
                No matches yet. Process a clip to add the first one:
                <code>python -m horizon all path/to/clip.mp4 --match-id ao</code>
              </p>
            )
          ) : (
            <>
              <p className="home-count">
                {matches.length} {matches.length === 1 ? 'match' : 'matches'} ready
              </p>
              <div className="match-list" role="list" aria-label="Matches">
                <span className="baseline" aria-hidden="true" />
                {matches.map((summary) => (
                  <MatchRow key={summary.id} match={toMatch(summary)} />
                ))}
              </div>
            </>
          )}
        </main>
      </div>

      <AccessibilityToggle />
    </div>
  );
}
