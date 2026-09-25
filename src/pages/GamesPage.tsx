import { GameCard, type Game } from '../components/games/GameCard';
import { AccessibilityToggle } from '../components/shared/AccessibilityToggle';
import '../components/games/games-glass.css';

// Sample games data
const games: Game[] = [
  {
    id: 'mun-ars-facup',
    homeTeam: {
      name: 'Manchester United',
      shortName: 'MUN',
      color: '#DA291C',
    },
    awayTeam: {
      name: 'Arsenal',
      shortName: 'ARS',
      color: '#EF0107',
    },
    competition: 'FA Cup',
    competitionShort: 'FA CUP',
    time: '17:30',
    status: 'live',
  },
  {
    id: 'liv-che-prem',
    homeTeam: {
      name: 'Liverpool',
      shortName: 'LIV',
      color: '#C8102E',
    },
    awayTeam: {
      name: 'Chelsea',
      shortName: 'CHE',
      color: '#034694',
    },
    competition: 'Premier League',
    competitionShort: 'PREM',
    time: '20:00',
    status: 'live',
  },
  {
    id: 'mci-tot-prem',
    homeTeam: {
      name: 'Manchester City',
      shortName: 'MCI',
      color: '#6CABDD',
    },
    awayTeam: {
      name: 'Tottenham Hotspur',
      shortName: 'TOT',
      color: '#132257',
    },
    competition: 'Premier League',
    competitionShort: 'PREM',
    time: '15:00',
    status: 'upcoming',
  },
  {
    id: 'bar-rma-laliga',
    homeTeam: {
      name: 'Barcelona',
      shortName: 'BAR',
      color: '#A50044',
    },
    awayTeam: {
      name: 'Real Madrid',
      shortName: 'RMA',
      color: '#FEBE10',
    },
    competition: 'La Liga',
    competitionShort: 'LA LIGA',
    time: '20:00',
    status: 'upcoming',
  },
  {
    id: 'psg-bay-ucl',
    homeTeam: {
      name: 'Paris Saint-Germain',
      shortName: 'PSG',
      color: '#004170',
    },
    awayTeam: {
      name: 'Bayern Munich',
      shortName: 'BAY',
      color: '#DC052D',
    },
    competition: 'Champions League',
    competitionShort: 'UCL',
    time: 'Tomorrow',
    status: 'upcoming',
  },
  {
    id: 'juv-int-seria',
    homeTeam: {
      name: 'Juventus',
      shortName: 'JUV',
      color: '#000000',
    },
    awayTeam: {
      name: 'Inter Milan',
      shortName: 'INT',
      color: '#010E80',
    },
    competition: 'Serie A',
    competitionShort: 'SERIE A',
    time: 'FT 2-1',
    status: 'replay',
  },
];

export function GamesPage() {
  const liveCount = games.filter((g) => g.status === 'live').length;

  return (
    <div className="games-page">
      {/* Skip link for keyboard navigation */}
      <a href="#main-content" className="skip-link">
        Skip to main content
      </a>

      <header className="games-header" role="banner">
        <div className="games-brand">
          <span className="games-logo" aria-hidden="true">ph</span>
          <h1 className="games-brand-title">Project Horizon</h1>
        </div>
        <p className="games-tagline">See the game through<br />the eyes of those who play it</p>

        {liveCount > 0 && (
          <div className="games-header-live" role="status" aria-live="polite">
            <span className="games-header-live-dot" aria-hidden="true" />
            <span className="games-header-live-text">Live now</span>
            <span className="games-header-live-count">/ {liveCount} matches</span>
          </div>
        )}
      </header>

      <main id="main-content" role="main">
        <h2 className="sr-only">Available Matches</h2>
        <div className="games-grid" role="list" aria-label="Football matches">
          {games.map((game) => (
            <GameCard key={game.id} game={game} />
          ))}
        </div>
      </main>

      <AccessibilityToggle />
    </div>
  );
}
