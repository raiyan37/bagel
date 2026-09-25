export interface Team {
  name: string;
  shortName: string;
  flag: string;
  color: string;
}

export interface Match {
  id: string;
  homeTeam: Team;
  awayTeam: Team;
  homeScore: number;
  awayScore: number;
  time: string;
  isLive: boolean;
  competition: string;
}

export interface Player {
  id: string;
  name: string;
  number: number;
  position: string;
  team: 'home' | 'away';
  teamColor: string;
  avatar: string;
  stats: PlayerStats;
  fieldPosition: {
    x: number;
    y: number;
  };
}

export interface PlayerStats {
  passes: number;
  passAccuracy: number;
  shots: number;
  shotsOnTarget: number;
  tackles: number;
  distance: number;
  speed: number;
  sprints: number;
}

export interface Commentary {
  id: string;
  text: string;
  time: string;
  type: 'action' | 'goal' | 'card' | 'general';
}

export interface NavItem {
  id: string;
  label: string;
  icon?: string;
  isLive?: boolean;
  isActive?: boolean;
}

export interface VideoState {
  isPlaying: boolean;
  currentTime: number;
  duration: number;
  volume: number;
  isMuted: boolean;
  isFullscreen: boolean;
}
