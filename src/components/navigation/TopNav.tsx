import { useState } from 'react';
import { LiveIndicator } from '../shared/LiveIndicator';
import './styles.css';

interface NavItem {
  id: string;
  label: string;
  isLive?: boolean;
}

const navItems: NavItem[] = [
  { id: 'live', label: 'Live Match', isLive: true },
  { id: 'home', label: 'Home' },
  { id: 'football', label: 'Football' },
  { id: 'nba', label: 'NBA' },
  { id: 'esports', label: 'Esports' },
];

export function TopNav() {
  const [activeId, setActiveId] = useState('live');

  return (
    <nav className="top-nav glass glass-noise">
      <div className="top-nav-items">
        {navItems.map((item) => (
          <button
            key={item.id}
            className={`top-nav-item ${activeId === item.id ? 'top-nav-item-active' : ''}`}
            onClick={() => setActiveId(item.id)}
          >
            {item.isLive && <LiveIndicator showText={false} size="sm" />}
            <span>{item.label}</span>
          </button>
        ))}
      </div>
    </nav>
  );
}
