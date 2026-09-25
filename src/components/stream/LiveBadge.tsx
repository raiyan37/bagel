import './live-glass.css';

export function LiveBadge() {
  return (
    <div className="live-glass">
      {/* Glass layers */}
      <div className="live-glass-bg" />
      <div className="live-glass-surface" />

      {/* Content */}
      <div className="live-glass-content">
        <div className="live-glass-pulse">
          <div className="live-glass-pulse-ring" />
          <div className="live-glass-pulse-dot" />
        </div>
        <span className="live-glass-text">LIVE</span>
      </div>
    </div>
  );
}
