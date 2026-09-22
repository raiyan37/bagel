import './styles.css';

interface LiveIndicatorProps {
  showText?: boolean;
  size?: 'sm' | 'md' | 'lg';
  className?: string;
}

export function LiveIndicator({
  showText = true,
  size = 'md',
  className = '',
}: LiveIndicatorProps) {
  const sizeClasses = {
    sm: 'live-indicator-sm',
    md: 'live-indicator-md',
    lg: 'live-indicator-lg',
  };

  return (
    <div className={`live-indicator ${sizeClasses[size]} ${className}`}>
      <span className="live-dot" />
      {showText && <span className="live-text">LIVE</span>}
    </div>
  );
}
