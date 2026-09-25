import type { ReactNode } from 'react';
import './styles.css';

interface GlassContainerProps {
  children: ReactNode;
  variant?: 'default' | 'light' | 'dark' | 'heavy';
  className?: string;
  withNoise?: boolean;
  withGlow?: boolean;
  interactive?: boolean;
  pill?: boolean;
  accentColor?: string;
  accentPosition?: 'left' | 'top' | 'none';
  onClick?: () => void;
}

export function GlassContainer({
  children,
  variant = 'default',
  className = '',
  withNoise = false,
  withGlow = false,
  interactive = false,
  pill = false,
  accentColor,
  accentPosition = 'none',
  onClick,
}: GlassContainerProps) {
  const variantClasses = {
    default: 'glass',
    light: 'glass-light',
    dark: 'glass-dark',
    heavy: 'glass-heavy',
  };

  const classes = [
    'glass-container',
    variantClasses[variant],
    withNoise && 'glass-noise',
    withGlow && 'glass-glow',
    interactive && 'glass-interactive',
    pill && 'glass-pill',
    accentPosition === 'left' && 'glass-accent-left',
    accentPosition === 'top' && 'glass-accent-top',
    className,
  ]
    .filter(Boolean)
    .join(' ');

  const style = accentColor
    ? ({ '--accent-color': accentColor } as React.CSSProperties)
    : undefined;

  return (
    <div className={classes} style={style} onClick={onClick}>
      {children}
    </div>
  );
}
