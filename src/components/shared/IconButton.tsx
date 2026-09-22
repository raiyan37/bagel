import type { ReactNode } from 'react';
import './styles.css';

interface IconButtonProps {
  children: ReactNode;
  onClick?: () => void;
  isActive?: boolean;
  size?: 'sm' | 'md' | 'lg';
  variant?: 'ghost' | 'glass' | 'solid';
  className?: string;
  title?: string;
}

export function IconButton({
  children,
  onClick,
  isActive = false,
  size = 'md',
  variant = 'glass',
  className = '',
  title,
}: IconButtonProps) {
  const sizeClasses = {
    sm: 'icon-btn-sm',
    md: 'icon-btn-md',
    lg: 'icon-btn-lg',
  };

  const variantClasses = {
    ghost: 'icon-btn-ghost',
    glass: 'icon-btn-glass',
    solid: 'icon-btn-solid',
  };

  return (
    <button
      className={`icon-btn ${sizeClasses[size]} ${variantClasses[variant]} ${isActive ? 'icon-btn-active' : ''} ${className}`}
      onClick={onClick}
      title={title}
    >
      {children}
    </button>
  );
}
