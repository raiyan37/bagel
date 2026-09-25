import { useState, useEffect } from 'react';
import type { Commentary } from '../../types';
import './styles.css';

interface CommentaryOverlayProps {
  commentaries: Commentary[];
  autoRotate?: boolean;
  rotateInterval?: number;
}

export function CommentaryOverlay({
  commentaries,
  autoRotate = true,
  rotateInterval = 5000,
}: CommentaryOverlayProps) {
  const [currentIndex, setCurrentIndex] = useState(0);
  const [isVisible, setIsVisible] = useState(true);

  useEffect(() => {
    if (!autoRotate || commentaries.length <= 1) return;

    const interval = setInterval(() => {
      setIsVisible(false);
      setTimeout(() => {
        setCurrentIndex((prev) => (prev + 1) % commentaries.length);
        setIsVisible(true);
      }, 300);
    }, rotateInterval);

    return () => clearInterval(interval);
  }, [autoRotate, rotateInterval, commentaries.length]);

  if (commentaries.length === 0) return null;

  const current = commentaries[currentIndex];

  return (
    <div className={`commentary-overlay glass glass-noise ${isVisible ? 'commentary-visible' : 'commentary-hidden'}`}>
      <span className="commentary-time">{current.time}</span>
      <p className="commentary-text">"{current.text}"</p>
    </div>
  );
}
