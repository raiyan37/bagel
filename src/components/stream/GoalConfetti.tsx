import { useEffect, useState } from 'react';
import './goal-confetti.css';

interface Particle {
  id: number;
  x: number;
  y: number;
  size: number;
  delay: number;
  duration: number;
  angle: number;
  distance: number;
  opacity: number;
}

interface GoalConfettiProps {
  active: boolean;
}

export function GoalConfetti({ active }: GoalConfettiProps) {
  const [particles, setParticles] = useState<Particle[]>([]);

  useEffect(() => {
    if (active) {
      // Generate particles bursting from score tracker area
      const newParticles: Particle[] = [];
      for (let i = 0; i < 32; i++) {
        const angle = (Math.PI * 2 * i) / 32 + (Math.random() - 0.5) * 0.4;
        newParticles.push({
          id: i,
          x: 50, // Center horizontally (where score tracker is)
          y: 8,  // Near top where score tracker is
          size: 3 + Math.random() * 3,
          delay: Math.random() * 0.2,
          duration: 1.4 + Math.random() * 0.8,
          angle: angle,
          distance: 80 + Math.random() * 120,
          opacity: 0.5 + Math.random() * 0.5,
        });
      }
      setParticles(newParticles);

      // Clear after animation
      const timer = setTimeout(() => setParticles([]), 2500);
      return () => clearTimeout(timer);
    } else {
      setParticles([]);
    }
  }, [active]);

  if (particles.length === 0) return null;

  return (
    <div className="goal-confetti">
      {particles.map((particle) => (
        <div
          key={particle.id}
          className="goal-confetti-particle"
          style={{
            left: `${particle.x}%`,
            top: `${particle.y}%`,
            width: particle.size,
            height: particle.size,
            opacity: particle.opacity,
            animationDelay: `${particle.delay}s`,
            animationDuration: `${particle.duration}s`,
            '--end-x': `${Math.cos(particle.angle) * particle.distance}px`,
            '--end-y': `${Math.sin(particle.angle) * particle.distance}px`,
          } as React.CSSProperties}
        />
      ))}
    </div>
  );
}
