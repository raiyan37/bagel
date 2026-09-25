import { useEffect, useState, useRef } from 'react';
import './processing-glass.css';

interface Node {
  id: number;
  x: number;
  y: number;
  layer: number;
}

interface Connection {
  from: Node;
  to: Node;
}

export function ProcessingAnimation() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationRef = useRef<number>(0);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Set canvas size
    const resize = () => {
      const rect = canvas.getBoundingClientRect();
      canvas.width = rect.width * window.devicePixelRatio;
      canvas.height = rect.height * window.devicePixelRatio;
      ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
    };
    resize();

    // Generate neural network nodes - fewer, more spaced
    const layers = [3, 5, 6, 5, 3];
    const nodes: Node[] = [];
    const canvasWidth = canvas.getBoundingClientRect().width;
    const canvasHeight = canvas.getBoundingClientRect().height;
    const layerSpacing = canvasWidth / (layers.length + 1);

    layers.forEach((nodeCount, layerIndex) => {
      const x = layerSpacing * (layerIndex + 1);
      const nodeSpacing = canvasHeight / (nodeCount + 1);

      for (let i = 0; i < nodeCount; i++) {
        nodes.push({
          id: nodes.length,
          x,
          y: nodeSpacing * (i + 1),
          layer: layerIndex,
        });
      }
    });

    // Generate connections between adjacent layers
    const connections: Connection[] = [];
    for (let i = 0; i < layers.length - 1; i++) {
      const currentLayerNodes = nodes.filter((n) => n.layer === i);
      const nextLayerNodes = nodes.filter((n) => n.layer === i + 1);

      currentLayerNodes.forEach((from) => {
        nextLayerNodes.forEach((to) => {
          // Connect about 50% of possible connections for cleaner look
          if (Math.random() > 0.5) {
            connections.push({ from, to });
          }
        });
      });
    }

    // Animation loop
    let time = 0;
    const animate = () => {
      time += 0.008; // Slower animation
      const rect = canvas.getBoundingClientRect();
      ctx.clearRect(0, 0, rect.width, rect.height);

      // Draw connections - subtle, minimal
      connections.forEach((conn, index) => {
        const waveOffset = (index * 0.15 + time * 1.5) % 1;

        ctx.beginPath();
        ctx.moveTo(conn.from.x, conn.from.y);
        ctx.lineTo(conn.to.x, conn.to.y);

        // Create gradient for data flow effect
        const gradient = ctx.createLinearGradient(
          conn.from.x,
          conn.from.y,
          conn.to.x,
          conn.to.y
        );

        // Subtle white pulse traveling along the line
        const pulseStart = Math.max(0, waveOffset - 0.15);
        const pulseEnd = Math.min(1, waveOffset + 0.15);

        gradient.addColorStop(0, 'rgba(255, 255, 255, 0.06)');
        gradient.addColorStop(pulseStart, 'rgba(255, 255, 255, 0.06)');
        gradient.addColorStop(waveOffset, 'rgba(255, 255, 255, 0.35)');
        gradient.addColorStop(pulseEnd, 'rgba(255, 255, 255, 0.06)');
        gradient.addColorStop(1, 'rgba(255, 255, 255, 0.06)');

        ctx.strokeStyle = gradient;
        ctx.lineWidth = 1;
        ctx.stroke();
      });

      // Draw nodes - minimal dots
      nodes.forEach((node, index) => {
        const pulse = Math.sin(time * 2 + index * 0.4) * 0.5 + 0.5;
        const baseSize = 2;
        const size = baseSize + pulse * 1.5;

        // Outer glow - very subtle
        ctx.beginPath();
        ctx.arc(node.x, node.y, size + 6, 0, Math.PI * 2);
        const glowGradient = ctx.createRadialGradient(
          node.x,
          node.y,
          0,
          node.x,
          node.y,
          size + 6
        );
        glowGradient.addColorStop(0, `rgba(255, 255, 255, ${0.15 * pulse})`);
        glowGradient.addColorStop(1, 'rgba(255, 255, 255, 0)');
        ctx.fillStyle = glowGradient;
        ctx.fill();

        // Core node
        ctx.beginPath();
        ctx.arc(node.x, node.y, size, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(255, 255, 255, ${0.4 + pulse * 0.4})`;
        ctx.fill();
      });

      animationRef.current = requestAnimationFrame(animate);
    };

    animate();

    window.addEventListener('resize', resize);

    return () => {
      cancelAnimationFrame(animationRef.current);
      window.removeEventListener('resize', resize);
    };
  }, []);

  return (
    <canvas ref={canvasRef} className="processing-canvas" />
  );
}

interface ProcessingStatusProps {
  progress: number;
  message: string;
}

export function ProcessingStatus({ progress, message }: ProcessingStatusProps) {
  const [displayedMessage, setDisplayedMessage] = useState('');
  const [charIndex, setCharIndex] = useState(0);

  useEffect(() => {
    setDisplayedMessage('');
    setCharIndex(0);
  }, [message]);

  useEffect(() => {
    if (charIndex < message.length) {
      const timeout = setTimeout(() => {
        setDisplayedMessage((prev) => prev + message[charIndex]);
        setCharIndex((prev) => prev + 1);
      }, 25);
      return () => clearTimeout(timeout);
    }
  }, [charIndex, message]);

  return (
    <div className="processing-status">
      <div className="processing-percentage">
        <span className="processing-percentage-value">{Math.round(progress)}</span>
      </div>

      <div className="processing-progress">
        <div className="processing-progress-track">
          <div
            className="processing-progress-fill"
            style={{ width: `${progress}%` }}
          />
          <div
            className="processing-progress-glow"
            style={{ left: `${progress}%` }}
          />
        </div>
      </div>

      <div className="processing-message">
        <span>{displayedMessage}</span>
        <span className="processing-cursor">|</span>
      </div>
    </div>
  );
}
