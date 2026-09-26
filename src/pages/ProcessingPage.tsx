import { useEffect, useRef, useState } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import { AsciiBagel } from '../components/processing/AsciiBagel';
import { AccessibilityToggle } from '../components/shared/AccessibilityToggle';
import '../components/processing/processing.css';

/* The pipeline has already produced these artifacts; this screen is the handover. */
const stages = [
  'Reading the clip',
  'Solving the court',
  'Estimating depth',
  'Building the point cloud',
  'Placing the player cameras',
  'Ready',
];

const thresholds = [0, 15, 35, 55, 75, 95];
const DURATION = 6000;
const STORAGE_KEY = 'processing_start_time';

function stageIndexFor(progress: number): number {
  let index = 0;
  for (let i = 0; i < thresholds.length; i += 1) {
    if (progress >= thresholds[i]) index = i;
  }
  return index;
}

export function ProcessingPage() {
  const navigate = useNavigate();
  const { gameId } = useParams<{ gameId: string }>();
  const [progress, setProgress] = useState(0);
  const startTimeRef = useRef<number>(0);
  const current = stageIndexFor(progress);

  useEffect(() => {
    const stored = sessionStorage.getItem(`${STORAGE_KEY}_${gameId}`);
    if (stored) {
      startTimeRef.current = parseInt(stored, 10);
    } else {
      startTimeRef.current = Date.now();
      sessionStorage.setItem(`${STORAGE_KEY}_${gameId}`, startTimeRef.current.toString());
    }

    const update = () => setProgress(Math.min(((Date.now() - startTimeRef.current) / DURATION) * 100, 100));
    update();
    const interval = setInterval(update, 50);
    return () => clearInterval(interval);
  }, [gameId]);

  useEffect(() => {
    if (progress < 100) return;
    const timeout = setTimeout(() => {
      sessionStorage.removeItem(`${STORAGE_KEY}_${gameId}`);
      navigate(`/stream/${gameId}`);
    }, 500);
    return () => clearTimeout(timeout);
  }, [progress, navigate, gameId]);

  return (
    <div className="loading" role="main" aria-label="Opening the match">
      <div className="loading-inner">
        <AsciiBagel />

        <div className="loading-stages">
          <span className="wordmark wordmark--small">bagel</span>
          <ol className="stage-list">
            {stages.map((stage, i) => (
              <li
                key={stage}
                className={`stage ${i < current ? 'stage--done' : ''} ${i === current ? 'stage--now' : ''}`}
              >
                {stage}
              </li>
            ))}
          </ol>
          <p className="sr-only" aria-live="polite">
            {stages[current]}
          </p>
          <span className="loading-rule" aria-hidden="true">
            <span className="loading-rule-fill" style={{ transform: `scaleX(${progress / 100})` }} />
          </span>
        </div>
      </div>

      <AccessibilityToggle />
    </div>
  );
}
