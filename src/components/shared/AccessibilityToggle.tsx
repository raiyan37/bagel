import { useLocation } from 'react-router-dom';
import { useAccessibility } from '../../contexts/AccessibilityContext';
import './accessibility.css';

export function AccessibilityToggle() {
  const { settings, focusMode, toggleReducedMotion, toggleFocusMode, announce } = useAccessibility();
  const { pathname } = useLocation();

  // Focus mode only means anything over a broadcast.
  const showFocusButton = pathname.startsWith('/stream');

  const handleFocusToggle = () => {
    toggleFocusMode();
    announce(focusMode ? 'Normal view restored' : 'Focus mode on. Press Q to return to the normal view.');
  };

  const handleReducedMotionToggle = () => {
    toggleReducedMotion();
    announce(settings.reducedMotion ? 'Animations enabled' : 'Animations reduced');
  };

  return (
    <div className="accessibility-toggle" role="group" aria-label="Accessibility options">
      {showFocusButton && (
        <button
          className={`accessibility-btn ${focusMode ? 'accessibility-btn--active' : ''}`}
          onClick={handleFocusToggle}
          aria-pressed={focusMode}
          title="Hide every control — press Q to return"
        >
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true">
            <path d="M3 8V5a2 2 0 0 1 2-2h3" strokeLinecap="round" strokeLinejoin="round" />
            <path d="M16 3h3a2 2 0 0 1 2 2v3" strokeLinecap="round" strokeLinejoin="round" />
            <path d="M21 16v3a2 2 0 0 1-2 2h-3" strokeLinecap="round" strokeLinejoin="round" />
            <path d="M8 21H5a2 2 0 0 1-2-2v-3" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
          <span className="accessibility-btn-label">Focus</span>
        </button>
      )}

      <button
        className={`accessibility-btn ${settings.reducedMotion ? 'accessibility-btn--active' : ''}`}
        onClick={handleReducedMotionToggle}
        aria-pressed={settings.reducedMotion}
        title="Toggle reduced motion"
      >
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true">
          {settings.reducedMotion ? (
            <path d="M5 12h14" strokeLinecap="round" />
          ) : (
            <>
              <path d="M5 12h14" strokeLinecap="round" />
              <path d="M12 5l7 7-7 7" strokeLinecap="round" strokeLinejoin="round" />
            </>
          )}
        </svg>
        <span className="accessibility-btn-label">Motion</span>
      </button>
    </div>
  );
}
