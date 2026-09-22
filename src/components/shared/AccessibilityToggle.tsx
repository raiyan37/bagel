import { useAccessibility } from '../../contexts/AccessibilityContext';
import './accessibility.css';

export function AccessibilityToggle() {
  const { settings, toggleHighContrast, toggleReducedMotion, announce } = useAccessibility();

  const handleHighContrastToggle = () => {
    toggleHighContrast();
    announce(settings.highContrast ? 'High contrast mode disabled' : 'High contrast mode enabled');
  };

  const handleReducedMotionToggle = () => {
    toggleReducedMotion();
    announce(settings.reducedMotion ? 'Animations enabled' : 'Animations reduced');
  };

  return (
    <div className="accessibility-toggle" role="group" aria-label="Accessibility options">
      <button
        className={`accessibility-btn ${settings.highContrast ? 'accessibility-btn--active' : ''}`}
        onClick={handleHighContrastToggle}
        aria-pressed={settings.highContrast}
        title="Toggle high contrast mode"
      >
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true">
          <circle cx="12" cy="12" r="10" />
          <path d="M12 2v20" />
          <path d="M12 2a10 10 0 0 1 0 20" fill="currentColor" />
        </svg>
        <span className="accessibility-btn-label">Contrast</span>
      </button>

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
