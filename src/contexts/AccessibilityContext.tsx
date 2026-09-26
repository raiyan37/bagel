import { createContext, useContext, useState, useEffect, type ReactNode } from 'react';

interface AccessibilitySettings {
  highContrast: boolean;
  reducedMotion: boolean;
}

interface AccessibilityContextType {
  settings: AccessibilitySettings;
  focusMode: boolean;
  toggleHighContrast: () => void;
  toggleReducedMotion: () => void;
  toggleFocusMode: () => void;
  announce: (message: string, priority?: 'polite' | 'assertive') => void;
}

const AccessibilityContext = createContext<AccessibilityContextType | null>(null);

const STORAGE_KEY = 'accessibility_settings';

export function AccessibilityProvider({ children }: { children: ReactNode }) {
  const [focusMode, setFocusMode] = useState(false);
  const [settings, setSettings] = useState<AccessibilitySettings>(() => {
    // Check localStorage for saved preferences
    const saved = localStorage.getItem(STORAGE_KEY);
    if (saved) {
      return JSON.parse(saved);
    }
    // Check system preferences
    return {
      highContrast: window.matchMedia('(prefers-contrast: more)').matches,
      reducedMotion: window.matchMedia('(prefers-reduced-motion: reduce)').matches,
    };
  });

  // Sync with system preferences
  useEffect(() => {
    const contrastQuery = window.matchMedia('(prefers-contrast: more)');
    const motionQuery = window.matchMedia('(prefers-reduced-motion: reduce)');

    const handleContrastChange = (e: MediaQueryListEvent) => {
      setSettings(prev => ({ ...prev, highContrast: e.matches }));
    };
    const handleMotionChange = (e: MediaQueryListEvent) => {
      setSettings(prev => ({ ...prev, reducedMotion: e.matches }));
    };

    contrastQuery.addEventListener('change', handleContrastChange);
    motionQuery.addEventListener('change', handleMotionChange);

    return () => {
      contrastQuery.removeEventListener('change', handleContrastChange);
      motionQuery.removeEventListener('change', handleMotionChange);
    };
  }, []);

  // Persist settings
  useEffect(() => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(settings));
  }, [settings]);

  // Apply classes to document
  useEffect(() => {
    const root = document.documentElement;
    root.classList.toggle('high-contrast', settings.highContrast);
    root.classList.toggle('reduced-motion', settings.reducedMotion);
  }, [settings]);

  // Focus mode strips the chrome down to the broadcast and the player views.
  // Deliberately not persisted: a reload should never strand you without controls.
  useEffect(() => {
    document.documentElement.classList.toggle('focus-mode', focusMode);
  }, [focusMode]);

  // Q is the way back out, since the buttons are hidden.
  useEffect(() => {
    if (!focusMode) return;

    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key.toLowerCase() !== 'q' || e.metaKey || e.ctrlKey || e.altKey) return;

      const target = e.target as HTMLElement | null;
      if (target?.isContentEditable || ['INPUT', 'TEXTAREA', 'SELECT'].includes(target?.tagName ?? '')) return;

      e.preventDefault();
      setFocusMode(false);
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [focusMode]);

  const toggleHighContrast = () => {
    setSettings(prev => ({ ...prev, highContrast: !prev.highContrast }));
  };

  const toggleFocusMode = () => {
    setFocusMode((prev) => !prev);
  };

  const toggleReducedMotion = () => {
    setSettings(prev => ({ ...prev, reducedMotion: !prev.reducedMotion }));
  };

  // Screen reader announcements
  const announce = (message: string, priority: 'polite' | 'assertive' = 'polite') => {
    const el = document.getElementById(`aria-live-${priority}`);
    if (el) {
      el.textContent = '';
      // Small delay to ensure screen reader picks up the change
      setTimeout(() => {
        el.textContent = message;
      }, 50);
    }
  };

  return (
    <AccessibilityContext.Provider value={{ settings, focusMode, toggleHighContrast, toggleReducedMotion, toggleFocusMode, announce }}>
      {/* Live regions for screen reader announcements */}
      <div
        id="aria-live-polite"
        aria-live="polite"
        aria-atomic="true"
        className="sr-only"
      />
      <div
        id="aria-live-assertive"
        aria-live="assertive"
        aria-atomic="true"
        className="sr-only"
      />
      {children}
    </AccessibilityContext.Provider>
  );
}

// The hook belongs with the context it reads; moving it to its own file would only relocate the
// same non-component export, so fast refresh is opted out of here instead.
// eslint-disable-next-line react-refresh/only-export-components
export function useAccessibility() {
  const context = useContext(AccessibilityContext);
  if (!context) {
    throw new Error('useAccessibility must be used within AccessibilityProvider');
  }
  return context;
}
