  1. Accessibility Toggle (bottom-right on all pages) with:
    - High Contrast mode - solid backgrounds, white borders, maximum text contrast
    - Reduced Motion mode - disables all animations
  2. Keyboard Navigation:
    - Skip link to main content
    - Game cards are keyboard focusable (Tab + Enter/Space)
    - Visible focus indicators
  3. Screen Reader Support:
    - ARIA labels on all interactive elements
    - Live region announcements (e.g., goal events)
    - Semantic HTML (<main>, <header>, <article>)
    - Decorative elements hidden from screen readers
  4. System Preference Detection:
    - Auto-detects prefers-reduced-motion
    - Auto-detects prefers-contrast: more
    - Settings persist in localStorage
