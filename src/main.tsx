import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { BrowserRouter } from 'react-router-dom'
import { AccessibilityProvider } from './contexts/AccessibilityContext'
import './index.css'
import App from './App.tsx'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <BrowserRouter>
      <AccessibilityProvider>
        <App />
      </AccessibilityProvider>
    </BrowserRouter>
  </StrictMode>,
)
