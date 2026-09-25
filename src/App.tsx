import { Routes, Route } from 'react-router-dom';
import { GamesPage } from './pages/GamesPage';
import { ProcessingPage } from './pages/ProcessingPage';
import { StreamPage } from './pages/StreamPage';

function App() {
  return (
    <Routes>
      <Route path="/" element={<GamesPage />} />
      <Route path="/processing/:gameId" element={<ProcessingPage />} />
      <Route path="/stream/:gameId" element={<StreamPage />} />
    </Routes>
  );
}

export default App;
