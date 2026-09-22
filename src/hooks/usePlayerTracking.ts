import { useState, useEffect, useCallback, useRef } from 'react';

interface TrackData {
  id: number;
  conf: number;
  bbox: [number, number, number, number]; // [x1, y1, x2, y2]
}

interface FrameData {
  frame: number;
  t: number;
  tracks: TrackData[];
}

interface TrackingData {
  videoW: number;
  videoH: number;
  fps: number;
  frames: FrameData[];
}

interface PlayerPosition {
  x: number; // percentage of video width (0-100)
  y: number; // percentage of video height (0-100)
  visible: boolean;
}

// Map track IDs to player IDs based on analysis of bruno.mov
// Mapping verified by visual position analysis
const TRACK_TO_PLAYER: Record<number, string> = {
  2: 'penalty-taker',     // Bruno Fernandes - left side (x≈150, y≈408), walking to spot
  1: 'referee',           // Anthony Taylor - left side behind Bruno (x≈183, y≈299), watching
  21: 'goalkeeper',       // David Raya - center-right (x≈901), frames 21-168
  103: 'goalkeeper',      // David Raya - after diving (x≈975), frames 220-245
  3: 'assistant-referee', // Assistant referee - far right low (x≈1238, y≈420)
};

export function usePlayerTracking(currentTime: number) {
  const [trackingData, setTrackingData] = useState<TrackingData | null>(null);
  const [positions, setPositions] = useState<Record<string, PlayerPosition>>({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const lastUpdateRef = useRef<number>(0);
  const lastKnownPositions = useRef<Record<string, PlayerPosition>>({});

  // Load tracking data
  useEffect(() => {
    fetch('/bruno_tracks.json')
      .then((res) => {
        if (!res.ok) throw new Error('Failed to load tracking data');
        return res.json();
      })
      .then((data: TrackingData) => {
        setTrackingData(data);
        setLoading(false);
      })
      .catch((err) => {
        setError(err.message);
        setLoading(false);
      });
  }, []);

  // Update positions based on current time
  const updatePositions = useCallback(() => {
    if (!trackingData) return;

    // Find the frame closest to current time
    const frames = trackingData.frames;
    let frameIndex = 0;

    for (let i = 0; i < frames.length; i++) {
      if (frames[i].t <= currentTime) {
        frameIndex = i;
      } else {
        break;
      }
    }

    const frame = frames[frameIndex];
    const newPositions: Record<string, PlayerPosition> = {};

    // Start with last known positions (to handle gaps in tracking)
    for (const playerId of new Set(Object.values(TRACK_TO_PLAYER))) {
      const lastKnown = lastKnownPositions.current[playerId];
      if (lastKnown) {
        // Keep last position but mark as not actively tracked
        newPositions[playerId] = { ...lastKnown, visible: true };
      } else {
        newPositions[playerId] = { x: 0, y: 0, visible: false };
      }
    }

    // Update positions for tracks in current frame
    for (const track of frame.tracks) {
      const playerId = TRACK_TO_PLAYER[track.id];
      if (playerId) {
        const [x1, y1, x2, y2] = track.bbox;
        // Calculate center of bounding box as percentage
        const centerX = ((x1 + x2) / 2 / trackingData.videoW) * 100;
        const centerY = ((y1 + y2) / 2 / trackingData.videoH) * 100;

        const pos = {
          x: centerX,
          y: centerY,
          visible: true,
        };
        newPositions[playerId] = pos;
        lastKnownPositions.current[playerId] = pos;
      }
    }

    setPositions(newPositions);
  }, [trackingData, currentTime]);

  // Throttle updates to ~30fps for performance
  useEffect(() => {
    const now = Date.now();
    if (now - lastUpdateRef.current < 33) return; // ~30fps
    lastUpdateRef.current = now;
    updatePositions();
  }, [updatePositions]);

  return { positions, loading, error };
}
