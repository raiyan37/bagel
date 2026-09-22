import { useEffect, useRef, forwardRef, useImperativeHandle } from 'react';
import type { ReactNode } from 'react';
import './styles.css';

declare global {
  interface Window {
    YT: typeof YT;
    onYouTubeIframeAPIReady: () => void;
  }
}

export interface StreamContainerRef {
  play: () => void;
  pause: () => void;
  seekTo: (seconds: number) => void;
  getCurrentTime: () => number;
  getDuration: () => number;
  getPlayerState: () => number;
}

interface StreamContainerProps {
  children: ReactNode;
  youtubeId?: string;
  videoSrc?: string;
  onTimeUpdate?: (currentTime: number, duration: number) => void;
  onStateChange?: (isPlaying: boolean) => void;
}

export const StreamContainer = forwardRef<StreamContainerRef, StreamContainerProps>(
  ({ children, youtubeId, videoSrc, onTimeUpdate, onStateChange }, ref) => {
    const playerRef = useRef<YT.Player | null>(null);
    const videoRef = useRef<HTMLVideoElement | null>(null);
    const containerRef = useRef<HTMLDivElement>(null);
    const intervalRef = useRef<number | null>(null);

    useImperativeHandle(ref, () => ({
      play: () => {
        if (videoSrc && videoRef.current) {
          videoRef.current.play();
        } else {
          playerRef.current?.playVideo();
        }
      },
      pause: () => {
        if (videoSrc && videoRef.current) {
          videoRef.current.pause();
        } else {
          playerRef.current?.pauseVideo();
        }
      },
      seekTo: (seconds: number) => {
        if (videoSrc && videoRef.current) {
          videoRef.current.currentTime = seconds;
        } else {
          playerRef.current?.seekTo(seconds, true);
        }
      },
      getCurrentTime: () => {
        if (videoSrc && videoRef.current) {
          return videoRef.current.currentTime;
        }
        return playerRef.current?.getCurrentTime() ?? 0;
      },
      getDuration: () => {
        if (videoSrc && videoRef.current) {
          return videoRef.current.duration;
        }
        return playerRef.current?.getDuration() ?? 0;
      },
      getPlayerState: () => {
        if (videoSrc && videoRef.current) {
          return videoRef.current.paused ? 2 : 1;
        }
        return playerRef.current?.getPlayerState() ?? -1;
      },
    }));

    // Handle native video element
    useEffect(() => {
      if (!videoSrc || !videoRef.current) return;

      const video = videoRef.current;

      const handleTimeUpdate = () => {
        onTimeUpdate?.(video.currentTime, video.duration);
      };

      const handlePlay = () => onStateChange?.(true);
      const handlePause = () => onStateChange?.(false);

      video.addEventListener('timeupdate', handleTimeUpdate);
      video.addEventListener('play', handlePlay);
      video.addEventListener('pause', handlePause);

      return () => {
        video.removeEventListener('timeupdate', handleTimeUpdate);
        video.removeEventListener('play', handlePlay);
        video.removeEventListener('pause', handlePause);
      };
    }, [videoSrc, onTimeUpdate, onStateChange]);

    useEffect(() => {
      if (!youtubeId) return;

      // Load YouTube IFrame API
      const tag = document.createElement('script');
      tag.src = 'https://www.youtube.com/iframe_api';
      const firstScriptTag = document.getElementsByTagName('script')[0];
      firstScriptTag.parentNode?.insertBefore(tag, firstScriptTag);

      window.onYouTubeIframeAPIReady = () => {
        playerRef.current = new window.YT.Player('youtube-player', {
          videoId: youtubeId,
          playerVars: {
            autoplay: 1,
            mute: 1,
            controls: 0,
            showinfo: 0,
            rel: 0,
            modestbranding: 1,
            playsinline: 1,
            loop: 1,
            playlist: youtubeId,
          },
          events: {
            onReady: (event) => {
              event.target.playVideo();
              // Start time update interval
              intervalRef.current = window.setInterval(() => {
                if (playerRef.current) {
                  const currentTime = playerRef.current.getCurrentTime();
                  const duration = playerRef.current.getDuration();
                  onTimeUpdate?.(currentTime, duration);
                }
              }, 1000);
            },
            onStateChange: (event) => {
              onStateChange?.(event.data === window.YT.PlayerState.PLAYING);
            },
          },
        });
      };

      // If API already loaded
      if (window.YT && window.YT.Player) {
        window.onYouTubeIframeAPIReady();
      }

      return () => {
        if (intervalRef.current) {
          clearInterval(intervalRef.current);
        }
        playerRef.current?.destroy();
      };
    }, [youtubeId, onTimeUpdate, onStateChange]);

    return (
      <div className="stream-container" ref={containerRef}>
        <div className="stream-video-wrapper">
          {youtubeId ? (
            <div id="youtube-player" className="stream-video youtube-embed" />
          ) : videoSrc ? (
            <video
              ref={videoRef}
              className="stream-video"
              src={videoSrc}
              autoPlay
              muted
              loop
              playsInline
            />
          ) : (
            <div className="stream-video-placeholder">
              <div className="field-lines">
                <div className="field-center-circle" />
                <div className="field-center-line" />
                <div className="field-penalty-left" />
                <div className="field-penalty-right" />
                <div className="field-goal-left" />
                <div className="field-goal-right" />
              </div>
            </div>
          )}
          <div className="stream-overlay">{children}</div>
        </div>
      </div>
    );
  }
);

StreamContainer.displayName = 'StreamContainer';
