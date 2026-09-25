# track_video.py
import json
from pathlib import Path

import cv2
from ultralytics import YOLO

VIDEO_PATH = "../src/assets/bruno.mov"
OUT_JSON = "../public/bruno_tracks.json"

# Use a fast model first; upgrade later if needed
MODEL_NAME = "yolov8n.pt"  # or yolov8s.pt for better accuracy

def main():
    model = YOLO(MODEL_NAME)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    video_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames_out = []
    frame_idx = 0

    # Tracking state is kept inside Ultralytics when persist=True
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # track() runs detection + tracking
        results = model.track(
            source=frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=0.25,
            iou=0.45,
            classes=[0],  # person class in COCO
            verbose=False,
        )

        r = results[0]
        tracks = []

        # r.boxes contains boxes; r.boxes.id contains track IDs (when available)
        if r.boxes is not None and r.boxes.xyxy is not None:
            xyxy = r.boxes.xyxy.cpu().numpy()  # (N,4)
            confs = r.boxes.conf.cpu().numpy() if r.boxes.conf is not None else None
            ids = r.boxes.id.cpu().numpy() if r.boxes.id is not None else None

            for i in range(len(xyxy)):
                x1, y1, x2, y2 = xyxy[i].tolist()
                conf = float(confs[i]) if confs is not None else 0.0
                track_id = int(ids[i]) if ids is not None else -1

                # Skip untracked detections if you want stable IDs only
                if track_id == -1:
                    continue

                tracks.append({
                    "id": track_id,
                    "conf": conf,
                    "bbox": [x1, y1, x2, y2],
                })

        t = frame_idx / fps
        frames_out.append({
            "frame": frame_idx,
            "t": t,
            "tracks": tracks,
        })

        frame_idx += 1

    cap.release()

    payload = {
        "videoW": video_w,
        "videoH": video_h,
        "fps": fps,
        "frames": frames_out,
    }

    Path(OUT_JSON).write_text(json.dumps(payload))
    print(f"Wrote {OUT_JSON} with {len(frames_out)} frames")

if __name__ == "__main__":
    main()