import numpy as np

from horizon.tracking import Detection, Detections, detections_from_arrays


def square(x0, y0, size):
    return np.array([[x0, y0], [x0 + size, y0], [x0 + size, y0 + size], [x0, y0 + size]], dtype=np.float32)


def test_detections_from_arrays_converts_and_drops_untracked():
    xyxy = np.array([[10, 20, 30, 60], [100, 50, 120, 90]], dtype=np.float32)
    conf = np.array([0.9, 0.4])
    ids = np.array([7.0, 3.0])
    dets = detections_from_arrays(xyxy, conf, ids, [square(10, 20, 20), square(100, 50, 20)])
    assert [d.track_id for d in dets] == [7, 3]
    assert dets[0].bbox == (10.0, 20.0, 30.0, 60.0)
    assert dets[0].foot == (20.0, 60.0)
    assert dets[0].head == (20.0, 20.0)
    assert len(dets[0].polygon) == 4
    assert detections_from_arrays(xyxy, conf, None, None) == []


def test_polygon_is_simplified_and_rasterised():
    t = np.linspace(0, 2 * np.pi, 400, endpoint=False)
    circle = np.stack([50 + 20 * np.cos(t), 50 + 20 * np.sin(t)], axis=1)
    det = detections_from_arrays(np.array([[30, 30, 70, 70]]), np.array([0.8]), np.array([1]), [circle])[0]
    assert 8 <= len(det.polygon) < 100
    mask = det.mask(100, 100)
    assert mask.dtype == bool
    assert abs(int(mask.sum()) - int(np.pi * 20**2)) < 120


def test_mask_falls_back_to_bbox_without_polygon():
    mask = Detection(track_id=1, conf=0.9, bbox=(10.0, 10.0, 20.0, 30.0)).mask(40, 40)
    assert mask[10:31, 10:21].all()
    assert mask.sum() == 11 * 21


def test_detections_json_roundtrip(tmp_path):
    frames = [[Detection(1, 0.9, (1.0, 2.0, 3.0, 4.0), ((1.0, 2.0), (3.0, 2.0), (3.0, 4.0)))], []]
    Detections(width=64, height=48, fps=25.0, frames=frames).save(tmp_path / "d.json")
    again = Detections.load(tmp_path / "d.json")
    assert (again.width, again.height, again.fps, len(again.frames)) == (64, 48, 25.0, 2)
    assert again.frames[0][0] == frames[0][0]
    assert again.by_track(0, 1) == frames[0][0]
    assert again.by_track(1, 1) is None
