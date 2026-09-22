import numpy as np
import pytest

from horizon.cli import main
from horizon.video import H264Writer, iter_frames, load_meta, probe, read_frame, transcode_clip


def write_color_clip(path, colors, size=(64, 48), fps=10.0):
    w, h = size
    with H264Writer(path, w, h, fps) as writer:
        for color in colors:
            writer.write(np.full((h, w, 3), color, dtype=np.uint8))


def test_writer_roundtrip(tmp_path):
    path = tmp_path / "clip.mp4"
    write_color_clip(path, [(200, 30, 30), (30, 200, 30), (30, 30, 200)] * 4)
    info = probe(path)
    assert (info.width, info.height) == (64, 48)
    assert info.fps == pytest.approx(10.0)
    frames = list(iter_frames(path))
    assert [i for i, _ in frames] == list(range(12))
    assert np.abs(frames[1][1].reshape(-1, 3).mean(axis=0) - (30, 200, 30)).max() < 12
    assert np.abs(read_frame(path, 2).reshape(-1, 3).mean(axis=0) - (30, 30, 200)).max() < 12


def test_writer_rejects_odd_sizes_and_bad_frames(tmp_path):
    with pytest.raises(ValueError):
        H264Writer(tmp_path / "odd.mp4", 63, 48, 10.0)
    with H264Writer(tmp_path / "ok.mp4", 64, 48, 10.0) as writer:
        with pytest.raises(ValueError):
            writer.write(np.zeros((48, 64), dtype=np.uint8))
        writer.write(np.zeros((48, 64, 3), dtype=np.uint8))


def test_transcode_trims_and_downscales(tmp_path):
    src = tmp_path / "raw.mp4"
    write_color_clip(src, [(i * 10, 100, 100) for i in range(20)], size=(128, 96), fps=10.0)
    info = transcode_clip(src, tmp_path / "out.mp4", start_s=0.5, duration_s=1.0, max_height=48)
    assert (info.width, info.height, info.frame_count) == (64, 48, 10)
    first = read_frame(tmp_path / "out.mp4", 0)
    assert abs(float(first[..., 0].mean()) - 50) < 12  # source frame 5 had red = 50


def test_init_command_writes_source_and_meta(tmp_path, monkeypatch):
    monkeypatch.setenv("HORIZON_DATA_ROOT", str(tmp_path / "data"))
    src = tmp_path / "raw.mp4"
    write_color_clip(src, [(90, 90, 90)] * 6, size=(64, 48), fps=12.0)
    assert main(["init", str(src), "--match-id", "unit"]) == 0
    assert (tmp_path / "data" / "unit" / "source.mp4").is_file()
    meta = load_meta(tmp_path / "data" / "unit" / "meta.json")
    assert (meta.width, meta.height, meta.frame_count) == (64, 48, 6)
    assert meta.fps == pytest.approx(12.0)
