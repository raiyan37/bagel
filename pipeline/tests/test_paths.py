import pytest

from horizon.paths import MatchPaths, validate_match_id


def test_match_paths_layout(tmp_path):
    paths = MatchPaths.for_match("demo-1", data_root=tmp_path).ensure()
    assert paths.root == tmp_path / "demo-1"
    assert paths.root.is_dir()
    assert paths.source_video.name == "source.mp4"
    assert paths.calibration.name == "calibration.json"
    assert paths.pov_video("near").name == "pov_near.mp4"
    assert paths.free_cam_video.name == "free_cam.mp4"


@pytest.mark.parametrize("bad", ["", "Demo", "../x", "a b", "x" * 65, "-lead"])
def test_invalid_match_ids_are_rejected(bad):
    with pytest.raises(ValueError):
        validate_match_id(bad)
