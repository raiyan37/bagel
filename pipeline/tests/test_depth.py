import numpy as np
import pytest
from synthetic import H, W, broadcast_camera

from horizon.depth import (
    DisparityAlignment,
    anchor_player_depth,
    court_fit_mask,
    fit_disparity_alignment,
    ground_depth_map,
)


def test_ground_depth_map_grows_towards_the_top_of_the_image():
    depth = ground_depth_map(broadcast_camera())
    assert depth.shape == (H, W)
    assert np.isfinite(depth).all()
    assert depth[-1, W // 2] < depth[H // 2, W // 2] < depth[0, W // 2]


def test_fit_recovers_affine_disparity_despite_outliers():
    cam = broadcast_camera()
    z = ground_depth_map(cam)
    scale, shift = 0.37, -0.004
    disparity = (1.0 / z - shift) / scale
    rng = np.random.default_rng(0)
    outliers = rng.random(disparity.shape) < 0.05
    disparity[outliers] += rng.uniform(0.5, 2.0, outliers.sum())
    mask = court_fit_mask(cam)
    fit = fit_disparity_alignment(disparity, z, mask)
    assert fit.scale == pytest.approx(scale, rel=1e-4)
    assert fit.shift == pytest.approx(shift, abs=1e-6)
    clean = mask & ~outliers
    assert np.allclose(fit.depth(disparity)[clean], z[clean], rtol=1e-4)


def test_fit_rejects_depth_like_output_and_tiny_masks():
    cam = broadcast_camera()
    z = ground_depth_map(cam)
    with pytest.raises(ValueError, match="disparity"):
        fit_disparity_alignment(z.copy(), z, court_fit_mask(cam))
    tiny = np.zeros((H, W), bool)
    tiny[400, 600:610] = True
    with pytest.raises(ValueError, match="ground pixels"):
        fit_disparity_alignment(1.0 / z, z, tiny)


def test_alignment_clamps_far_values():
    depth = DisparityAlignment(scale=1.0, shift=0.0, residual=0.0).depth(np.array([0.0, 0.5]), max_depth=120.0)
    assert depth.tolist() == [120.0, 2.0]


def test_court_fit_mask_excludes_net_and_players():
    cam = broadcast_camera()
    (u_court, v_court), (u_net, v_net) = cam.project(np.array([[0.0, -6.0, 0.0], [0.0, 0.0, 0.8]]))[0]
    players = np.zeros((H, W), bool)
    players[int(v_court) - 5 : int(v_court) + 5, int(u_court) - 5 : int(u_court) + 5] = True
    assert court_fit_mask(cam)[int(round(v_court)), int(round(u_court))]
    assert not court_fit_mask(cam)[int(round(v_net)), int(round(u_net))]
    assert not court_fit_mask(cam, exclude=players)[int(v_court), int(u_court)]


def test_anchor_player_depth_moves_feet_to_the_calibrated_depth():
    depth = np.zeros((40, 20))
    rows = np.arange(40, dtype=float)[:, None].repeat(20, axis=1)
    depth[:] = 5.0 + 0.01 * rows
    mask = np.zeros((40, 20), bool)
    mask[10:30, 5:15] = True
    depth[12, 8] = 15.0  # outlier inside the person
    anchored = anchor_player_depth(depth, mask, foot_depth=20.0, band=0.6)
    assert anchored.shape == (int(mask.sum()),)
    feet = anchored.reshape(20, 10)[-2:]
    assert np.allclose(feet, 20.0, atol=0.006)
    assert anchored.max() == pytest.approx(20.6)
    assert anchored.min() >= 19.4
