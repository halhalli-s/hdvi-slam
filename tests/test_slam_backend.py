"""Test the iSAM2 back-end produces a consistent trajectory.

Adds a first keyframe (priors only) and a second keyframe connected by a known
ICP transform, then checks the optimized relative pose matches the input. IMU
factors are omitted (``imu_factor=None``) so this isolates the pose graph.
"""

from __future__ import annotations

import numpy as np
import gtsam
import pytest

from core.slam_backend import SlamBackend

CONFIG = {
    "backend": {
        "bias_between_sigma": 0.001,
        "icp_rotation_sigma": 0.05,
        "icp_translation_sigma": 0.05,
    }
}


def test_two_keyframes_consistent() -> None:
    backend = SlamBackend(CONFIG)

    # Keyframe 0 at the origin (priors seeded).
    pose0 = backend.add_keyframe(np.eye(4), 1.0, None, None)
    np.testing.assert_allclose(pose0.matrix(), np.eye(4), atol=1e-6)

    # Keyframe 1: a known relative motion (translation + small yaw).
    rel = np.eye(4)
    angle = np.deg2rad(15.0)
    rel[:3, :3] = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    rel[:3, 3] = np.array([0.5, 0.1, 0.0])

    pose1 = backend.add_keyframe(rel, 0.95, None, None)

    # Optimized pose1 should equal pose0 * rel (pose0 == identity here).
    np.testing.assert_allclose(pose1.matrix(), rel, atol=1e-3)

    # And the recovered relative pose between the two should match the input.
    poses = backend.get_all_poses()
    assert set(poses.keys()) == {0, 1}
    recovered_rel = poses[0].between(poses[1]).matrix()
    np.testing.assert_allclose(recovered_rel, rel, atol=1e-3)


def test_loop_closure_update_runs() -> None:
    backend = SlamBackend(CONFIG)
    backend.add_keyframe(np.eye(4), 1.0, None, None)

    step = np.eye(4)
    step[:3, 3] = [0.5, 0.0, 0.0]
    for _ in range(3):
        backend.add_keyframe(step, 0.9, None, None)

    # Close a loop from the last keyframe back toward keyframe 0.
    n = backend.index - 1
    lc = np.eye(4)
    lc[:3, 3] = [-1.5, 0.0, 0.0]  # roughly the inverse of the accumulated path
    backend.add_loop_closure_factor(n, 0, lc, 0.8)

    poses = backend.get_all_poses()
    assert len(poses) == backend.index
    # Optimization stayed finite / valid.
    for p in poses.values():
        assert np.all(np.isfinite(p.matrix()))


def test_set_initial_bias_seeds_before_kf0() -> None:
    backend = SlamBackend(CONFIG)

    accel_bias = np.array([0.037, -0.012, 0.021])
    gyro_bias = np.array([0.0075, -0.0061, 0.0093])
    bias = gtsam.imuBias.ConstantBias(accel_bias, gyro_bias)

    backend.set_initial_bias(bias)
    expected = np.concatenate([accel_bias, gyro_bias])  # [accel(3); gyro(3)]
    np.testing.assert_allclose(backend.current_bias.vector(), expected, atol=1e-12)


def test_set_initial_bias_ordering_guard() -> None:
    backend = SlamBackend(CONFIG)
    # Once keyframe 0 exists (index advanced), seeding must be rejected.
    backend.add_keyframe(np.eye(4), 1.0, None, None)
    with pytest.raises(RuntimeError):
        backend.set_initial_bias(gtsam.imuBias.ConstantBias())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
