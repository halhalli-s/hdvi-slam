"""Test the point-to-plane ICP front-end recovers a known transform.

Builds a geometrically rich cloud (a box surface has three orthogonal
faces, so translation is fully constrained in every axis), applies a known
rigid transform to create the source, and checks that ``align`` recovers the
inverse mapping that brings source back onto target.
"""

from __future__ import annotations

import numpy as np
import open3d as o3d
import pytest

from core.icp_frontend import align
from core.types import CloudFrame

CONFIG = {
    "icp": {
        "voxel_size": 0.01,
        "max_correspondence_distance": 0.1,
        "max_iteration": 50,
        "relative_fitness": 1e-8,
        "relative_rmse": 1e-8,
        "normal_radius": 0.05,
        "normal_max_nn": 30,
    }
}


def _box_cloud() -> o3d.geometry.PointCloud:
    mesh = o3d.geometry.TriangleMesh.create_box(width=1.0, height=1.0, depth=1.0)
    pcd = mesh.sample_points_uniformly(number_of_points=5000)
    return pcd


def test_align_recovers_known_transform() -> None:
    target_cloud = _box_cloud()

    # Known transform: small rotation about Z + a translation.
    angle = np.deg2rad(8.0)
    T_known = np.eye(4)
    T_known[:3, :3] = o3d.geometry.get_rotation_matrix_from_axis_angle(
        np.array([0.0, 0.0, angle])
    )
    T_known[:3, 3] = np.array([0.05, -0.03, 0.04])

    # Source is the target moved by T_known.
    source_cloud = o3d.geometry.PointCloud(target_cloud)
    source_cloud.transform(T_known)

    source = CloudFrame(cloud=source_cloud, timestamp_ns=1)
    target = CloudFrame(cloud=target_cloud, timestamp_ns=0)

    # align returns the transform mapping source -> target, i.e. inv(T_known).
    T_est, fitness, rmse = align(source, target, CONFIG)

    T_expected = np.linalg.inv(T_known)

    assert fitness > 0.9
    assert rmse < 0.02
    np.testing.assert_allclose(T_est, T_expected, atol=1e-2)

    # Composing the estimate onto the known motion should give identity.
    residual = T_est @ T_known
    np.testing.assert_allclose(residual, np.eye(4), atol=1e-2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
