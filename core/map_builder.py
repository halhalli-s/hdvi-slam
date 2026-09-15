"""Dense 3D occupancy mapping with OctoMap.

Wraps a single ``octomap.OcTree``. As each keyframe is finalized by the
back-end, its point cloud is transformed into the world frame using the
*optimized* pose and inserted with ray casting, so both occupied surfaces and
the free space in front of them are modeled. The occupied voxel centers can be
pulled out for visualization.

What this module does NOT do:
  * It does not estimate poses — it trusts the pose the back-end gives it.
  * It does not choose keyframes or run ICP.
  * It does not publish to any viewer (that is viz/).

Note: OctoMap Python bindings vary between forks (wkentaro/octomap-python vs
others). The insertion/iteration calls below target the common numpy-friendly
API; adjust if your installed binding differs. Marked with TODO where the API
is fork-sensitive.
"""

from __future__ import annotations

import numpy as np
import gtsam
import octomap

from core.types import CloudFrame


class MapBuilder:
    """Incremental OctoMap builder fed by optimized keyframes."""

    def __init__(self, resolution: float) -> None:
        self.resolution = float(resolution)
        self.tree = octomap.OcTree(self.resolution)

    def insert_cloud(self, cloud: CloudFrame, pose: gtsam.Pose3) -> None:
        """Transform a cloud by ``pose`` and insert it with ray casting.

        The sensor origin is the pose translation; ray casting from that origin
        to each point carves free space and marks the endpoint occupied.
        """
        points = np.asarray(cloud.cloud.points, dtype=np.float64)
        if points.size == 0:
            return

        # Transform points from the body/camera frame into the world frame.
        T = pose.matrix()  # 4x4
        homog = np.hstack([points, np.ones((points.shape[0], 1))])
        world_points = (T @ homog.T).T[:, :3]

        origin = np.asarray(pose.translation(), dtype=np.float64)

        # TODO(binding): insertPointCloud signature differs across octomap
        # forks. wkentaro/octomap-python accepts (Nx3 float array, origin (3,)).
        self.tree.insertPointCloud(world_points, origin)

    def get_occupied_points(self) -> np.ndarray:
        """Return occupied voxel centers as an ``(N, 3)`` array for viz."""
        occupied = []
        # TODO(binding): leaf iteration API is fork-sensitive. This targets the
        # wkentaro binding's tree iterator with getCoordinate()/getSize().
        try:
            for node in self.tree.begin_tree():
                if node.isLeaf() and self.tree.isNodeOccupied(node):
                    c = node.getCoordinate()
                    occupied.append([c[0], c[1], c[2]])
        except AttributeError:
            # Fallback for bindings exposing getOccupied()/numpy export.
            occupied = list(getattr(self.tree, "getOccupiedVoxelCenters", lambda: [])())

        if not occupied:
            return np.empty((0, 3), dtype=np.float64)
        return np.asarray(occupied, dtype=np.float64)
