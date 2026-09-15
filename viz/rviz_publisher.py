"""RViz2 publisher — the ONLY module that imports ROS.

Bridges estimator output to RViz2 for live, incremental inspection:

  * Trajectory  -> ``nav_msgs/Path`` on ``/slam/trajectory`` (frame ``map``).
    Each optimized keyframe pose is appended to the Path and the whole Path is
    republished, so RViz shows the trajectory grow live.
  * Map         -> ``sensor_msgs/PointCloud2`` on ``/slam/map`` (frame ``map``).
    Each keyframe's world-frame cloud is accumulated into a running buffer,
    voxel-downsampled to keep it bounded, and republished in full.

Both publishers use QoS reliable, depth 10.

Node lifecycle: the orchestrator does not init rclpy itself, so by default this
class initializes rclpy and owns a Node. Alternatively, pass an existing
``node`` to have it borrow one (and leave rclpy shutdown to the owner). If ROS
is unavailable or setup fails, the publisher disables itself and every publish
call becomes a no-op — the SLAM pipeline runs headless without ROS.

What this module does NOT do:
  * No estimation, mapping, or ICP — it only serializes and publishes what the
    back-end / worker hand it.
  * Zero ROS imports leak outside this file; core/ never touches rclpy.
"""

from __future__ import annotations

import threading
from typing import Optional

import numpy as np
import open3d as o3d
import gtsam


class RvizPublisher:
    """Publishes the growing trajectory and accumulated map to RViz2."""

    def __init__(
        self,
        node: Optional[object] = None,
        frame_id: str = "map",
        map_voxel_size: float = 0.05,
        node_name: str = "sachislam_viz",
    ) -> None:
        self.frame_id = frame_id
        self.map_voxel_size = float(map_voxel_size)
        self.node_name = node_name

        self._enabled = False
        self._owns_rclpy = False
        self._node = None
        self._path_pub = None
        self._map_pub = None

        # Persistent Path message; poses are appended each keyframe.
        self._path_msg = None
        # World-frame map buffer, FULLY REBUILT from current poses each publish.
        self._accum = o3d.geometry.PointCloud()
        # Source of truth for map geometry: each keyframe's LOCAL-frame cloud,
        # keyed by keyframe index -> (points (N,3) float64, colors (N,3) or None).
        # Kept in the publisher so the map rebuild is self-contained (no reaching
        # into core's LoopClosureDetector._history), preserving the viz/core
        # separation while still having a single owner of the geometry. Clouds
        # are pre-downsampled to map_voxel_size at store time (the MAP's copy
        # only; ICP's cloud is untouched).
        self._keyframe_clouds: dict = {}

        # --- Map rebuild runs on its OWN thread so it can never throttle the
        # keyframe/backend loop. publish_keyframe (called on the keyframe worker)
        # only stores the cloud + stashes poses + publishes the cheap Path, then
        # signals this thread. The heavy transform/voxelize/serialize happens
        # here, off the frontend's critical path. -----------------------------
        self._map_lock = threading.Lock()          # guards the two fields below
        self._latest_poses: dict = {}              # newest {index: Pose3} snapshot
        self._map_dirty = threading.Event()         # set when new data arrives
        self._map_thread: Optional[threading.Thread] = None
        self._map_thread_running = False
        # Change detection: matrices of poses used for the last rebuild.
        self._last_rebuild_poses: dict = {}         # {index: 4x4 np.ndarray}
        # Rebuild only if a pose moved more than this (max abs elem of the 4x4
        # delta -> ~1 mm translation / ~0.06 deg rotation), or count changed.
        self._pose_change_threshold = 1.0e-3
        self._last_cloud_msg = None                 # cached msg for republish

        # Cached ROS message types / helpers (populated on successful setup).
        self._PoseStamped = None
        self._Header = None
        self._pc2 = None

        try:
            self._setup(node)
            self._enabled = True
        except Exception as exc:  # noqa: BLE001 - run headless if ROS is absent
            print(
                f"[RvizPublisher] ROS setup failed ({exc}); running without RViz.",
                flush=True,
            )

    # ------------------------------------------------------------------ setup
    def _setup(self, node: Optional[object]) -> None:
        # Lazy ROS imports: this is the one and only rclpy import site.
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import QoSProfile, ReliabilityPolicy
        from geometry_msgs.msg import PoseStamped
        from nav_msgs.msg import Path
        from sensor_msgs.msg import PointCloud2, PointField
        from sensor_msgs_py import point_cloud2
        from std_msgs.msg import Header

        if node is not None:
            self._node = node
        else:
            if not rclpy.ok():
                rclpy.init()
                self._owns_rclpy = True
            self._node = Node(self.node_name)

        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        self._path_pub = self._node.create_publisher(Path, "/slam/trajectory", qos)
        self._map_pub = self._node.create_publisher(PointCloud2, "/slam/map", qos)

        self._PoseStamped = PoseStamped
        self._Header = Header
        self._pc2 = point_cloud2
        # XYZRGB PointCloud2 fields: xyz float32 + packed rgb float32 (RViz
        # interprets a FLOAT32 field named "rgb" as bit-packed 0xRRGGBB).
        self._xyzrgb_fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="rgb", offset=12, datatype=PointField.FLOAT32, count=1),
        ]

        self._path_msg = Path()
        self._path_msg.header.frame_id = self.frame_id

        # Launch the background map thread (daemon so it never blocks exit).
        self._map_thread_running = True
        self._map_thread = threading.Thread(
            target=self._map_loop, name="rviz-map", daemon=True
        )
        self._map_thread.start()

    # ----------------------------------------------------------------- helpers
    def _now_header(self):
        header = self._Header()
        header.stamp = self._node.get_clock().now().to_msg()
        header.frame_id = self.frame_id
        return header

    def _pose_stamped(self, pose: gtsam.Pose3):
        ps = self._PoseStamped()
        ps.header = self._now_header()
        t = pose.translation()
        ps.pose.position.x = float(t[0])
        ps.pose.position.y = float(t[1])
        ps.pose.position.z = float(t[2])
        q = pose.rotation().toQuaternion()
        ps.pose.orientation.w = float(q.w())
        ps.pose.orientation.x = float(q.x())
        ps.pose.orientation.y = float(q.y())
        ps.pose.orientation.z = float(q.z())
        return ps

    def _make_xyzrgb_cloud(self, xyz: np.ndarray, colors01: np.ndarray):
        """Build an XYZRGB PointCloud2 with bit-packed rgb (no float64 round-trip).

        ``colors01`` are 0-1 floats. They are quantized to 0-255 and packed as
        ``0x00RRGGBB`` in a uint32, then reinterpreted bit-for-bit as float32 for
        the "rgb" field. Filling a structured array whose dtype matches the
        declared fields lets create_cloud() memcpy it, preserving the packed
        bit pattern (a plain float64 tuple round-trip would corrupt it).
        """
        n = xyz.shape[0]
        c = np.clip(colors01 * 255.0, 0, 255).astype(np.uint32)
        rgb_u32 = (c[:, 0] << 16) | (c[:, 1] << 8) | c[:, 2]
        rgb_f32 = rgb_u32.view(np.float32)

        dt = self._pc2.dtype_from_fields(self._xyzrgb_fields)
        structured = np.zeros(n, dtype=dt)
        structured["x"] = xyz[:, 0]
        structured["y"] = xyz[:, 1]
        structured["z"] = xyz[:, 2]
        structured["rgb"] = rgb_f32
        return self._pc2.create_cloud(
            self._now_header(), self._xyzrgb_fields, structured
        )

    def _poses_changed(self, poses: dict) -> bool:
        """True if the optimized poses materially differ from the last rebuild.

        Rebuild when the keyframe set changed, or any pose's 4x4 moved by more
        than ``_pose_change_threshold`` (max abs element of the delta). This is
        the simple, robust detector: compare current poses to a cached copy from
        the last rebuild.
        """
        if set(poses) != set(self._last_rebuild_poses):
            return True
        for k, p in poses.items():
            prev = self._last_rebuild_poses.get(k)
            if prev is None:
                return True
            if np.abs(p.matrix() - prev).max() > self._pose_change_threshold:
                return True
        return False

    def _rebuild_and_publish(self, poses: dict, clouds: dict) -> None:
        """Rebuild the world map from CURRENT poses + stored local clouds, publish.

        For every keyframe index ``k``, transform its (already map-downsampled)
        LOCAL cloud by the current optimized pose ``poses[k]`` and accumulate;
        colors ride through row-aligned with their points. One final voxel
        downsample merges per-keyframe overlaps. Runs ONLY on the map thread.
        """
        pts_list = []
        col_list = []
        have_colors = True
        for k in sorted(poses):
            entry = clouds.get(k)
            if entry is None:
                continue  # pose exists but its local cloud not stored (yet)
            local_pts, local_cols = entry
            if local_pts.shape[0] == 0:
                continue
            # Transform local -> world with the CURRENT optimized pose.
            T = poses[k].matrix()
            homog = np.hstack([local_pts, np.ones((local_pts.shape[0], 1))])
            world = (T @ homog.T).T[:, :3]
            pts_list.append(world)
            if local_cols is not None and local_cols.shape[0] == local_pts.shape[0]:
                col_list.append(local_cols)
            else:
                have_colors = False

        if not pts_list:
            self._accum = o3d.geometry.PointCloud()
            self._last_rebuild_poses = {k: poses[k].matrix() for k in poses}
            return

        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(np.vstack(pts_list))
        if have_colors and len(col_list) == len(pts_list):
            # Colors stacked in the SAME keyframe order as points -> row-aligned.
            cloud.colors = o3d.utility.Vector3dVector(np.vstack(col_list))
        # Final voxel downsample merges per-keyframe overlaps (each keyframe is
        # already at map_voxel_size, so this is cheap); averages colors with
        # points so they stay row-aligned.
        self._accum = cloud.voxel_down_sample(self.map_voxel_size)

        world_pts = np.asarray(self._accum.points, dtype=np.float32)
        world_cols = np.asarray(self._accum.colors)  # (N,3) in 0-1, or empty
        if world_cols.shape[0] == world_pts.shape[0] and world_pts.shape[0] > 0:
            cloud_msg = self._make_xyzrgb_cloud(world_pts, world_cols)
        else:
            # No colors available -> plain XYZ (headless / color-less path).
            cloud_msg = self._pc2.create_cloud_xyz32(
                self._now_header(), world_pts.tolist()
            )
        self._last_cloud_msg = cloud_msg
        self._last_rebuild_poses = {k: poses[k].matrix() for k in poses}
        self._map_pub.publish(cloud_msg)

    def _map_loop(self) -> None:
        """Background thread: rebuild the map only when poses change; else republish.

        Never runs on the keyframe/backend path, so a slow rebuild cannot throttle
        the frontend. Wakes on new-data signal or on a periodic timeout (so late
        RViz subscribers still receive the last map).
        """
        while self._map_thread_running:
            self._map_dirty.wait(timeout=0.5)
            self._map_dirty.clear()
            if not self._map_thread_running:
                break
            # Brief critical section: snapshot poses + cloud refs, then release
            # the lock so the heavy rebuild does not block publish_keyframe.
            with self._map_lock:
                poses = dict(self._latest_poses)
                clouds = dict(self._keyframe_clouds)  # values are immutable arrays
            if not poses:
                continue
            try:
                if self._poses_changed(poses):
                    self._rebuild_and_publish(poses, clouds)
                elif self._last_cloud_msg is not None:
                    # Nothing changed -> cheap republish of the cached buffer.
                    self._last_cloud_msg.header.stamp = (
                        self._node.get_clock().now().to_msg()
                    )
                    self._map_pub.publish(self._last_cloud_msg)
            except Exception as exc:  # noqa: BLE001 - never kill the map thread
                print(f"[RvizPublisher] map rebuild failed ({exc}).", flush=True)

    # ------------------------------------------------------------------- publish
    def publish_keyframe(
        self,
        pose: gtsam.Pose3,
        index: int,
        local_points: np.ndarray,
        all_poses: Optional[dict] = None,
        colors: Optional[np.ndarray] = None,
    ) -> None:
        """Store this keyframe's LOCAL cloud + Path (cheap); signal the map thread.

        Runs on the keyframe worker, so it does only cheap work: pre-downsample
        and store the keyframe's local cloud, rebuild+publish the (cheap) Path
        from ``all_poses``, stash the poses, and signal the background map thread
        to (re)build the heavy point cloud. The map rebuild itself happens off
        this thread and can never throttle keyframe/backend processing. A no-op
        if ROS setup failed.
        """
        if not self._enabled:
            return
        try:
            # --- Store LOCAL cloud, PRE-DOWNSAMPLED to map resolution ---------
            # This is the MAP's private copy; ICP's cloud is untouched. Voxelize
            # once here so the rebuild transforms far fewer points per keyframe.
            pts = np.asarray(local_points, dtype=np.float64).reshape(-1, 3)
            cols = None
            if colors is not None:
                c = np.asarray(colors, dtype=np.float64).reshape(-1, 3)
                if c.shape[0] == pts.shape[0]:
                    cols = c
            if pts.shape[0] > 0:
                tmp = o3d.geometry.PointCloud()
                tmp.points = o3d.utility.Vector3dVector(pts)
                if cols is not None:
                    tmp.colors = o3d.utility.Vector3dVector(cols)
                tmp = tmp.voxel_down_sample(self.map_voxel_size)
                ds_pts = np.asarray(tmp.points)
                ds_cols = np.asarray(tmp.colors) if tmp.has_colors() else None
            else:
                ds_pts, ds_cols = pts, cols

            poses_snapshot = (
                dict(all_poses) if all_poses else {index: pose}
            )
            with self._map_lock:
                self._keyframe_clouds[index] = (ds_pts, ds_cols)
                self._latest_poses = poses_snapshot
            # Wake the map thread to rebuild with the new keyframe/poses.
            self._map_dirty.set()

            # --- Trajectory: rebuild from the backend's CURRENT optimized set
            # (cheap; stays inline). Reflects iSAM2 revisions / loop closures. --
            if all_poses:
                self._path_msg.poses = [
                    self._pose_stamped(all_poses[k]) for k in sorted(all_poses)
                ]
            else:
                # Fallback: no optimized set supplied -> keep append behavior.
                self._path_msg.poses.append(self._pose_stamped(pose))
            self._path_msg.header.stamp = self._node.get_clock().now().to_msg()
            self._path_pub.publish(self._path_msg)
        except Exception as exc:  # noqa: BLE001 - never take down the pipeline
            print(
                f"[RvizPublisher] publish failed ({exc}); disabling RViz output.",
                flush=True,
            )
            self._enabled = False

    def shutdown(self) -> None:
        """Tear down the ROS node (and rclpy if we initialized it)."""
        # Stop the background map thread first.
        self._map_thread_running = False
        self._map_dirty.set()
        if self._map_thread is not None:
            self._map_thread.join(timeout=2.0)
            self._map_thread = None
        if self._node is not None:
            try:
                self._node.destroy_node()
            except Exception:  # noqa: BLE001
                pass
            self._node = None
        if self._owns_rclpy:
            try:
                import rclpy

                if rclpy.ok():
                    rclpy.shutdown()
            except Exception:  # noqa: BLE001
                pass
            self._owns_rclpy = False
        self._enabled = False
