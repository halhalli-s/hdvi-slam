"""Point-to-plane ICP front-end: relative pose between two point clouds.

Given a ``source`` and ``target`` :class:`~core.types.CloudFrame`, this module
estimates the rigid transform that best aligns the source onto the target,
using Open3D's point-to-plane ICP. It is the visual odometry primitive that
produces the between-factor constraints for the back-end and is reused for
loop-closure verification.

What this module does NOT do:
  * It does not decide *when* to register frames (see core.keyframe_trigger).
  * It does not touch the factor graph (see core.slam_backend).
  * It knows nothing about IMU, gravity, or global poses — it only returns a
    relative transform plus quality metrics. The caller composes it into a
    trajectory.

Convention: the returned transform ``T`` maps points expressed in the source
frame into the target frame (``p_target = T @ p_source``). When ``target`` is
the previous keyframe and ``source`` is the current one, ``T`` is therefore the
pose of the current frame expressed in the previous frame (``^prev T_curr``),
which is exactly what a BetweenFactor(X_prev, X_curr) expects.
"""

from __future__ import annotations

import time  # [ICP-PROF]
from typing import Tuple

import numpy as np
import open3d as o3d

from core.types import CloudFrame

_ICP_PROF: dict = {}  # [ICP-PROF] per-call segment timings, filled by _preprocess


def _preprocess(
    cloud: o3d.geometry.PointCloud,
    voxel_size: float,
    estimate_normals: bool,
    normal_radius: float,
    normal_max_nn: int,
    max_points: int = 0,
) -> o3d.geometry.PointCloud:
    """Voxel-downsample and (optionally) estimate normals on a copy.

    ``max_points`` (0 = disabled): if the voxel-downsampled cloud still exceeds
    it, re-voxelize the ORIGINAL cloud at a coarser leaf size
    ``voxel_size * sqrt(len(down) / max_points)`` so point count lands near the
    cap while keeping UNIFORM spacing (normals are estimated on this cloud, so
    uniform spacing matters — hence re-voxelizing, not random thinning). Done
    BEFORE normal estimation.
    """
    _prof_t = time.perf_counter()  # [ICP-PROF]
    used_voxel = voxel_size
    down = cloud.voxel_down_sample(voxel_size)
    if max_points > 0 and len(down.points) > max_points:
        used_voxel = voxel_size * float(np.sqrt(len(down.points) / max_points))
        down = cloud.voxel_down_sample(used_voxel)
    _ICP_PROF["voxel_ms"] = (time.perf_counter() - _prof_t) * 1e3  # [ICP-PROF]
    _ICP_PROF["voxel_used"] = used_voxel  # [ICP-PROF] actual leaf size applied
    _ICP_PROF["normals_ms"] = 0.0  # [ICP-PROF]
    if estimate_normals:
        _prof_t = time.perf_counter()  # [ICP-PROF]
        down.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=normal_radius, max_nn=normal_max_nn
            )
        )
        _ICP_PROF["normals_ms"] = (time.perf_counter() - _prof_t) * 1e3  # [ICP-PROF]
    return down


def align(
    source: CloudFrame,
    target: CloudFrame,
    config: dict,
    initial_transform: np.ndarray = np.eye(4),
    max_corr_override: float | None = None,
    max_iter_override: int | None = None,
) -> Tuple[np.ndarray, float, float]:
    """Register ``source`` onto ``target`` with point-to-plane ICP.

    Args:
        source: The cloud to be moved (typically the current keyframe).
        target: The reference cloud (typically the previous keyframe).
        config: Parsed config dict; reads the ``icp`` section.
        initial_transform: 4x4 initial guess for the source->target transform.
        max_corr_override: If not None, use this max correspondence distance
            instead of ``icp.max_correspondence_distance`` (e.g. loop-closure
            coarse stage). Does not affect normal-estimation radius.
        max_iter_override: If not None, use this ICP iteration cap instead of
            ``icp.max_iteration``.

    Returns:
        ``(transform, fitness, inlier_rmse)`` where ``transform`` is a 4x4
        numpy array mapping source-frame points into the target frame,
        ``fitness`` is the fraction of source points with a correspondence
        (higher is better), and ``inlier_rmse`` is the RMSE over those
        correspondences (lower is better).
    """
    icp_cfg = config["icp"]
    voxel_size = float(icp_cfg["voxel_size"])
    cfg_max_corr = float(icp_cfg["max_correspondence_distance"])
    max_corr = cfg_max_corr if max_corr_override is None else float(max_corr_override)
    max_iter = (
        int(icp_cfg["max_iteration"])
        if max_iter_override is None
        else int(max_iter_override)
    )
    # normal_radius tracks the CONFIG corr distance, not an override.
    normal_radius = float(icp_cfg.get("normal_radius", cfg_max_corr))
    normal_max_nn = int(icp_cfg.get("normal_max_nn", 30))
    max_points = int(icp_cfg.get("max_points", 30000))

    # Downsample both clouds. Point-to-plane needs normals on the *target*
    # (the surface we project residuals onto); the source only needs points.
    # Each cloud is capped to max_points independently (adaptive re-voxelize).
    prof_src_in = len(source.cloud.points)  # [ICP-PROF]
    prof_tgt_in = len(target.cloud.points)  # [ICP-PROF]
    source_down = _preprocess(
        source.cloud, voxel_size, estimate_normals=False,
        normal_radius=normal_radius, normal_max_nn=normal_max_nn,
        max_points=max_points,
    )
    prof_src_voxel_ms = _ICP_PROF["voxel_ms"]  # [ICP-PROF]
    prof_src_voxel_used = _ICP_PROF["voxel_used"]  # [ICP-PROF]
    target_down = _preprocess(
        target.cloud, voxel_size, estimate_normals=True,
        normal_radius=normal_radius, normal_max_nn=normal_max_nn,
        max_points=max_points,
    )
    prof_tgt_voxel_ms = _ICP_PROF["voxel_ms"]  # [ICP-PROF]
    prof_tgt_voxel_used = _ICP_PROF["voxel_used"]  # [ICP-PROF]
    prof_tgt_normals_ms = _ICP_PROF["normals_ms"]  # [ICP-PROF]

    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
        relative_fitness=float(icp_cfg["relative_fitness"]),
        relative_rmse=float(icp_cfg["relative_rmse"]),
        max_iteration=max_iter,
    )

    _prof_reg_t = time.perf_counter()  # [ICP-PROF]
    result = o3d.pipelines.registration.registration_icp(
        source_down,
        target_down,
        max_corr,
        np.asarray(initial_transform, dtype=np.float64),
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        criteria,
    )
    prof_reg_ms = (time.perf_counter() - _prof_reg_t) * 1e3  # [ICP-PROF]
    print(f"[ICP-PROF] src={prof_src_in}->{len(source_down.points)}@{prof_src_voxel_used:.3f}m tgt={prof_tgt_in}->{len(target_down.points)}@{prof_tgt_voxel_used:.3f}m src_voxel={prof_src_voxel_ms:.1f}ms tgt_voxel={prof_tgt_voxel_ms:.1f}ms tgt_normals={prof_tgt_normals_ms:.1f}ms reg_icp={prof_reg_ms:.1f}ms fitness={float(result.fitness):.3f} rmse={float(result.inlier_rmse):.4f}", flush=True)  # [ICP-PROF]

    return np.asarray(result.transformation), float(result.fitness), float(result.inlier_rmse)
