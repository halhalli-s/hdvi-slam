"""Loop-closure detection by spatial proximity + ICP verification.

Maintains a lightweight history of past keyframes (pose + cloud) and, for each
new keyframe, looks for an older one that is physically nearby. Any nearby
candidate is verified geometrically by re-running ICP against its stored cloud;
only a high-fitness match becomes a loop-closure constraint. This corrects the
drift that pure odometry accumulates.

What this module does NOT do:
  * It does not run ICP itself — the caller injects an ``icp_align_fn`` so the
    single Open3D code path in core.icp_frontend stays authoritative.
  * It does not add anything to the graph — it returns a candidate for
    core.slam_backend.add_loop_closure_factor to apply.
  * It uses only the optimized poses passed in; it holds no optimizer state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np
import gtsam

from core.types import CloudFrame


@dataclass
class _KeyframeRecord:
    index: int
    position: np.ndarray  # (3,) translation of the pose in the world frame
    cloud: CloudFrame


# An align function with the signature of core.icp_frontend.align bound to a
# config: (source, target, initial_transform, max_corr_override,
# max_iter_override) -> (transform, fitness, rmse). The last two are optional
# overrides (None = use config) used for the two-stage coarse/fine verification.
AlignFn = Callable[
    [CloudFrame, CloudFrame, np.ndarray, Optional[float], Optional[int]],
    Tuple[np.ndarray, float, float],
]


class LoopClosureDetector:
    """Proximity-based loop-closure detector with ICP verification."""

    def __init__(self) -> None:
        self._history: List[_KeyframeRecord] = []

    def add_keyframe(self, index: int, pose: gtsam.Pose3, cloud: CloudFrame) -> None:
        """Record a keyframe so future frames can close a loop against it."""
        self._history.append(
            _KeyframeRecord(index=index, position=pose.translation(), cloud=cloud)
        )

    def check(
        self,
        current_index: int,
        current_pose: gtsam.Pose3,
        current_cloud: CloudFrame,
        all_poses: dict,
        icp_align_fn: AlignFn,
        config: dict,
    ) -> List[Tuple[int, np.ndarray, float]]:
        """Look for loop closures for the current keyframe.

        Args:
            current_index: Index of the current keyframe.
            current_pose: Its optimized pose (world frame).
            current_cloud: Its point cloud.
            all_poses: The backend's CURRENT optimized ``{index: Pose3}`` set.
                Used to seed the verification ICP from LIVE poses (T_i is
                keyframe i's current pose, not its possibly-stale insertion-time
                pose in the history).
            icp_align_fn: ``(source, target, initial_transform) ->
                (transform, fitness, rmse)``.
            config: Parsed config; reads the ``loop_closure`` section.

        Returns:
            A list of ``(matched_index, transform, fitness)`` for EVERY verified
            candidate that passed the fitness gate, strongest first and capped at
            ``max_accepted_factors``. Each ``transform`` maps the current cloud
            into the matched keyframe's frame (``^{matched} T_{current}``),
            suitable for BetweenFactor(X(current), X(matched)). Empty list if no
            candidate is close enough or none passes the fitness gate.
        """
        lc_cfg = config["loop_closure"]
        # Rate-limit: only run the (expensive) check every Nth keyframe. Returns
        # immediately BEFORE any proximity search or ICP, so skipped keyframes
        # cost nothing.
        check_interval = int(lc_cfg.get("check_interval", 5))
        if current_index % check_interval != 0:
            return []
        radius = float(lc_cfg["proximity_radius_m"])
        # Stricter than the odometry gate: a false loop closure is far more
        # damaging than a dropped one.
        min_fitness = float(lc_cfg.get("min_fitness", 0.88))
        # Exclude temporal neighbors so recent keyframes can't be returned as
        # loop candidates.
        min_index_gap = int(lc_cfg.get("min_index_gap", 10))
        # Cap the number of verification ICPs per keyframe: rank candidates by
        # distance (nearest-first) and verify only the K closest. Turns the
        # per-keyframe cost from O(N) into O(1). Does NOT change the spatial
        # radius, the ICP verification, the fitness gate, or the match-picking.
        max_candidates = int(lc_cfg.get("max_candidates", 3))
        # Cap how many accepted loop-closure factors we return per keyframe.
        max_accepted = int(lc_cfg.get("max_accepted_factors", 3))
        # Two-stage coarse/fine verification params (stage 1 = wide-radius
        # capture, stage 2 = normal-radius refine). Gating uses fine fitness.
        coarse_corr = float(lc_cfg.get("coarse_corr_distance", 0.3))
        coarse_iter = int(lc_cfg.get("coarse_max_iteration", 10))
        # TEMP: seed the coarse stage with identity instead of the graph-derived
        # relative pose. The graph seed is still computed and logged regardless.
        use_identity_seed = bool(lc_cfg.get("use_identity_seed", False))

        current_pos = current_pose.translation()

        # 1) Proximity search: nearby historical keyframes, excluding the
        #    immediately previous one(s) which are already odometry neighbors.
        candidates: List[_KeyframeRecord] = []
        for rec in self._history:
            if current_index - rec.index <= min_index_gap:
                continue
            if float(np.linalg.norm(rec.position - current_pos)) <= radius:
                candidates.append(rec)

        if not candidates:
            return []

        # Prefer the spatially closest candidate first.
        candidates.sort(key=lambda r: float(np.linalg.norm(r.position - current_pos)))

        # 2) Geometric verification via ICP on ONLY the K closest candidates,
        #    gated on fitness. (candidates is already sorted nearest-first.)
        accepted_matches: List[Tuple[int, np.ndarray, float]] = []
        for rec in candidates[:max_candidates]:
            # Seed verification ICP with the relative pose between the two
            # keyframes' LIVE optimized poses: T_init = T_i^{-1} * T_j = ^{i}T_{j}
            # (Pose3.between) — exactly what align(current, rec) estimates. T_i is
            # keyframe i's CURRENT optimized pose (all_poses), not the stale
            # insertion-time one. Far better than identity across a large loop;
            # no IMU. Falls back to identity if i's pose is unavailable.
            pose_i = all_poses.get(rec.index)
            if pose_i is not None:
                t_init_mat = pose_i.between(current_pose).matrix()
                seed_mag = float(np.linalg.norm(t_init_mat[:3, 3]))
            else:
                t_init_mat = np.eye(4)
                seed_mag = 0.0
            # TEMP: optionally ignore the computed seed and start from identity.
            # seed_mag above still reflects the graph-derived seed for the log.
            coarse_seed = np.eye(4) if use_identity_seed else t_init_mat
            # Stage 1 (coarse): wide correspondence radius + few iterations to
            # pull the seed into rough alignment. Its fitness is NOT gated on
            # (measured at a loose radius, not comparable).
            coarse_transform, coarse_fit, _c_rmse = icp_align_fn(
                current_cloud, rec.cloud, coarse_seed, coarse_corr, coarse_iter
            )
            # Stage 2 (fine): refine from the coarse result at the normal ICP
            # radius/iterations (config defaults via None overrides). Gate here.
            transform, fitness, rmse = icp_align_fn(
                current_cloud, rec.cloud, coarse_transform, None, None
            )
            accepted = fitness > min_fitness
            print(
                f"[LC-GATE] candidate KF {rec.index}: seed|t|={seed_mag:.3f}m "
                f"{'[identity-seed] ' if use_identity_seed else ''}"
                f"coarse_fit={coarse_fit:.3f} fine_fit={fitness:.3f} "
                f"fine_rmse={rmse:.4f} "
                f"{'ACCEPT' if accepted else 'REJECT'} (min_fitness={min_fitness})",
                flush=True,
            )
            if accepted:
                accepted_matches.append((rec.index, transform, fitness))

        # Strongest closures first; cap how many are returned per keyframe.
        accepted_matches.sort(key=lambda m: m[2], reverse=True)
        return accepted_matches[:max_accepted]
