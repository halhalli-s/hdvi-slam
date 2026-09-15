"""Incremental factor-graph back-end (GTSAM iSAM2).

Owns a single :class:`gtsam.ISAM2` instance and the running estimate of the
trajectory (pose, velocity, bias per keyframe). Each keyframe contributes an
ICP odometry edge, an IMU preintegration factor, and a constant-bias
random-walk edge; loop closures add extra between-factors. Every ``update``
is incremental, so re-optimization after a loop closure is cheap.

What this module does NOT do:
  * It does not run ICP (core.icp_frontend) or preintegrate IMU
    (core.imu_preintegrator) — it consumes their outputs.
  * It does not detect loop closures (core.loop_closure); it only applies a
    verified one via :meth:`add_loop_closure_factor`.
  * It does not build the map or visualize anything.

State keys use GTSAM shorthand: ``X(i)`` pose, ``V(i)`` velocity, ``B(i)``
bias for keyframe ``i``.
"""

from __future__ import annotations

import time  # [LC-SOLVE] wall-clock timing of the post-loop-closure full solve
from typing import Dict, Optional

import numpy as np
import gtsam
from gtsam.symbol_shorthand import B, V, X

from core.imu_preintegrator import ImuPreintegrator


class SlamBackend:
    """Single-owner wrapper around an iSAM2 factor graph."""

    def __init__(self, config: Optional[dict] = None) -> None:
        config = config or {}
        backend_cfg = config.get("backend", {})

        params = gtsam.ISAM2Params()
        self.isam = gtsam.ISAM2(params)

        # Next keyframe index to assign.
        self.index: int = 0

        # Running estimate, updated after every isam.update().
        self._estimate: gtsam.Values = gtsam.Values()

        # Current-state trackers used to seed initial values / prediction.
        self.current_pose: gtsam.Pose3 = gtsam.Pose3()
        self.current_velocity: np.ndarray = np.zeros(3)
        self.current_bias: gtsam.imuBias.ConstantBias = gtsam.imuBias.ConstantBias()

        # --- Noise models -----------------------------------------------------
        # Priors on the very first keyframe.
        self._prior_pose_noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.array([0.001, 0.001, 0.001, 0.001, 0.001, 0.001])
        )
        self._prior_vel_noise = gtsam.noiseModel.Isotropic.Sigma(3, 0.1)
        # Anisotropic B(0) prior. The measured accel bias is derived parallel to
        # the gravity/up direction, so its perpendicular components are UNMEASURED
        # (not zero) — hence a loose accel sigma. Gyro bias is measured directly
        # from the mean gyro, so it gets a tight sigma. Tangent order is
        # [accel(3); gyro(3)] — verified against the installed gtsam
        # (ConstantBias.vector() == [acc; gyro]).
        self._prior_bias_noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.array(
                [float(backend_cfg.get("prior_bias_accel_sigma", 0.1))] * 3
                + [float(backend_cfg.get("prior_bias_gyro_sigma", 1e-3))] * 3
            )
        )

        # Base ICP odometry between-factor sigmas; scaled inversely with fitness.
        # Tightened from 0.05 — real ICP at fitness ~0.99 delivers sub-degree,
        # sub-cm relative pose between adjacent keyframes.
        self._icp_rot_sigma = float(backend_cfg.get("icp_rotation_sigma", 0.005))     # ~0.3°
        self._icp_trans_sigma = float(backend_cfg.get("icp_translation_sigma", 0.01)) # 1 cm

        # Loop-closure between-factor sigmas — separate from odometry, slightly
        # looser because LC connects potentially distant keyframes with more
        # geometric uncertainty than adjacent-frame odometry.
        self._lc_rot_sigma = float(backend_cfg.get("lc_rotation_sigma", 0.01))        # ~0.6°
        self._lc_trans_sigma = float(backend_cfg.get("lc_translation_sigma", 0.02))   # 2 cm

        # Bias random-walk between-factor sigma (~0.001 by spec).
        self._bias_between_sigma = float(backend_cfg.get("bias_between_sigma", 0.001))

        # Extra iSAM2 relinearization sweeps after a loop closure (no new
        # factors), so a big multi-variable correction settles before we read
        # the estimate. Odometry keyframes keep a single update.
        self._lc_extra_updates = int(backend_cfg.get("lc_extra_updates", 2))

        # Reuse the existing debug flag to gate the [LC-SOLVE] timing print
        # for the full post-loop-closure solve.
        self._debug = bool(config.get("debug", {}).get("verbose_trigger", False))

        # Weak-edge threshold for the odometry ICP edge: below this fitness the
        # BetweenFactor is still added but with its sigmas inflated by
        # ``icp_reject_sigma_scale`` (the ImuFactor still connects X(i)).
        self._icp_min_fitness = float(config.get("icp", {}).get("min_fitness", 0.5))
        self._icp_reject_sigma_scale = float(
            backend_cfg.get("icp_reject_sigma_scale", 15.0)
        )

    def set_initial_bias(self, bias: gtsam.imuBias.ConstantBias) -> None:
        """Seed the value used for B(0)'s prior and initial estimate."""
        if self.index != 0:
            raise RuntimeError("set_initial_bias() must be called before keyframe 0")
        self.current_bias = bias

    # ------------------------------------------------------------------ helpers
    def _icp_noise(
        self, fitness: float, extra_scale: float = 1.0
    ) -> gtsam.noiseModel.Diagonal:
        """Build a Pose3 between-factor noise model scaled by ICP fitness.

        Higher fitness -> smaller sigma -> the optimizer trusts the edge more.
        Order matches GTSAM's Pose3 tangent: [rx, ry, rz, tx, ty, tz].
        ``extra_scale`` multiplies the sigmas further (e.g. to add a
        low-fitness ICP edge weakly instead of dropping it).
        """
        # sigma / fitness^2: quadratic (vs linear) so a marginal edge is
        # down-weighted much harder — a 0.5-fitness edge gets 4x sigma, not 2x.
        scale = extra_scale / max(float(fitness), 1e-3) ** 2
        sigmas = np.array(
            [self._icp_rot_sigma * scale] * 3 + [self._icp_trans_sigma * scale] * 3
        )
        return gtsam.noiseModel.Diagonal.Sigmas(sigmas)

    def _lc_noise(self, fitness: float) -> gtsam.noiseModel.Diagonal:
        """Loop-closure noise model — separate sigmas from odometry ICP."""
        scale = 1.0 / max(float(fitness), 1e-3)
        sigmas = np.array(
            [self._lc_rot_sigma * scale] * 3 + [self._lc_trans_sigma * scale] * 3
        )
        return gtsam.noiseModel.Diagonal.Sigmas(sigmas)

    def _bias_noise(self) -> gtsam.noiseModel.Diagonal:
        return gtsam.noiseModel.Isotropic.Sigma(6, self._bias_between_sigma)

    def _refresh_estimate(self, best: bool = False) -> None:
        # ``best`` selects the full (batch) solve ``calculateBestEstimate()``,
        # used only after a loop closure; odometry keyframes keep the cheaper
        # incremental ``calculateEstimate()``.
        if best:
            _t = time.perf_counter()
            self._estimate = self.isam.calculateBestEstimate()
            if self._debug:
                print(
                    f"[LC-SOLVE] calculateBestEstimate="
                    f"{(time.perf_counter() - _t) * 1e3:.1f}ms",
                    flush=True,
                )
        else:
            self._estimate = self.isam.calculateEstimate()

    # ------------------------------------------------------------------- public
    def add_keyframe(
        self,
        icp_transform: Optional[np.ndarray],
        icp_fitness: float,
        imu_factor: Optional[gtsam.ImuFactor] = None,
        preintegrated_meas: Optional[gtsam.PreintegratedImuMeasurements] = None,
    ) -> gtsam.Pose3:
        """Add one keyframe and return its optimized pose.

        The very first keyframe (index 0) seeds priors on X(0)/V(0)/B(0). Every
        subsequent keyframe adds:
          * a BetweenFactorPose3 from the ICP transform (fitness-scaled noise),
          * the ImuFactor (if provided), and
          * a BetweenFactorConstantBias random-walk edge (if IMU is in use).

        ``imu_factor`` / ``preintegrated_meas`` may be ``None`` (e.g. in tests or
        a vision-only configuration), in which case only the ICP edge is added.
        """
        graph = gtsam.NonlinearFactorGraph()
        values = gtsam.Values()
        i = self.index

        if i == 0:
            pose0 = (
                gtsam.Pose3(np.asarray(icp_transform, dtype=np.float64))
                if icp_transform is not None
                else gtsam.Pose3()
            )
            graph.push_back(gtsam.PriorFactorPose3(X(0), pose0, self._prior_pose_noise))
            graph.push_back(
                gtsam.PriorFactorVector(V(0), self.current_velocity, self._prior_vel_noise)
            )
            graph.push_back(
                gtsam.PriorFactorConstantBias(B(0), self.current_bias, self._prior_bias_noise)
            )
            values.insert(X(0), pose0)
            values.insert(V(0), self.current_velocity)
            values.insert(B(0), self.current_bias)
        else:
            prev_pose = self._estimate.atPose3(X(i - 1))
            rel = gtsam.Pose3(np.asarray(icp_transform, dtype=np.float64))

            # Odometry edge from ICP. Down-weight a low-fitness registration:
            # still add the BetweenFactor (so it contributes some information)
            # but with its sigmas inflated so a bad ICP can't dominate the
            # graph. Only weaken when the ImuFactor is present to still connect
            # X(i); with no IMU (tests / vision-only) we keep the normal edge or
            # the graph is indeterminate.
            if icp_fitness < self._icp_min_fitness and imu_factor is not None:
                print(
                    f"[ICP-REJECT] KF {i}: fitness={icp_fitness:.3f} < "
                    f"{self._icp_min_fitness} — ICP BetweenFactor added weakly "
                    f"(sigma x{self._icp_reject_sigma_scale})",
                    flush=True,
                )
                graph.push_back(
                    gtsam.BetweenFactorPose3(
                        X(i - 1),
                        X(i),
                        rel,
                        self._icp_noise(icp_fitness, self._icp_reject_sigma_scale),
                    )
                )
            else:
                graph.push_back(
                    gtsam.BetweenFactorPose3(
                        X(i - 1), X(i), rel, self._icp_noise(icp_fitness)
                    )
                )
            predicted_pose = prev_pose.compose(rel)
            values.insert(X(i), predicted_pose)

            if imu_factor is not None:
                graph.push_back(imu_factor)
                # Constant-bias random walk edge B(i-1) -> B(i).
                graph.push_back(
                    gtsam.BetweenFactorConstantBias(
                        B(i - 1), B(i), gtsam.imuBias.ConstantBias(), self._bias_noise()
                    )
                )

                # Seed V(i)/B(i). Predict velocity via the preintegration if we
                # have it, otherwise carry the previous velocity forward.
                if preintegrated_meas is not None:
                    nav0 = gtsam.NavState(prev_pose, self.current_velocity)
                    nav1 = preintegrated_meas.predict(nav0, self.current_bias)
                    predicted_vel = nav1.velocity()
                else:
                    predicted_vel = self.current_velocity

                values.insert(V(i), predicted_vel)
                values.insert(B(i), self.current_bias)

        self.isam.update(graph, values)
        self._refresh_estimate()

        # Refresh trackers from the optimized estimate.
        self.current_pose = self._estimate.atPose3(X(i))
        if self._estimate.exists(V(i)):
            self.current_velocity = self._estimate.atVector(V(i))
        if self._estimate.exists(B(i)):
            self.current_bias = self._estimate.atConstantBias(B(i))

        self.index += 1
        return self.current_pose

    def add_loop_closure_factor(
        self, from_key: int, to_key: int, transform: np.ndarray, fitness: float
    ) -> None:
        """Apply a verified loop closure as a BetweenFactorPose3.

        ``transform`` maps the ``from_key`` frame into the ``to_key`` frame
        (i.e. ``^{to} T_{from}``), matching BetweenFactor(X(from), X(to))
        semantics. This is a cheap incremental iSAM2 update.
        """
        graph = gtsam.NonlinearFactorGraph()
        graph.push_back(
            gtsam.BetweenFactorPose3(
                X(from_key),
                X(to_key),
                gtsam.Pose3(np.asarray(transform, dtype=np.float64)),
                self._lc_noise(fitness),
            )
        )
        self.isam.update(graph, gtsam.Values())
        # Let relinearization settle: a loop closure perturbs many variables at
        # once, so run a few extra empty updates before refreshing the estimate.
        for _ in range(self._lc_extra_updates):
            self.isam.update(gtsam.NonlinearFactorGraph(), gtsam.Values())
        self._refresh_estimate(best=True)
        # The current keyframe's pose may have shifted after the loop closure.
        if self.index > 0:
            self.current_pose = self._estimate.atPose3(X(self.index - 1))

    def get_all_poses(self) -> Dict[int, gtsam.Pose3]:
        """Return the optimized pose of every keyframe as ``{index: Pose3}``."""
        return {k: self._estimate.atPose3(X(k)) for k in range(self.index)}