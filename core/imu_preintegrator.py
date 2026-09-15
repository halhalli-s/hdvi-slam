"""IMU preintegration wrapper around GTSAM's PreintegratedImuMeasurements.

Accumulates high-rate (~200 Hz) IMU samples between keyframes into a single
compound measurement (deltaR, deltaP, deltaV) with its covariance, then hands
the back-end a ready-made :class:`gtsam.ImuFactor` connecting two keyframe
states. Between keyframes it also exposes the accumulated motion so the
keyframe trigger can decide when enough has happened to warrant a new node.

What this module does NOT do:
  * It does not own the factor graph or call ``isam.update`` (core.slam_backend).
  * It does not decide when to create a keyframe (core.keyframe_trigger); it
    only reports how far/much it has integrated.
  * It does not read raw hardware — it consumes :class:`~core.types.ImuSample`
    objects produced by a driver.

This class uses the plain :class:`gtsam.ImuFactor` (not the combined factor),
so bias evolution is modeled separately by the back-end via a
``BetweenFactorConstantBias`` random-walk edge.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import gtsam
from gtsam.symbol_shorthand import B, V, X

from core.types import ImuSample


class ImuPreintegrator:
    """Stateful accumulator of IMU measurements between two keyframes."""

    def __init__(self, config: dict) -> None:
        imu_cfg = config["imu"]
        noise = imu_cfg["noise"]

        gravity = float(imu_cfg.get("gravity_magnitude", 9.81))

        # MakeSharedU: navigation frame has gravity pointing in -Z (Z up).
        params = gtsam.PreintegrationParams.MakeSharedU(gravity)

        i3 = np.eye(3)
        params.setGyroscopeCovariance(float(noise["gyro_noise_density"]) ** 2 * i3)
        params.setAccelerometerCovariance(float(noise["accel_noise_density"]) ** 2 * i3)
        params.setIntegrationCovariance(float(noise["integration_sigma"]) ** 2 * i3)

        # Extrinsic: pose of the IMU sensor in the body frame.
        bps = imu_cfg["body_P_sensor"]
        t = np.asarray(bps["translation"], dtype=np.float64)
        qw, qx, qy, qz = bps.get("rotation_quaternion", [1.0, 0.0, 0.0, 0.0])
        body_P_sensor = gtsam.Pose3(
            gtsam.Rot3.Quaternion(qw, qx, qy, qz),
            gtsam.Point3(t[0], t[1], t[2]),
        )
        params.setBodyPSensor(body_P_sensor)

        self._params = params
        # Bias random walk sigmas are handed to the back-end so it can size the
        # BetweenFactorConstantBias edge consistently with this config.
        self.gyro_bias_rw = float(noise["gyro_bias_random_walk"])
        self.accel_bias_rw = float(noise["accel_bias_random_walk"])

        self._bias = gtsam.imuBias.ConstantBias()  # zero initial bias
        self.pim = gtsam.PreintegratedImuMeasurements(self._params, self._bias)

        self._last_ts_ns: Optional[int] = None

    def set_initial_bias(self, bias: gtsam.imuBias.ConstantBias) -> None:
        """Re-base the PIM on a measured startup bias.

        Must be called before the first make_factor_and_reset(). Without this
        the KF0->KF1 window integrates with zero bias regardless of what the
        backend holds, because the PIM is constructed with ConstantBias().
        Discards any accumulated integration.
        """
        self._bias = bias
        self.pim.resetIntegrationAndSetBias(bias)

    def integrate(self, sample: ImuSample) -> None:
        """Fold one IMU sample into the running preintegration.

        The first sample only establishes the time baseline (no dt yet).
        """
        if self._last_ts_ns is None:
            self._last_ts_ns = sample.timestamp_ns
            return

        dt = (sample.timestamp_ns - self._last_ts_ns) * 1e-9
        self._last_ts_ns = sample.timestamp_ns
        if dt <= 0.0:
            # Out-of-order or duplicate timestamp; skip rather than corrupt.
            return

        self.pim.integrateMeasurement(sample.accel, sample.gyro, dt)

    def current_delta(self) -> Tuple[float, float]:
        """Accumulated motion since the last reset, for keyframe triggering.

        Returns:
            ``(translation_norm_m, rotation_angle_rad)`` where the translation
            is ``||deltaPij||`` and the rotation is the geodesic angle of
            ``deltaRij``.
        """
        delta_p = np.asarray(self.pim.deltaPij())
        trans_norm = float(np.linalg.norm(delta_p))

        delta_r = self.pim.deltaRij()  # gtsam.Rot3
        rot_angle = float(np.linalg.norm(gtsam.Rot3.Logmap(delta_r)))
        return trans_norm, rot_angle

    def gravity_cancelled_translation(
        self, R_i: gtsam.Rot3
    ) -> Tuple[float, float, float]:
        """Gravity-cancelled translation estimate for the keyframe TRIGGER only.

        GTSAM's PIM does NOT remove gravity: ``deltaPij`` is the raw
        double-integrated specific force in body-frame i, so on a stationary
        tilted rig it balloons (~0.5*g*dt^2). Gravity is only cancelled later,
        inside the ImuFactor, via the NavState position prediction

            p_j = p_i + v_i*dt + 0.5 * n_gravity * dt^2 + R_i * deltaPij

        (Forster et al., IEEE T-RO 2017, Eq. 33; GTSAM NavState::update /
        ManifoldPreintegration::predict). Here we reproduce only the
        gravity-cancelling part — dropping the unknown ``v_i*dt`` term — purely
        to decide keyframes:

            t_cancelled = R_i @ deltaPij + 0.5 * n_gravity * dt^2

        ``n_gravity`` is the SIGNED world gravity vector from the params
        (``MakeSharedU`` -> ``[0, 0, -g]``), so the gravity term is ADDED.
        Verified to reproduce ``pim.predict()`` to ~1e-14 for a static rig.

        This only READS the PIM; it does not integrate, reset, or otherwise
        modify it. Returns ``(cancelled_norm, raw_norm, dt)``.
        """
        delta_p = np.asarray(self.pim.deltaPij(), dtype=np.float64)
        dt = float(self.pim.deltaTij())
        n_gravity = np.asarray(self._params.n_gravity, dtype=np.float64)
        t_cancelled = R_i.matrix() @ delta_p + 0.5 * n_gravity * dt * dt
        return (
            float(np.linalg.norm(t_cancelled)),
            float(np.linalg.norm(delta_p)),
            dt,
        )

    def predicted_translation(
        self,
        R_i: gtsam.Rot3,
        v_i: np.ndarray,
        meas: Optional[gtsam.PreintegratedImuMeasurements] = None,
    ) -> np.ndarray:
        """Full IMU-predicted translation ``^prev t_curr`` — for the ICP seed.

        Unlike :meth:`gravity_cancelled_translation` (trigger-only, which
        deliberately drops the velocity term), this reproduces the complete
        NavState position delta including ``v_i``:

            t = R_i @ deltaPij + v_i * dt + 0.5 * n_gravity * dt^2

        (Forster et al., IEEE T-RO 2017, Eq. 33; GTSAM NavState::update /
        ManifoldPreintegration::predict). ``n_gravity`` is the SIGNED world
        gravity vector from the params (``MakeSharedU`` -> ``[0, 0, -g]``), so
        the gravity term is ADDED. Read-only: no integrate, no reset.

        ``meas`` selects which measurement to read; it defaults to the live PIM,
        but the odometry ICP passes the pre-reset snapshot from the ImuFactor
        because ``self.pim`` is already reset by :meth:`make_factor_and_reset`
        by the time the seed is built. Returns the translation, shape ``(3,)``.
        """
        pim = self.pim if meas is None else meas
        delta_p = np.asarray(pim.deltaPij(), dtype=np.float64)
        dt = float(pim.deltaTij())
        v_i = np.asarray(v_i, dtype=np.float64).reshape(3)
        n_gravity = np.asarray(self._params.n_gravity, dtype=np.float64)
        return R_i.matrix() @ delta_p + v_i * dt + 0.5 * n_gravity * dt * dt

    def make_factor_and_reset(
        self, key_i: int, key_j: int, current_bias: gtsam.imuBias.ConstantBias
    ) -> gtsam.ImuFactor:
        """Build the ImuFactor for [key_i -> key_j] and reset the integrator.

        The factor ties pose/velocity at ``key_i`` and ``key_j`` through the
        bias estimated at ``key_i``. After building it, integration is reset
        and re-based on ``current_bias`` so the next window starts fresh.
        """
        factor = gtsam.ImuFactor(
            X(key_i), V(key_i), X(key_j), V(key_j), B(key_i), self.pim
        )
        self._bias = current_bias
        self.pim.resetIntegrationAndSetBias(current_bias)
        return factor

    @property
    def measurements(self) -> gtsam.PreintegratedImuMeasurements:
        """Direct access to the underlying PIM (e.g. for state prediction)."""
        return self.pim
