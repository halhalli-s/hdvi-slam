"""Keyframe selection policy.

Decides whether enough motion has accumulated since the last keyframe to
justify creating a new one. It reads the motion estimate straight from the IMU
preintegrator (cheap, high-rate, no ICP required) so the expensive front-end
only runs when the platform has actually moved.

What this module does NOT do:
  * It does not integrate IMU data (core.imu_preintegrator).
  * It does not run ICP or touch the graph.
  * It holds no state — the accumulated motion lives in the preintegrator.
"""

from __future__ import annotations

import numpy as np
import gtsam

from core.imu_preintegrator import ImuPreintegrator


def should_trigger(
    preintegrator: ImuPreintegrator, config: dict, current_rotation: gtsam.Rot3
) -> tuple:
    """Evaluate the keyframe trigger; return decision + diagnostics (no print).

    The translation test uses a GRAVITY-CANCELLED estimate, because the raw
    ``deltaPij`` off the PIM still contains uncancelled gravity and balloons on
    a stationary tilted rig (firing spurious keyframes). Gravity cancellation
    needs the current orientation ``R_i`` from the backend. The rotation test is
    gravity-immune and is kept exactly as before.

    Printing is intentionally left to the caller (run_slam) so it can log once
    at fire time (or per-evaluation under ``debug.verbose_trigger``) and include
    the keyframe index. Logic, thresholds, and computation are unchanged.

    Args:
        preintegrator: The active preintegrator.
        config: Parsed config; reads the ``keyframe`` section.
        current_rotation: Orientation ``R_i`` at the start of the current
            window (the last keyframe's optimized rotation).

    Returns:
        ``(fired, deltaTij, raw_norm, cancelled_norm, rot_angle_rad)`` where
        ``fired`` is True if the gravity-cancelled ``||deltaP|| >= translation_m``
        OR the rotation angle ``>= rotation_deg``.
    """
    kf_cfg = config["keyframe"]
    trans_thresh = float(kf_cfg["translation_m"])
    rot_thresh_rad = np.deg2rad(float(kf_cfg["rotation_deg"]))

    cancelled_norm, raw_norm, dt = preintegrator.gravity_cancelled_translation(
        current_rotation
    )
    # Rotation trigger is gravity-immune -> unchanged.
    _, rot_angle = preintegrator.current_delta()

    fired = cancelled_norm >= trans_thresh or rot_angle >= rot_thresh_rad
    return fired, dt, raw_norm, cancelled_norm, rot_angle
