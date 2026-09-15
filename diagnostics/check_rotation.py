#!/usr/bin/env python3
"""Standalone diagnostic: gyro-integrated orientation vs the accel gravity ref.

Seeds gyro bias and the initial orientation EXACTLY as the pipeline does (reuses
GeminiDriver.collect_static_imu for the static window, and
scripts.run_slam._gravity_to_initial_pose for the gravity tilt — no math is
re-derived here), then integrates gyro into an orientation estimate using the
pipeline's convention and prints, at ~5 Hz:

  * gyro-integrated roll/pitch/yaw (deg),
  * accelerometer-derived roll/pitch (deg) — gravity is an absolute reference
    for those two; yaw is unobservable from gravity, so no accel yaw is shown,
  * their roll/pitch divergence, and elapsed time so drift is visible.

Gyro roll/pitch drift over time; accel roll/pitch do not — the divergence over
30-60 s is the trust number. Yaw has no reference: its drift is only visible by
physically returning to a known heading and reading the gyro yaw.

What this does NOT do:
  * It does not modify the driver, core, config, or the SLAM pipeline.
  * It does not run ICP, the backend, or depth processing (depth callback is a
    no-op). TEMPORARY diagnostic (greppable via [DIAG], per project_info §9).

Runs until Ctrl-C.
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import gtsam

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from drivers.camera_driver import GeminiDriver
from core.types import ImuSample  # noqa: F401 - imu_callback sample type
from scripts.run_slam import _gravity_to_initial_pose

DEFAULT_CONFIG = os.path.join(_ROOT, "config", "gemini_435le.yaml")

PRINT_HZ = 5.0


def _accel_roll_pitch_deg(accel: np.ndarray):
    """Accel gravity-align rotation -> (roll, pitch) in deg, gtsam RPY convention.

    Reuses the pipeline's _gravity_to_initial_pose (shortest-arc that maps the
    measured 'up' onto world +Z), then reads roll/pitch off the SAME Rot3 that
    seeds the gyro estimate — so the two are directly comparable (yaw from the
    shortest arc is ~0 and unobservable, hence not returned).
    """
    T, _roll_nominal, _pitch_nominal = _gravity_to_initial_pose(accel)
    R = gtsam.Rot3(np.asarray(T, dtype=np.float64)[:3, :3])
    return np.degrees(R.roll()), np.degrees(R.pitch())


def main() -> None:
    driver = GeminiDriver(DEFAULT_CONFIG)

    # --- Seed bias + initial orientation exactly as the pipeline does ---------
    window = float(driver.config["imu"].get("startup_bias_window_s", 10.0))
    print(
        f"[DIAG] hold the camera STILL for the {window:.0f}s bias/tilt window...",
        flush=True,
    )
    mean_accel, mean_gyro, _std_accel, _std_gyro = driver.collect_static_imu(window)
    # Pipeline convention: gyro turn-on bias == mean gyro over the static window.
    gyro_bias = np.asarray(mean_gyro, dtype=np.float64)
    T0, roll0, pitch0 = _gravity_to_initial_pose(mean_accel)
    R0 = gtsam.Rot3(np.asarray(T0, dtype=np.float64)[:3, :3])
    print(
        f"[DIAG] gyro bias (rad/s): "
        f"[{gyro_bias[0]:.6f}, {gyro_bias[1]:.6f}, {gyro_bias[2]:.6f}]",
        flush=True,
    )
    print(f"[DIAG] initial tilt (nominal): roll={roll0:.2f} pitch={pitch0:.2f} deg", flush=True)

    # --- Shared state: written on the SDK callback thread, read on main. It's a
    # diagnostic and the writes are atomic single-object rebinds under the GIL,
    # so no locking is used.
    state = {"R": R0, "accel": np.asarray(mean_accel, dtype=np.float64), "last_ts_ns": None}

    def on_imu(s: ImuSample) -> None:
        state["accel"] = s.accel
        if state["last_ts_ns"] is None:
            state["last_ts_ns"] = s.timestamp_ns
            return
        dt = (s.timestamp_ns - state["last_ts_ns"]) * 1e-9
        state["last_ts_ns"] = s.timestamp_ns
        if dt <= 0.0:
            return
        # Bias-corrected body angular rate; right-multiply, matching the PIM /
        # pipeline convention R_{k+1} = R_k * Exp((gyro - bias) * dt).
        omega = np.asarray(s.gyro, dtype=np.float64) - gyro_bias
        state["R"] = state["R"].compose(gtsam.Rot3.Expmap(omega * dt))

    t0 = time.perf_counter()
    driver.start(depth_callback=lambda frame: None, imu_callback=on_imu)
    print(
        "[DIAG] integrating gyro. Rotate the rig and return to start to expose "
        "yaw drift. Ctrl-C to stop.",
        flush=True,
    )
    try:
        while True:
            time.sleep(1.0 / PRINT_HZ)
            elapsed = time.perf_counter() - t0
            R = state["R"]
            g_roll = np.degrees(R.roll())
            g_pitch = np.degrees(R.pitch())
            g_yaw = np.degrees(R.yaw())
            a_roll, a_pitch = _accel_roll_pitch_deg(state["accel"])
            print(
                f"[DIAG-ROT] t={elapsed:6.1f}s "
                f"gyro(rpy)=[{g_roll:+7.2f},{g_pitch:+7.2f},{g_yaw:+7.2f}] "
                f"accel(rp)=[{a_roll:+7.2f},{a_pitch:+7.2f}] "
                f"drift(rp)=[{g_roll - a_roll:+6.2f},{g_pitch - a_pitch:+6.2f}]deg",
                flush=True,
            )
    except KeyboardInterrupt:
        print("[DIAG] stopping.", flush=True)
    finally:
        driver.stop()


if __name__ == "__main__":
    main()
