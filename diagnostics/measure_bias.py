#!/usr/bin/env python3
"""Standalone diagnostic: measure IMU turn-on biases while stationary.

Collects RUNTIME_S seconds of accel/gyro with the camera held completely still,
then reports the mean gyro vector (the raw gyro bias) and an APPROXIMATE accel
bias, plus tilt and noise stats. Purpose: obtain real bias values to seed into
SlamBackend.current_bias at KF 0 instead of assuming zero.

What this does NOT do:
  * It does not modify the driver, core, config, or the SLAM pipeline.
  * It does not itself change current_bias — it only measures and reports.
  * The "approximate accel bias" is ENTANGLED with tilt error: without a known
    reference orientation (or multi-orientation calibration) accel bias and a
    small mounting tilt cannot be perfectly separated — treat it as a rough
    estimate. TEMPORARY diagnostic (greppable via [DIAG], per project_info §9).
"""

from __future__ import annotations

import os
import time
import threading

import numpy as np
import yaml

from drivers.camera_driver import GeminiDriver
from core.types import ImuSample

DEFAULT_CONFIG = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "config",
    "gemini_435le.yaml",
)

RUNTIME_S = 30.0


def main() -> None:
    with open(DEFAULT_CONFIG, "r") as f:
        config = yaml.safe_load(f)
    g_mag = float(config["imu"].get("gravity_magnitude", 9.81))

    accels: list = []
    gyros: list = []
    timestamps: list = []

    def on_imu(sample: ImuSample) -> None:
        accels.append(sample.accel)
        gyros.append(sample.gyro)
        timestamps.append(sample.timestamp_ns)

    driver = GeminiDriver(DEFAULT_CONFIG)
    print("[DIAG] Place the camera on a stable, LEVEL surface and DO NOT touch it")
    print(f"[DIAG] for the full {RUNTIME_S:.0f}s collection.")
    for c in (3, 2, 1):
        print(f"[DIAG]   {c}...", flush=True)
        time.sleep(1.0)
    print("[DIAG] ...starting collection.", flush=True)

    t0 = time.perf_counter()
    driver.start(depth_callback=lambda frame: None, imu_callback=on_imu)
    try:
        threading.Event().wait(RUNTIME_S)
    except KeyboardInterrupt:
        print("[DIAG] interrupted — stopping early (fewer samples collected).", flush=True)
    finally:
        driver.stop()
    duration = time.perf_counter() - t0

    n = len(accels)
    print(f"[DIAG] total samples: {n}")
    print(f"[DIAG] wall-clock duration: {duration:.3f} s")
    if n == 0:
        print("[DIAG] no samples collected — cannot compute biases.")
        return
    print(f"[DIAG] effective rate: {n / duration:.1f} Hz")

    accel = np.asarray(accels, dtype=np.float64)
    gyro = np.asarray(gyros, dtype=np.float64)
    mean_accel = accel.mean(axis=0)
    mean_gyro = gyro.mean(axis=0)

    # --- Accelerometer ---
    a_norm = float(np.linalg.norm(mean_accel))
    print(f"[DIAG] mean accel (m/s^2): [{mean_accel[0]:.3f}, {mean_accel[1]:.3f}, {mean_accel[2]:.3f}]")
    print(f"[DIAG] |mean accel| = {a_norm:.3f} m/s^2  (|norm - {g_mag:.2f}| = {abs(a_norm - g_mag):.3f})")
    if abs(a_norm - g_mag) > 0.1:
        print("[DIAG] WARNING: |accel| far from gravity — accelerometer scale factor may be off.")
    astd = accel.std(axis=0)
    print(f"[DIAG] accel per-axis std (m/s^2): [{astd[0]:.4f}, {astd[1]:.4f}, {astd[2]:.4f}]")

    # --- Gyro (mean == raw bias) ---
    print(f"[DIAG] MEASURED GYRO BIAS (rad/s): [{mean_gyro[0]:.6f}, {mean_gyro[1]:.6f}, {mean_gyro[2]:.6f}]")
    gdeg = np.degrees(mean_gyro)
    print(f"[DIAG] gyro bias in deg/s: [{gdeg[0]:.4f}, {gdeg[1]:.4f}, {gdeg[2]:.4f}]")
    gstd = gyro.std(axis=0)
    print(f"[DIAG] gyro per-axis std (rad/s): [{gstd[0]:.6f}, {gstd[1]:.6f}, {gstd[2]:.6f}]")

    # --- Tilt from mean accel (Y=down convention) ---
    roll = np.degrees(np.arctan2(mean_accel[0], mean_accel[1]))
    pitch = np.degrees(np.arctan2(mean_accel[2], mean_accel[1]))
    print(f"[DIAG] estimated tilt: roll={roll:.2f}deg pitch={pitch:.2f}deg")

    # --- Approx accel bias (entangled with tilt) ---
    # Static specific force should equal -gravity_body == g_mag * up_dir, where
    # up_dir is the measured accel direction. Residual is a rough bias estimate.
    up_dir = mean_accel / a_norm
    accel_bias = mean_accel - g_mag * up_dir
    print(f"[DIAG] APPROX ACCEL BIAS (m/s^2): [{accel_bias[0]:.4f}, {accel_bias[1]:.4f}, {accel_bias[2]:.4f}]")
    print("[DIAG] NOTE: approx accel bias is entangled with tilt error — rough estimate only.")

    # --- Verdict ---
    gyro_small = bool(np.all(np.abs(mean_gyro) < 0.001))
    accel_small = bool(np.all(np.abs(accel_bias) < 0.05))
    gyro_big = bool(np.any(np.abs(mean_gyro) > 0.01))
    accel_big = bool(np.any(np.abs(accel_bias) > 0.1))
    if gyro_small and accel_small:
        print("[DIAG] VERDICT: biases are small — bias seeding may not help much.")
    elif gyro_big or accel_big:
        print("[DIAG] VERDICT: significant biases detected. Seeding these into current_bias "
              "at KF 0 should materially improve early-run pose accuracy.")
    else:
        print("[DIAG] VERDICT: moderate biases — seeding is worth doing.")


if __name__ == "__main__":
    main()
