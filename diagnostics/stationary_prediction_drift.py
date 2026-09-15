#!/usr/bin/env python3
"""Standalone diagnostic: does seeding IMU bias reduce stationary prediction drift?

The smoking-gun before/after test. With the camera held still, IMU preintegration
should predict ~zero motion — any predicted drift is bias (and noise) integrated
as fake motion. Three phases:
  (A) integrate a stationary window with ZERO bias (mimics the current pipeline),
  (B) measure turn-on biases from a 30 s stationary window,
  (C) integrate a fresh stationary window with the MEASURED bias seeded.
Reports position/velocity/orientation drift at 2 s / 5 s / 10 s horizons for A vs C.

What this does NOT do:
  * It does not modify the driver, core, config, or the SLAM pipeline.
  * It uses the pipeline's real PreintegrationParams (via ImuPreintegrator) and
    _gravity_to_initial_pose, so results reflect the actual estimator math.
  * The measured accel bias is entangled with tilt error (see measure_bias.py) —
    treat the accel component as approximate. TEMPORARY diagnostic ([DIAG], §9).
"""

from __future__ import annotations

import os
import sys
import time
import threading

import numpy as np
import yaml
import gtsam

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from drivers.camera_driver import GeminiDriver
from core.types import ImuSample  # noqa: F401 - spec-mandated import
from core.imu_preintegrator import ImuPreintegrator
from scripts.run_slam import _gravity_to_initial_pose

DEFAULT_CONFIG = os.path.join(_ROOT, "config", "gemini_435le.yaml")

DRIFT_S = 11.0      # per drift window (>10 so the 10 s horizon is reached)
BIAS_S = 30.0       # stationary window for bias measurement
HORIZONS = (2.0, 5.0, 10.0)


def _countdown(msg: str) -> None:
    print(f"[DIAG] {msg}")
    for c in (3, 2, 1):
        print(f"[DIAG]   {c}...", flush=True)
        time.sleep(1.0)


def _collect(driver: GeminiDriver, duration_s: float):
    """Stream IMU for duration_s; return (accels Nx3, gyros Nx3, ts_ns N)."""
    accels: list = []
    gyros: list = []
    ts: list = []

    def on_imu(s: ImuSample) -> None:
        accels.append(s.accel)
        gyros.append(s.gyro)
        ts.append(s.timestamp_ns)

    driver.start(depth_callback=lambda frame: None, imu_callback=on_imu)
    try:
        threading.Event().wait(duration_s)
    finally:
        driver.stop()
    return (
        np.asarray(accels, dtype=np.float64),
        np.asarray(gyros, dtype=np.float64),
        np.asarray(ts, dtype=np.int64),
    )


def _drift(params, accels, gyros, ts, bias, label: str) -> None:
    """Integrate a stationary window and report predicted drift at each horizon."""
    if len(ts) < 2:
        print(f"[DIAG] {label}: insufficient samples ({len(ts)}).", flush=True)
        return
    T0, roll, pitch = _gravity_to_initial_pose(accels.mean(axis=0))
    pose0 = gtsam.Pose3(T0)
    state0 = gtsam.NavState(pose0, np.zeros(3))
    pim = gtsam.PreintegratedImuMeasurements(params, bias)
    print(f"[DIAG] --- {label} (init tilt roll={roll:.2f} pitch={pitch:.2f}) ---", flush=True)

    done: set = set()
    elapsed = 0.0
    for k in range(1, len(ts)):
        dt = (ts[k] - ts[k - 1]) * 1e-9
        if dt > 0:
            pim.integrateMeasurement(accels[k], gyros[k], dt)
        elapsed = (ts[k] - ts[0]) * 1e-9
        for h in HORIZONS:
            if h not in done and elapsed >= h:
                done.add(h)
                nav = pim.predict(state0, bias)
                pos = float(np.linalg.norm(np.asarray(nav.position())))
                vel = float(np.linalg.norm(np.asarray(nav.velocity())))
                ori = float(np.degrees(np.linalg.norm(
                    gtsam.Rot3.Logmap(pose0.rotation().between(nav.attitude())))))
                print(f"[DIAG]   t={h:.0f}s: pos_drift={pos:.3f}m "
                      f"vel_drift={vel:.3f}m/s ori_drift={ori:.2f}deg", flush=True)
    for h in HORIZONS:
        if h not in done:
            print(f"[DIAG]   t={h:.0f}s: not reached (only {elapsed:.1f}s collected)", flush=True)


def main() -> None:
    with open(DEFAULT_CONFIG, "r") as f:
        config = yaml.safe_load(f)
    g_mag = float(config["imu"].get("gravity_magnitude", 9.81))
    # Reuse the pipeline's exact preintegration params (gravity/noise/extrinsic).
    params = ImuPreintegrator(config)._params  # noqa: SLF001 - read-only diagnostic

    driver = GeminiDriver(DEFAULT_CONFIG)
    print("[DIAG] Stationary prediction-drift test. Keep the camera COMPLETELY still")
    print("[DIAG] on a stable surface for the whole run (~1.5 min, 3 phases).")
    try:
        _countdown("PHASE A (zero-bias drift, mimics current pipeline) starting in")
        aA, gA, tA = _collect(driver, DRIFT_S)
        print(f"[DIAG] Phase A: {len(tA)} samples", flush=True)
        _drift(params, aA, gA, tA, gtsam.imuBias.ConstantBias(), "PHASE A: zero bias")

        _countdown("PHASE B (bias measurement) starting in")
        aB, gB, tB = _collect(driver, BIAS_S)
        if len(tB) == 0:
            print("[DIAG] Phase B: no samples — cannot measure bias. Aborting.", flush=True)
            return
        gyro_bias = gB.mean(axis=0)
        mean_accel = aB.mean(axis=0)
        accel_bias = mean_accel - g_mag * (mean_accel / np.linalg.norm(mean_accel))
        print(f"[DIAG] measured gyro bias (rad/s): "
              f"[{gyro_bias[0]:.6f}, {gyro_bias[1]:.6f}, {gyro_bias[2]:.6f}]", flush=True)
        print(f"[DIAG] approx accel bias (m/s^2, tilt-entangled): "
              f"[{accel_bias[0]:.4f}, {accel_bias[1]:.4f}, {accel_bias[2]:.4f}]", flush=True)
        bias = gtsam.imuBias.ConstantBias(accel_bias, gyro_bias)

        _countdown("PHASE C (measured-bias drift) starting in")
        aC, gC, tC = _collect(driver, DRIFT_S)
        print(f"[DIAG] Phase C: {len(tC)} samples", flush=True)
        _drift(params, aC, gC, tC, bias, "PHASE C: measured bias seeded")

        print("[DIAG] Compare A vs C: if C's pos/ori drift is materially smaller, "
              "bias seeding will improve early-run pose accuracy.", flush=True)
    except KeyboardInterrupt:
        print("[DIAG] interrupted — stopping.", flush=True)
    finally:
        driver.stop()


if __name__ == "__main__":
    main()
