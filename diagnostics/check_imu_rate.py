#!/usr/bin/env python3
"""Standalone diagnostic: measure the actual IMU sample rate at the callback.

Runs the real GeminiDriver for 10 s and counts IMU samples delivered to the
Python callback, to verify whether the IMU streams at its native ~200 Hz or is
throttled to depth rate (~15 Hz) by the driver's FULL_FRAME_REQUIRE aggregate
mode. Reports effective Hz + inter-sample dt stats, then a verdict.
Does NOT modify the driver/core/config/pipeline, does no accel/gyro processing
(only timestamps), and is a TEMPORARY diagnostic (greppable via [DIAG], §9).
"""

from __future__ import annotations

import os
import threading
import time

import numpy as np
import yaml

from drivers.camera_driver import GeminiDriver
from core.types import ImuSample

DEFAULT_CONFIG = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "config",
    "gemini_435le.yaml",
)

RUNTIME_S = 10.0


def main() -> None:
    with open(DEFAULT_CONFIG, "r") as f:
        config = yaml.safe_load(f)
    expected_hz = config["imu"]["rate_hz"]
    timestamps: list[int] = []

    def on_imu(sample: ImuSample) -> None:
        timestamps.append(sample.timestamp_ns)

    driver = GeminiDriver(DEFAULT_CONFIG)
    print(f"[DIAG] collecting IMU for {RUNTIME_S:.0f}s (expected {expected_hz} Hz)...", flush=True)
    t0 = time.perf_counter()
    driver.start(depth_callback=lambda frame: None, imu_callback=on_imu)
    try:
        threading.Event().wait(RUNTIME_S)
    except KeyboardInterrupt:
        print("[DIAG] interrupted — stopping early.", flush=True)
    finally:
        driver.stop()
    duration = time.perf_counter() - t0

    n = len(timestamps)
    print(f"[DIAG] total samples received: {n}")
    print(f"[DIAG] wall-clock duration: {duration:.3f} s")
    if n == 0:
        print("[DIAG] no samples — cannot compute rate.")
        return
    eff_hz = n / duration
    print(f"[DIAG] effective rate: {eff_hz:.1f} Hz")
    if n >= 2:
        dts_ms = np.diff(np.asarray(timestamps, dtype=np.float64)) * 1e-6
        print(
            f"[DIAG] inter-sample dt (ms): mean={dts_ms.mean():.3f} "
            f"std={dts_ms.std():.3f} min={dts_ms.min():.3f} max={dts_ms.max():.3f}"
        )
    span_s = (timestamps[-1] - timestamps[0]) * 1e-9
    print(f"[DIAG] timestamp span: {span_s:.3f} s (wall clock: {duration:.3f} s)")
    print(f"[DIAG] coverage: {100 * span_s / duration:.1f}%")
    if eff_hz > 150:
        print("[DIAG] VERDICT: IMU running at native rate (~200Hz) — no throttling.")
    elif eff_hz < 30:
        print("[DIAG] VERDICT: IMU is THROTTLED to depth rate (~15Hz). "
              "FULL_FRAME_REQUIRE is bottlenecking IMU. This is a real problem — "
              "noise densities in config assume 200Hz.")
    else:
        print("[DIAG] VERDICT: IMU running at unusual rate — investigate driver aggregation.")


if __name__ == "__main__":
    main()
