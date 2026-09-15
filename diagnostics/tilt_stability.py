#!/usr/bin/env python3
"""Standalone diagnostic: is the 1-second gravity/tilt init stable enough?

Repeats the pipeline's gravity-averaging + tilt-conversion (the exact
scripts.run_slam._gravity_to_initial_pose used to seed KF 0) many times per
window length, with the camera held still, and reports how much the resulting
roll/pitch estimate varies. Compares 1 s / 5 s / 15 s windows so we can decide
whether to extend collect_static_gravity() before seeding the initial pose.

What this does NOT do:
  * It does not modify the driver, core, config, or the SLAM pipeline.
  * It tests only the STABILITY (repeatability) of the roll/pitch estimate — it
    does NOT measure ACCURACY against ground truth. A truly level reference /
    external attitude source would be needed to judge absolute correctness.
  * TEMPORARY diagnostic (greppable via [DIAG], per project_info §9).
"""

from __future__ import annotations

import os
import sys
import time
import threading

import numpy as np
import yaml
import gtsam  # noqa: F401 - used transitively by _gravity_to_initial_pose

# Add project root so the pipeline's own tilt math is importable verbatim.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from drivers.camera_driver import GeminiDriver
from core.types import ImuSample  # noqa: F401 - spec-mandated import
from scripts.run_slam import _gravity_to_initial_pose

DEFAULT_CONFIG = os.path.join(_ROOT, "config", "gemini_435le.yaml")

TRIALS = 20
WINDOW_SIZES_S = [1.0, 5.0, 15.0]


def _countdown(msg: str) -> None:
    print(f"[DIAG] {msg}")
    for c in (3, 2, 1):
        print(f"[DIAG]   {c}...", flush=True)
        time.sleep(1.0)


def _std(vals: list) -> float:
    return float(np.std(vals)) if vals else float("nan")


def _report_window(n: float, rolls: list, pitches: list) -> None:
    if not rolls:
        print(f"[DIAG] window {n:g}s: no successful trials.", flush=True)
        return
    for name, vals in (("roll ", rolls), ("pitch", pitches)):
        a = np.asarray(vals)
        print(
            f"[DIAG] {name}: mean={a.mean():.2f} deg  std={a.std():.2f} deg  "
            f"min={a.min():.2f}  max={a.max():.2f}  range={a.max() - a.min():.2f}",
            flush=True,
        )


def _summary_and_verdict(results: dict) -> None:
    r1, p1 = results.get(1.0, ([], []))
    rs1, ps1 = _std(r1), _std(p1)
    if r1:
        if rs1 < 0.1 and ps1 < 0.1:
            print("[DIAG] VERDICT: 1-second gravity init is stable enough — no need to extend.")
        elif rs1 > 0.5 or ps1 > 0.5:
            print("[DIAG] VERDICT: 1-second gravity init is UNSTABLE. Compare against longer "
                  "windows below. Extending collect_static_gravity() to the smallest stable "
                  "window will materially reduce initial-pose noise.")
        else:
            print("[DIAG] VERDICT: 1-second init is marginal — longer windows show meaningful "
                  "improvement.")
    else:
        print("[DIAG] VERDICT: no 1s-window data collected.")

    print("[DIAG] SUMMARY:")
    rec = None
    for n in WINDOW_SIZES_S:
        rolls, pitches = results.get(n, ([], []))
        rs, ps = _std(rolls), _std(pitches)
        print(f"[DIAG]  {n:>2g}s window:  roll_std={rs:.2f}°  pitch_std={ps:.2f}°", flush=True)
        if rec is None and rolls and rs < 0.1 and ps < 0.1:
            rec = n
    if rec is not None:
        print(f"[DIAG]   Recommendation: use {rec:g}s window (smallest with std < 0.1°).")
    else:
        print("[DIAG]   Recommendation: none of the tested windows reached std < 0.1° — "
              "consider a longer window or investigate sensor noise.")


def main() -> None:
    with open(DEFAULT_CONFIG, "r") as f:
        yaml.safe_load(f)  # validate the config path parses; values unused here

    # collect_static_gravity() opens & closes its OWN accel-only pipeline each
    # call, so we do NOT call driver.start()/stop() around it (that would open a
    # second, conflicting pipeline). One driver instance is reused across trials.
    driver = GeminiDriver(DEFAULT_CONFIG)

    print("[DIAG] Tilt-stability test: place the camera on a stable, LEVEL surface and")
    print("[DIAG] DO NOT touch it for the full run (~7-10 min across 3 window sizes).")

    results: dict = {}
    try:
        for n in WINDOW_SIZES_S:
            print(f"[DIAG] === Testing window size: {n:g} s ({TRIALS} trials) ===", flush=True)
            _countdown(f"starting {n:g}s window in")
            rolls: list = []
            pitches: list = []
            results[n] = (rolls, pitches)  # live ref so partial data survives Ctrl-C
            for k in range(1, TRIALS + 1):
                print(f"[DIAG] trial {k}/{TRIALS}...", flush=True)
                try:
                    accel = driver.collect_static_gravity(n)
                    _T, roll, pitch = _gravity_to_initial_pose(accel)
                    rolls.append(roll)
                    pitches.append(pitch)
                except Exception as exc:  # noqa: BLE001 - one bad trial != abort
                    print(f"[DIAG] trial failed: {exc}", flush=True)
            _report_window(n, rolls, pitches)
    except KeyboardInterrupt:
        print("[DIAG] interrupted — reporting collected stats so far.", flush=True)
        driver.stop()  # safe no-op if no pipeline is currently open

    _summary_and_verdict(results)


if __name__ == "__main__":
    main()
