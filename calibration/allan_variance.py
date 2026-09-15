#!/usr/bin/env python3
"""Compute Allan deviation from a static IMU log and derive noise parameters.

Consumes a CSV produced by ``record_imu.py`` (columns
``timestamp_ns,gx,gy,gz,ax,ay,az``, recorded with the sensor stationary) and:

  1. Computes the non-overlapping Allan deviation for cluster sizes that are
     powers of two, up to N/10.
  2. Converts cluster sizes to averaging time tau using the actual sample rate
     recovered from the timestamps.
  3. Extracts, per axis:
       - noise density N            : Allan deviation at tau = 1 s
                                       (angle/velocity random walk).
       - bias instability B         : the minimum of the curve.
       - bias random walk K         : from the right (+1/2 slope) side,
                                       K = sigma * sqrt(3) / sqrt(tau).
  4. Prints the four config values (worst axis for each), ready to paste into
     config/gemini_435le.yaml.
  5. Plots the six curves on two log-log subplots and saves a PNG.

This is an offline analysis tool: it imports numpy and matplotlib only — no
sensor SDK, no gtsam. It knows nothing about the SLAM pipeline.

Units: results come out in whatever units the log is in. The Orbbec gyro is
reported in rad/s and the accel in m/s^2, giving:
    gyro_noise_density      [rad/s/sqrt(Hz)]
    accel_noise_density     [m/s^2/sqrt(Hz)]
    gyro_bias_random_walk   [rad/s^2/sqrt(Hz)]
    accel_bias_random_walk  [m/s^3/sqrt(Hz)]

Usage:
    python calibration/allan_variance.py calibration/data/static_YYYYMMDD_HHMMSS.csv
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime

import matplotlib

matplotlib.use("Agg")  # headless: we only save a PNG
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
AXES = ["gx", "gy", "gz", "ax", "ay", "az"]


def load_csv(path: str) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Return (timestamp_ns, {axis: samples})."""
    data = np.genfromtxt(path, delimiter=",", names=True)
    ts = np.asarray(data["timestamp_ns"], dtype=np.float64)
    series = {ax: np.asarray(data[ax], dtype=np.float64) for ax in AXES}
    return ts, series


def sample_rate_hz(timestamp_ns: np.ndarray) -> float:
    """Recover the actual mean sample rate from the timestamps."""
    span_s = (timestamp_ns[-1] - timestamp_ns[0]) * 1e-9
    if span_s <= 0:
        raise ValueError("Non-increasing timestamps; cannot compute sample rate.")
    return (len(timestamp_ns) - 1) / span_s


def cluster_sizes(n: int) -> np.ndarray:
    """Powers of two from 1 up to N/10 (need >=2 clusters to difference)."""
    max_m = max(1, n // 10)
    sizes = []
    m = 1
    while m <= max_m and (n // m) >= 2:
        sizes.append(m)
        m *= 2
    return np.asarray(sizes, dtype=int)


def allan_deviation(x: np.ndarray, sizes: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Non-overlapping Allan deviation of signal ``x``.

    For each cluster size m: split into non-overlapping groups of m samples,
    average each group, take consecutive differences of the group means,
    square, average, divide by 2, sqrt.

    Returns (tau, adev) arrays aligned with ``sizes``.
    """
    n = len(x)
    taus = []
    adevs = []
    for m in sizes:
        k = n // m  # number of full clusters
        if k < 2:
            continue
        # Group means of the first k*m samples.
        means = x[: k * m].reshape(k, m).mean(axis=1)
        diffs = np.diff(means)
        avar = 0.5 * np.mean(diffs**2)
        taus.append(m * dt)
        adevs.append(np.sqrt(avar))
    return np.asarray(taus), np.asarray(adevs)


def noise_density_at_1s(tau: np.ndarray, adev: np.ndarray) -> float:
    """Allan deviation at tau = 1 s (log-log interpolation)."""
    log_tau = np.log10(tau)
    log_adev = np.log10(adev)
    return float(10.0 ** np.interp(0.0, log_tau, log_adev))  # log10(1s) == 0


def bias_instability(adev: np.ndarray) -> tuple[float, float]:
    """Minimum of the curve; returns (value, tau_at_min index-based value)."""
    idx = int(np.argmin(adev))
    return float(adev[idx]), idx


def bias_random_walk(tau: np.ndarray, adev: np.ndarray, min_idx: int) -> float:
    """Rate random walk K from the right (+1/2 slope) side of the curve.

    Uses the rightmost available point (largest tau, past the bias-instability
    floor): K = sigma * sqrt(3) / sqrt(tau).
    """
    # Prefer a point clearly to the right of the minimum; fall back to the last.
    right_idx = len(tau) - 1
    if right_idx <= min_idx:
        right_idx = len(tau) - 1
    sigma = adev[right_idx]
    t = tau[right_idx]
    return float(sigma * np.sqrt(3.0) / np.sqrt(t))


def main() -> None:
    parser = argparse.ArgumentParser(description="Allan variance from an IMU CSV.")
    parser.add_argument("csv", help="Path to a static_*.csv log.")
    args = parser.parse_args()

    ts, series = load_csv(args.csv)
    fs = sample_rate_hz(ts)
    dt = 1.0 / fs
    n = len(ts)
    sizes = cluster_sizes(n)

    print(f"Loaded {n} samples from {args.csv}")
    print(f"Actual sample rate: {fs:.2f} Hz (dt = {dt*1e3:.3f} ms)")
    print(f"Cluster sizes: {sizes.tolist()}\n")

    results: dict[str, dict[str, float]] = {}
    curves: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for ax in AXES:
        tau, adev = allan_deviation(series[ax], sizes, dt)
        curves[ax] = (tau, adev)
        n_density = noise_density_at_1s(tau, adev)
        b_instab, min_idx = bias_instability(adev)
        k_rw = bias_random_walk(tau, adev, min_idx)
        results[ax] = {
            "noise_density": n_density,
            "bias_instability": b_instab,
            "bias_random_walk": k_rw,
        }

    # --- Worst axis for each config value ----------------------------------
    gyro_axes = ["gx", "gy", "gz"]
    accel_axes = ["ax", "ay", "az"]

    gyro_noise_density = max(results[a]["noise_density"] for a in gyro_axes)
    accel_noise_density = max(results[a]["noise_density"] for a in accel_axes)
    gyro_bias_random_walk = max(results[a]["bias_random_walk"] for a in gyro_axes)
    accel_bias_random_walk = max(results[a]["bias_random_walk"] for a in accel_axes)

    # --- Per-axis detail ----------------------------------------------------
    print("Per-axis results (noise_density @1s | bias_instability | bias_random_walk):")
    for ax in AXES:
        r = results[ax]
        print(
            f"  {ax}: {r['noise_density']:.6g} | "
            f"{r['bias_instability']:.6g} | {r['bias_random_walk']:.6g}"
        )

    print("\n" + "=" * 60)
    print("Config values for config/gemini_435le.yaml (worst axis each):")
    print("=" * 60)
    print(f"    gyro_noise_density:     {gyro_noise_density:.6g}   # rad/s/sqrt(Hz)")
    print(f"    accel_noise_density:    {accel_noise_density:.6g}   # m/s^2/sqrt(Hz)")
    print(f"    gyro_bias_random_walk:  {gyro_bias_random_walk:.6g}   # rad/s^2/sqrt(Hz)")
    print(f"    accel_bias_random_walk: {accel_bias_random_walk:.6g}   # m/s^3/sqrt(Hz)")
    print("=" * 60, flush=True)

    # --- Plot ---------------------------------------------------------------
    fig, (ax_gyro, ax_accel) = plt.subplots(1, 2, figsize=(14, 6))

    for ax in gyro_axes:
        tau, adev = curves[ax]
        ax_gyro.loglog(tau, adev, marker="o", markersize=3, label=ax)
    ax_gyro.set_title("Gyroscope Allan Deviation")
    ax_gyro.set_xlabel(r"$\tau$ [s]")
    ax_gyro.set_ylabel(r"$\sigma(\tau)$ [rad/s]")
    ax_gyro.grid(True, which="both", ls=":", alpha=0.5)
    ax_gyro.axvline(1.0, color="gray", ls="--", alpha=0.5)
    ax_gyro.legend()

    for ax in accel_axes:
        tau, adev = curves[ax]
        ax_accel.loglog(tau, adev, marker="o", markersize=3, label=ax)
    ax_accel.set_title("Accelerometer Allan Deviation")
    ax_accel.set_xlabel(r"$\tau$ [s]")
    ax_accel.set_ylabel(r"$\sigma(\tau)$ [m/s$^2$]")
    ax_accel.grid(True, which="both", ls=":", alpha=0.5)
    ax_accel.axvline(1.0, color="gray", ls="--", alpha=0.5)
    ax_accel.legend()

    fig.suptitle("SachiSLAM IMU Allan Deviation")
    fig.tight_layout()

    os.makedirs(DATA_DIR, exist_ok=True)
    out_stamp = datetime.now().strftime("%Y%m%d")
    out_path = os.path.join(DATA_DIR, f"allan_deviation_{out_stamp}.png")
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved plot: {out_path}")


if __name__ == "__main__":
    main()
