#!/usr/bin/env python3
"""Record a static IMU log for Allan variance characterization.

Streams the Orbbec Gemini 435Le's onboard accelerometer and gyroscope and dumps
every synchronized sample to CSV. Run this with the camera sitting perfectly
still (on a solid, vibration-free surface) for as long as possible — a few hours
is ideal for a good Allan variance estimate; the more data, the further right
(larger tau) the curve extends. Then feed the CSV to ``allan_variance.py``.

This is a standalone characterization utility, not part of the SLAM pipeline.
Like ``drivers/``, it is an edge tool and therefore may import pyorbbecsdk;
nothing in ``core/`` depends on it.

Usage:
    python calibration/record_imu.py            # record for 2 hours (default)
    python calibration/record_imu.py --duration 3600   # or a fixed # of seconds
    # Ctrl+C stops early at any point.

Output:
    calibration/data/static_YYYYMMDD_HHMMSS.csv
    columns: timestamp_ns,gx,gy,gz,ax,ay,az
"""

from __future__ import annotations

import argparse
import csv
import os
import signal
import time
from datetime import datetime

from pyorbbecsdk import (
    Config,
    OBFrameAggregateOutputMode,
    OBFrameType,
    Pipeline,
)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


class _Stopper:
    """Flips to True on the first SIGINT so the main loop can exit cleanly."""

    def __init__(self) -> None:
        self.stop = False
        signal.signal(signal.SIGINT, self._handle)

    def _handle(self, signum, frame) -> None:  # noqa: ANN001 - signal handler
        print("\nCtrl+C received — finishing up...", flush=True)
        self.stop = True


def record(duration_s: float | None) -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(DATA_DIR, f"static_{stamp}.csv")

    config = Config()
    config.enable_accel_stream()
    config.enable_gyro_stream()
    # FULL_FRAME_REQUIRE: only emit an aggregated frameset once BOTH an accel
    # and a gyro frame are available, so each wait_for_frames() gives us a
    # matched pair to write on one CSV row.
    config.set_frame_aggregate_output_mode(
        OBFrameAggregateOutputMode.FULL_FRAME_REQUIRE
    )

    pipeline = Pipeline()
    pipeline.start(config)

    stopper = _Stopper()
    count = 0
    start_perf = time.perf_counter()
    last_report = start_perf
    first_ts_ns: int | None = None
    last_ts_ns: int | None = None

    print(f"Recording IMU to {csv_path}")
    print("Keep the camera perfectly still. Press Ctrl+C to stop.\n", flush=True)

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp_ns", "gx", "gy", "gz", "ax", "ay", "az"])

        while not stopper.stop:
            frames = pipeline.wait_for_frames(100)
            if frames is None:
                continue

            accel_frame = frames.get_frame(OBFrameType.ACCEL_FRAME)
            gyro_frame = frames.get_frame(OBFrameType.GYRO_FRAME)
            if accel_frame is None or gyro_frame is None:
                continue

            accel = accel_frame.as_accel_frame()
            gyro = gyro_frame.as_gyro_frame()

            # SDK timestamps are in microseconds; the pipeline speaks ns.
            timestamp_ns = accel.get_timestamp_us() * 1000

            writer.writerow(
                [
                    timestamp_ns,
                    gyro.get_x(),
                    gyro.get_y(),
                    gyro.get_z(),
                    accel.get_x(),
                    accel.get_y(),
                    accel.get_z(),
                ]
            )

            count += 1
            if first_ts_ns is None:
                first_ts_ns = timestamp_ns
            last_ts_ns = timestamp_ns

            now = time.perf_counter()
            if now - last_report >= 10.0:
                elapsed = now - start_perf
                print(
                    f"  {count} samples | {elapsed:6.1f} s elapsed "
                    f"| ~{count / elapsed:6.1f} Hz",
                    flush=True,
                )
                last_report = now

            if duration_s is not None and (now - start_perf) >= duration_s:
                break

    # --- Summary ------------------------------------------------------------
    elapsed = time.perf_counter() - start_perf
    pipeline.stop()

    print("\n" + "=" * 60)
    print(f"Saved: {csv_path}")
    print(f"Total samples: {count}")
    print(f"Elapsed (wall): {elapsed:.1f} s")
    if count > 1 and first_ts_ns is not None and last_ts_ns is not None:
        span_s = (last_ts_ns - first_ts_ns) * 1e-9
        if span_s > 0:
            rate = (count - 1) / span_s
            print(f"Actual sample rate (from timestamps): {rate:.2f} Hz")
    print("=" * 60, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Record a static IMU log.")
    parser.add_argument(
        "--duration",
        type=float,
        default=7200.0,
        help="Recording duration in seconds; stops gracefully when reached "
        "(default: 7200 = 2 hours). Ctrl+C exits early at any time.",
    )
    args = parser.parse_args()
    record(args.duration)


if __name__ == "__main__":
    main()
