#!/usr/bin/env python3
"""Throwaway probe: enumerate the accel/gyro stream profiles the device advertises.

Answers: what IMU sample rates + full-scale ranges the Gemini 435Le exposes, and
which profile the SDK picks by default (i.e. what enable_accel_stream() /
enable_gyro_stream() select when called with no arguments). Reads the device
only; changes nothing. Tagged [IMU-PROF].
"""

from __future__ import annotations

import pyorbbecsdk as ob


def _dump(profile_list, as_specific: str, label: str) -> None:
    n = profile_list.get_count()
    print(f"[IMU-PROF] {label}: {n} advertised profiles")
    for i in range(n):
        p = getattr(profile_list.get_stream_profile_by_index(i), as_specific)()
        print(
            f"[IMU-PROF]   idx={i} rate={p.get_sample_rate()} "
            f"range={p.get_full_scale_range()} format={p.get_format()}"
        )


def main() -> None:
    pipe = ob.Pipeline()  # raises "No device found" if the camera isn't connected

    accel_list = pipe.get_stream_profile_list(ob.OBSensorType.ACCEL_SENSOR)
    gyro_list = pipe.get_stream_profile_list(ob.OBSensorType.GYRO_SENSOR)

    _dump(accel_list, "as_accel_stream_profile", "ACCEL")
    _dump(gyro_list, "as_gyro_stream_profile", "GYRO")

    # Default = what UNKNOWN/UNKNOWN resolves to (== enable_*_stream() no-args).
    accel_def = accel_list.get_accel_stream_profile(
        ob.OBAccelFullScaleRange.ACCEL_FS_UNKNOWN,
        ob.OBGyroSampleRate.SAMPLE_RATE_UNKNOWN,
    )
    gyro_def = gyro_list.get_gyro_stream_profile(
        ob.OBGyroFullScaleRange.FS_UNKNOWN,
        ob.OBGyroSampleRate.SAMPLE_RATE_UNKNOWN,
    )
    print(
        f"[IMU-PROF] DEFAULT accel rate={accel_def.get_sample_rate()} "
        f"range={accel_def.get_full_scale_range()}"
    )
    print(
        f"[IMU-PROF] DEFAULT gyro rate={gyro_def.get_sample_rate()} "
        f"range={gyro_def.get_full_scale_range()}"
    )


if __name__ == "__main__":
    main()
