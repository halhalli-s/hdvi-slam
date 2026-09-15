"""Sensor-agnostic data contract for the SachiSLAM pipeline.

This module defines the *only* types that cross the boundary between the
hardware layer (``drivers/``) and the estimation layer (``core/``). A driver
converts whatever its vendor SDK produces into these dataclasses; every core
component consumes and produces these dataclasses.

What this module does NOT do:
  * It does not import any sensor SDK (pyorbbecsdk) — that lives in drivers/.
  * It does not import gtsam or ROS.
  * It performs no processing, filtering, or coordinate transforms. It is a
    pure data carrier. Keeping it dependency-light is what lets core/ stay
    testable without hardware.

Depending on open3d and numpy is deliberate: point clouds are the currency of
the front-end, and Open3D is a processing library, not a sensor SDK.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import open3d as o3d


@dataclass
class CloudFrame:
    """A single depth frame, already converted to a metric point cloud.

    Attributes:
        cloud: Open3D point cloud in the camera/body frame, units in meters.
        timestamp_ns: Capture time in nanoseconds (monotonic device clock).
            Used to associate the frame with preintegrated IMU measurements.
    """

    cloud: o3d.geometry.PointCloud
    timestamp_ns: int


@dataclass
class ImuSample:
    """A single synchronized gyro + accelerometer reading.

    The values are raw sensor-frame measurements (specific force and angular
    rate). Extrinsics (``body_P_sensor``) and bias correction are applied
    downstream by the preintegrator — not here.

    Attributes:
        timestamp_ns: Sample time in nanoseconds (monotonic device clock).
        gyro: Angular velocity, shape ``(3,)``, rad/s, sensor frame.
        accel: Specific force, shape ``(3,)``, m/s^2, sensor frame.
    """

    timestamp_ns: int
    gyro: np.ndarray  # shape (3,)
    accel: np.ndarray  # shape (3,)

    def __post_init__(self) -> None:
        # Normalize to float64 (3,) arrays so downstream math is predictable
        # regardless of how the driver built them.
        self.gyro = np.asarray(self.gyro, dtype=np.float64).reshape(3)
        self.accel = np.asarray(self.accel, dtype=np.float64).reshape(3)
