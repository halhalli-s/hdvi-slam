"""Test IMU preintegration accumulates translation as expected.

Feeds a sequence of constant-acceleration, zero-rotation samples and checks
that the preintegrated position delta follows the kinematic prediction
``deltaP ~= 0.5 * a * t^2`` and velocity delta ``deltaV ~= a * t``.

Note: deltaPij / deltaVij are integrals of the *measured* specific force, so
they are independent of the navigation-frame gravity setting — gravity is only
applied at ``predict`` time. That keeps this test a clean kinematics check.
"""

from __future__ import annotations

import numpy as np
import gtsam
import pytest

from core.imu_preintegrator import ImuPreintegrator
from core.types import ImuSample

CONFIG = {
    "imu": {
        "gravity_magnitude": 9.81,
        "body_P_sensor": {
            "translation": [0.0, 0.0, 0.0],  # identity extrinsic for clean math
            "rotation_quaternion": [1.0, 0.0, 0.0, 0.0],
        },
        "noise": {
            "gyro_noise_density": 0.005,
            "gyro_bias_random_walk": 0.0001,
            "accel_noise_density": 0.01,
            "accel_bias_random_walk": 0.0002,
            "integration_sigma": 0.0001,
        },
    }
}


def test_deltap_grows_quadratically() -> None:
    pre = ImuPreintegrator(CONFIG)

    accel = np.array([1.0, 0.0, 0.0])  # 1 m/s^2 along +X
    gyro = np.zeros(3)

    dt = 0.005  # 200 Hz
    n_steps = 200  # -> total_time = 1.0 s
    t_ns = 0
    step_ns = int(dt * 1e9)

    # First sample only sets the time baseline.
    pre.integrate(ImuSample(timestamp_ns=t_ns, gyro=gyro, accel=accel))
    for _ in range(n_steps):
        t_ns += step_ns
        pre.integrate(ImuSample(timestamp_ns=t_ns, gyro=gyro, accel=accel))

    total_time = n_steps * dt
    trans_norm, rot_angle = pre.current_delta()

    expected_p = 0.5 * 1.0 * total_time**2  # 0.5 m
    expected_v = 1.0 * total_time           # 1.0 m/s

    # Discrete Euler integration is slightly off from the closed form.
    assert trans_norm == pytest.approx(expected_p, rel=0.05)
    assert rot_angle == pytest.approx(0.0, abs=1e-6)

    delta_v = float(np.linalg.norm(np.asarray(pre.pim.deltaVij())))
    assert delta_v == pytest.approx(expected_v, rel=0.02)


def _integrate_constant_accel(pre, accel, dt=0.005, n_steps=200):
    """Feed a constant-accel, zero-gyro window into the preintegrator."""
    gyro = np.zeros(3)
    t_ns = 0
    step_ns = int(dt * 1e9)
    pre.integrate(ImuSample(timestamp_ns=t_ns, gyro=gyro, accel=accel))
    for _ in range(n_steps):
        t_ns += step_ns
        pre.integrate(ImuSample(timestamp_ns=t_ns, gyro=gyro, accel=accel))


def test_predicted_translation_matches_predict_and_velocity_shift() -> None:
    pre = ImuPreintegrator(CONFIG)
    _integrate_constant_accel(pre, np.array([1.0, 0.0, 0.0]))

    R_i = gtsam.Rot3()  # identity
    bias = pre.pim.biasHat()  # zero; matches the raw deltas predicted_translation reads
    dt = float(pre.pim.deltaTij())

    # (1) Static (v_i = 0) reproduces GTSAM's NavState position prediction.
    t_pred = pre.predicted_translation(R_i, np.zeros(3))
    nav = pre.pim.predict(
        gtsam.NavState(gtsam.Pose3(R_i, gtsam.Point3(0.0, 0.0, 0.0)), np.zeros(3)),
        bias,
    )
    np.testing.assert_allclose(t_pred, np.asarray(nav.position()), atol=1e-9)

    # (2) A nonzero v_i shifts the result by exactly v_i * dt.
    v_i = np.array([0.1, -0.2, 0.05])
    t_pred_v = pre.predicted_translation(R_i, v_i)
    np.testing.assert_allclose(t_pred_v - t_pred, v_i * dt, atol=1e-12)


def test_set_initial_bias_rebases_pim() -> None:
    pre = ImuPreintegrator(CONFIG)

    accel_bias = np.array([0.037, -0.012, 0.021])
    gyro_bias = np.array([0.0075, -0.0061, 0.0093])
    bias = gtsam.imuBias.ConstantBias(accel_bias, gyro_bias)

    pre.set_initial_bias(bias)

    # The underlying PIM must now integrate around the seeded bias, not zero.
    # Tangent order is [accel(3); gyro(3)].
    expected = np.concatenate([accel_bias, gyro_bias])
    np.testing.assert_allclose(pre.pim.biasHat().vector(), expected, atol=1e-12)
    np.testing.assert_allclose(pre._bias.vector(), expected, atol=1e-12)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
