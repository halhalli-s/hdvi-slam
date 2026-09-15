#!/usr/bin/env python3
"""SachiSLAM main orchestrator — wires the hardware and core together.

This is the only place that knows about *both* the sensor driver and the full
core pipeline. It owns the threading model and the flow of data:

    IMU thread   : ImuSample  -> preintegrator.integrate()
    Depth thread : CloudFrame -> keyframe trigger -> [backpressure gate]
                                -> ICP -> backend -> loop closure -> map -> viz

Everything downstream of the backpressure gate runs on a single worker so the
back-end is never entered re-entrantly.

What this module does NOT do:
  * No estimation math of its own — it only sequences calls into core/.
  * It is the integration entry point and therefore requires the physical
    camera (GeminiDriver.start raises NotImplementedError without it). Swap in
    a playback/mock driver emitting CloudFrame/ImuSample to run offline.

--------------------------------------------------------------------------------
Backpressure: a single busy/free flag, NOT a queue.
--------------------------------------------------------------------------------
The back-end (ICP + iSAM2) is slower and burstier than the ~15 fps depth
stream. If we buffered every keyframe trigger in a QUEUE, the queue would grow
without bound whenever the back-end fell behind, and we would spend our time
optimizing stale frames from seconds ago — latency that never recovers, and a
map that lags reality.

Instead we keep ONE flag (`_backend_busy`). When the motion threshold is
crossed:
  * If the back-end is free  -> claim it, snapshot the *current* latest cloud
    and the *current* preintegration, and process.
  * If the back-end is busy  -> DISCARD the trigger. Critically, we do NOT
    reset the preintegrator, so motion keeps accumulating. By the time the
    back-end frees up, the accumulated delta is larger, the trigger fires
    again, and we naturally take a keyframe that spans the bigger gap. This
    self-adapts keyframe spacing to available compute and always operates on
    the freshest data instead of a backlog.

--------------------------------------------------------------------------------
Stationary detector: gate the trigger + periodic PIM reset.
--------------------------------------------------------------------------------
When the camera is physically stationary, accel bias slowly drifts. The PIM
integrates that drift as (fake) motion; on the order of seconds, cancelled|dP|
crosses the trigger threshold from bias drift alone, and a keyframe fires with
no real motion behind it. Those spurious keyframes get placed at slightly
different positions each time (ICP has no way to disambiguate motion below
sensor noise on identical clouds), producing a scattered pose cluster instead
of a single stationary point.

Fix: watch accel/gyro magnitude standard deviation over a short rolling
window; when both are below noise-floor thresholds, we're stationary. In that
state:
  * refuse to fire keyframes (the gate below in on_depth), and
  * periodically reset the PIM in place with the current bias, so the window
    stays short and doesn't accumulate garbage. When real motion resumes, the
    detector flips off within a few ms and the first real KF has a clean,
    short PIM window.
"""

from __future__ import annotations

import collections
import os
import threading
import time

import numpy as np
import yaml
import gtsam

# Core is sensor- and middleware-agnostic; drivers/ and viz/ are the edges.
from core.icp_frontend import align
from core.imu_preintegrator import ImuPreintegrator
from core.keyframe_trigger import should_trigger
from core.loop_closure import LoopClosureDetector
from core.map_builder import MapBuilder
from core.slam_backend import SlamBackend
from core.types import CloudFrame, ImuSample
from drivers.camera_driver import GeminiDriver
from viz.rviz_publisher import RvizPublisher

DEFAULT_CONFIG = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "config",
    "gemini_435le.yaml",
)


def _gravity_to_initial_pose(accel_body: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Roll/pitch-only initial pose (4x4) from an averaged static accel vector.

    Convention check (verified against gtsam.PreintegrationParams.MakeSharedU,
    which sets n_gravity = (0, 0, -g), i.e. world Z points up):

        For a static body the predicted nav acceleration must vanish:
            0 = R_nb * a_measured + n_gravity
        => R_nb * a_measured = -n_gravity = (0, 0, +g)
        => R_nb * (a_measured / |a_measured|) = +Z

    So the initial rotation R0 = R_nb is the one that rotates the measured
    specific-force direction (which points 'up' for a static IMU) onto world +Z.
    We build that as the shortest-arc rotation; its axis lies in the world
    horizontal plane, so it introduces no rotation about vertical => yaw = 0,
    exactly as required (yaw is unobservable from gravity alone).

    Returns the 4x4 homogeneous pose (rotation only, zero translation) plus the
    tilt roll/pitch in degrees, measured relative to the IMU's Y-down 'level'
    nominal, purely for a human sanity check.
    """
    a = np.asarray(accel_body, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(a))
    if norm < 1e-6:
        raise ValueError("Degenerate gravity vector (near-zero norm).")

    up = a / norm  # body-frame 'up' == specific-force direction for a static IMU
    z = np.array([0.0, 0.0, 1.0])
    v = np.cross(up, z)
    s = float(np.linalg.norm(v))
    c = float(np.dot(up, z))

    if s < 1e-8:
        # Already (anti-)aligned with +Z; no unique shortest-arc axis.
        rot = gtsam.Rot3() if c > 0 else gtsam.Rot3.Rodrigues(np.pi, 0.0, 0.0)
    else:
        rotvec = (v / s) * float(np.arctan2(s, c))
        rot = gtsam.Rot3.Rodrigues(rotvec[0], rotvec[1], rotvec[2])

    T = np.eye(4)
    T[:3, :3] = rot.matrix()

    # Human-readable tilt from the Y-down level nominal (gravity along +Y body).
    g_hat = -up  # gravity direction in body (down)
    roll_deg = float(np.degrees(np.arctan2(g_hat[0], g_hat[1])))
    pitch_deg = float(np.degrees(np.arctan2(g_hat[2], g_hat[1])))
    return T, roll_deg, pitch_deg


class SlamOrchestrator:
    """Owns the pipeline, the threading model, and the backpressure gate."""

    # --- Stationary detector tunables ----------------------------------------
    # Rolling window length in samples (200 Hz IMU => 20 samples = 100 ms).
    STATIONARY_WINDOW = 20
    # Enter/exit std thresholds + min dwell are config-driven (imu.stationary),
    # read in __init__. They replaced single class-constant thresholds that
    # sensor noise straddled, flapping the state.
    # How often to reset the PIM while stationary, so bias drift doesn't
    # accumulate over the whole stationary period.
    STATIONARY_PIM_RESET_PERIOD = 0.5    # seconds

    def __init__(self, config_path: str = DEFAULT_CONFIG) -> None:
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        # --- Core components -------------------------------------------------
        self.preintegrator = ImuPreintegrator(self.config)
        self.backend = SlamBackend(self.config)
        self.loop_closure = LoopClosureDetector()
        self.map_builder = MapBuilder(float(self.config["octomap"]["resolution"]))
        self.viz = RvizPublisher(
            map_voxel_size=float(self.config.get("viz", {}).get("map_voxel_size", 0.05))
        )

        # --- Edges -----------------------------------------------------------
        self.driver = GeminiDriver(config_path)

        # --- Shared state / synchronization ---------------------------------
        # Protects _latest_cloud, _prev_keyframe_cloud, _backend_busy.
        self._state_lock = threading.Lock()
        # Serializes IMU integration vs. factor construction+reset. Also
        # covers the stationary-detector state and periodic PIM reset.
        self._imu_lock = threading.Lock()

        self._latest_cloud: CloudFrame | None = None
        self._prev_keyframe_cloud: CloudFrame | None = None
        self._backend_busy = False

        # Initial (KF 0) pose. Defaults to identity; overwritten by the static
        # gravity-averaging step in run() so KF 0 is seeded with the true tilt.
        self._initial_pose: np.ndarray = np.eye(4)

        # Logging: track latest IMU timestamp + a depth-frame counter so we can
        # report imu/depth timestamp drift for the first ~50 frames only.
        self._last_imu_ts_ns: int | None = None
        self._depth_frame_count = 0

        # --- Stationary detector state --------------------------------------
        self._accel_history: collections.deque = collections.deque(
            maxlen=self.STATIONARY_WINDOW
        )
        self._gyro_history: collections.deque = collections.deque(
            maxlen=self.STATIONARY_WINDOW
        )
        self._is_stationary: bool = False
        self._last_pim_reset_time: float = time.perf_counter()
        # Master switch: when false, the stationary detector is skipped entirely
        # (no trigger gating, no periodic PIM reset) and the trigger runs purely
        # on the gravity-cancelled dP threshold.
        self._use_stationary_detector = bool(
            self.config["keyframe"].get("use_stationary_detector", True)
        )
        # Hysteresis (enter/exit) + dwell for the stationary detector, so sensor
        # noise near a single threshold can't flap the state (config-driven).
        stat_cfg = self.config["imu"].get("stationary", {})
        self._enter_gyro_std = float(stat_cfg.get("enter_gyro_std", 0.005))
        self._exit_gyro_std = float(stat_cfg.get("exit_gyro_std", 0.015))
        self._enter_accel_std = float(stat_cfg.get("enter_accel_std", 0.02))
        self._exit_accel_std = float(stat_cfg.get("exit_accel_std", 0.05))
        self._stationary_min_dwell_s = float(stat_cfg.get("min_dwell_s", 1.0))
        self._last_stationary_transition: float = 0.0

        # --- Trigger-owned window-start rotation ----------------------------
        # The keyframe trigger needs the orientation at the START of the current
        # PIM window to cancel gravity. Reading backend.current_pose on the poll
        # thread is stale while the worker runs add_keyframe (up to ~1.4s). So we
        # maintain our OWN window-start rotation here: seeded at each PIM reset,
        # carried forward by the closed gyro deltaRij (never re-read from the
        # backend during a window), and re-anchored to the optimized rotation
        # when the worker finishes a keyframe. Guarded by its own lock so the
        # worker's handoff never blocks the poll thread.
        self._trigger_rot_lock = threading.Lock()
        self._trigger_R_start: gtsam.Rot3 = gtsam.Rot3()

    # ---------------------------------------------------- trigger rotation
    def _seed_trigger_rotation(self, rotation: gtsam.Rot3) -> None:
        """Set the trigger's window-start rotation to an absolute value.

        Used for the authoritative handoff after add_keyframe() (item 4) and the
        one-time startup seed — the best rotation available at those moments.
        """
        with self._trigger_rot_lock:
            self._trigger_R_start = rotation

    def _carry_trigger_rotation(self, delta_R: gtsam.Rot3) -> None:
        """Carry the window-start rotation across a PIM reset by the closed gyro
        deltaRij — no backend read, so no staleness is reintroduced."""
        with self._trigger_rot_lock:
            self._trigger_R_start = self._trigger_R_start.compose(delta_R)

    def _tracked_trigger_rotation(self) -> gtsam.Rot3:
        """Current orientation per the trigger's own tracking: window-start
        rotation composed with the live PIM's deltaRij (poll thread)."""
        with self._trigger_rot_lock:
            R_start = self._trigger_R_start
        return R_start.compose(self.preintegrator.pim.deltaRij())

    # ------------------------------------------------------------- callbacks
    def on_imu(self, sample: ImuSample) -> None:
        """IMU thread: fold the sample into the running preintegration.

        Also updates the stationary detector and, while stationary, resets
        the PIM periodically so it doesn't accumulate bias drift into a
        multi-meter fake motion.
        """
        with self._imu_lock:
            self.preintegrator.integrate(sample)
            self._last_imu_ts_ns = sample.timestamp_ns

            # Config master switch: skip the detector entirely (no gating, no
            # periodic PIM reset). _is_stationary stays False, so on_depth's gate
            # never fires and the trigger runs purely on the dP threshold.
            if not self._use_stationary_detector:
                return

            # --- Stationary detector -----------------------------------------
            self._accel_history.append(float(np.linalg.norm(sample.accel)))
            self._gyro_history.append(float(np.linalg.norm(sample.gyro)))

            if len(self._accel_history) >= self.STATIONARY_WINDOW:
                accel_std = float(np.std(self._accel_history))
                gyro_std = float(np.std(self._gyro_history))

                # Hysteresis + dwell (replaces a single threshold that noise
                # straddled): enter stationary only when BOTH stds are below the
                # (low) enter thresholds; leave only when EITHER rises above the
                # (high) exit threshold; hold in the dead band. A minimum dwell
                # since the last transition blocks rapid flapping either way.
                now = time.perf_counter()
                can_change = (
                    now - self._last_stationary_transition
                    >= self._stationary_min_dwell_s
                )
                new_state = self._is_stationary
                if can_change:
                    if self._is_stationary:
                        # Leave on clear motion on EITHER axis (above exit).
                        if (
                            accel_std > self._exit_accel_std
                            or gyro_std > self._exit_gyro_std
                        ):
                            new_state = False
                    else:
                        # Enter only when clearly still on BOTH axes (below enter).
                        if (
                            accel_std < self._enter_accel_std
                            and gyro_std < self._enter_gyro_std
                        ):
                            new_state = True

                # Print only on an actual state transition (tag + format kept).
                if new_state != self._is_stationary:
                    self._last_stationary_transition = now
                    if new_state:
                        print(
                            f"[STATIONARY] detected (accel_std={accel_std:.4f} "
                            f"gyro_std={gyro_std:.4f}) — trigger gated, "
                            f"PIM will reset every {self.STATIONARY_PIM_RESET_PERIOD:.1f}s",
                            flush=True,
                        )
                    else:
                        print(
                            f"[STATIONARY] motion resumed (accel_std={accel_std:.4f} "
                            f"gyro_std={gyro_std:.4f})",
                            flush=True,
                        )
                    self._is_stationary = new_state

                # While stationary, periodically reset the PIM so the window
                # stays short. Under _imu_lock so we can't race with the
                # keyframe worker's make_factor_and_reset (which also holds it).
                if self._is_stationary:
                    now = time.perf_counter()
                    if now - self._last_pim_reset_time > self.STATIONARY_PIM_RESET_PERIOD:
                        # Trigger rotation source at THIS reset: carry forward by
                        # the closed deltaRij (gyro), NOT backend.current_pose —
                        # this fires every 0.5s on the poll thread and a backend
                        # read would reintroduce the very staleness we removed.
                        closed_dR = self.preintegrator.pim.deltaRij()
                        self.preintegrator.pim.resetIntegrationAndSetBias(
                            self.backend.current_bias
                        )
                        self._carry_trigger_rotation(closed_dR)
                        self._last_pim_reset_time = now

    def _log_trigger_fire(self, dt, raw_norm, cancelled_norm, rot_angle) -> None:
        """One consolidated [TRIGGER] line at fire time (with the keyframe index).

        Printed only when a keyframe actually fires (the default, quiet mode).
        Per-evaluation [TRIGGER] output is available via debug.verbose_trigger.
        """
        print(
            f"[TRIGGER] KF={self.backend.index} deltaTij={dt:.3f}s "
            f"raw|dP|={raw_norm:.3f}m cancelled|dP|={cancelled_norm:.3f}m "
            f"rot={np.degrees(rot_angle):.1f}deg",
            flush=True,
        )

    def _should_trigger_with_velocity(self) -> bool:
        """Velocity-inclusive keyframe trigger (config-gated policy).

        The core gravity-cancelled trigger drops the ``v_i*dt`` term, so steady
        motion is under-counted and keyframes fire late. Here we use the full
        ``predicted_translation(R_i, v_i)`` (same R_i/v_i sources as the ICP
        seed). If velocity looks stale/implausible (``||v_i|| > max``), fall back
        to the gravity-cancelled value. Kept in the orchestrator so
        core.keyframe_trigger stays untouched (§9). Reads the live PIM only.
        """
        kf_cfg = self.config["keyframe"]
        trans_thresh = float(kf_cfg["translation_m"])
        rot_thresh_rad = np.deg2rad(float(kf_cfg["rotation_deg"]))
        max_v = float(kf_cfg.get("max_trigger_velocity", 2.0))
        use_tracked = bool(kf_cfg.get("use_tracked_rotation", False))

        # Two R_i sources: the (stale-while-worker-busy) backend read, and the
        # trigger's own gyro-tracked window-start rotation. Compute both so the
        # divergence is always visible in [TRIGGER]; the flag picks which drives
        # the decision. Backend reads (not PIM) need no _imu_lock.
        R_backend = self.backend.current_pose.rotation()
        v_i = np.asarray(self.backend.current_velocity, dtype=np.float64)
        v_mag = float(np.linalg.norm(v_i))

        # --- PIM reads under _imu_lock: serialize against on_imu's integrate /
        # reset, which now runs on a DIFFERENT thread (poll) than this trigger
        # (depth worker). Copy the scalars out here, release immediately, then
        # evaluate/print OUTSIDE the lock. The lock is NEVER held across ICP,
        # backend.add_keyframe, loop closure, or the worker spawn. ------------
        with self._imu_lock:
            R_tracked = self._tracked_trigger_rotation()  # deltaRij read
            cancelled_backend, raw_norm, dt = (
                self.preintegrator.gravity_cancelled_translation(R_backend)
            )
            cancelled_tracked = self.preintegrator.gravity_cancelled_translation(
                R_tracked
            )[0]
            _, rot_angle = self.preintegrator.current_delta()
            R_i = R_tracked if use_tracked else R_backend
            pred_norm = (
                float(np.linalg.norm(self.preintegrator.predicted_translation(R_i, v_i)))
                if v_mag <= max_v
                else None
            )

        # --- Evaluate against the copied-out values (lock released) ----------
        cancelled_norm = cancelled_tracked if use_tracked else cancelled_backend
        if v_mag > max_v:
            # Stale-velocity guard: a bad v_i would inflate the predicted
            # distance and fire spurious keyframes — use the gravity-cancelled
            # value instead. Printed per occurrence.
            print(
                f"[TRIG-FALLBACK] ||v_i||={v_mag:.3f}m/s > {max_v}m/s — "
                f"using gravity-cancelled translation for the trigger",
                flush=True,
            )
            pred_norm = cancelled_norm
        trigger_dist = pred_norm

        verbose = self.config.get("debug", {}).get("verbose_trigger", False)
        if verbose:
            print(
                f"[TRIGGER] deltaTij={dt:.3f}s raw|dP|={raw_norm:.3f}m "
                f"cancelled_backend|dP|={cancelled_backend:.3f}m "
                f"cancelled_tracked|dP|={cancelled_tracked:.3f}m "
                f"pred|dP|={pred_norm:.3f}m rot={np.degrees(rot_angle):.1f}deg",
                flush=True,
            )
        fired = trigger_dist >= trans_thresh or rot_angle >= rot_thresh_rad
        if fired and not verbose:
            self._log_trigger_fire(dt, raw_norm, cancelled_norm, rot_angle)
        return fired

    def on_depth(self, frame: CloudFrame) -> None:
        """Depth thread: stash the frame, then maybe fire a keyframe.

        This runs at ~15 fps and must return fast, so the heavy work is
        dispatched to a worker thread behind the backpressure gate.
        """
        with self._state_lock:
            self._latest_cloud = frame

        verbose = self.config.get("debug", {}).get("verbose_trigger", False)

        # Timestamp-drift diagnostic: first ~50 frames only, and only under the
        # verbose flag (would flood at 200 Hz IMU otherwise).
        self._depth_frame_count += 1
        if verbose and self._depth_frame_count <= 50 and self._last_imu_ts_ns is not None:
            drift_ms = abs(frame.timestamp_ns - self._last_imu_ts_ns) / 1e6
            print(
                f"[frame {self._depth_frame_count:3d}] imu/depth ts drift = "
                f"{drift_ms:.3f} ms",
                flush=True,
            )

        # Cheap check — reads accumulated motion straight from the IMU. Default
        # path uses the core gravity-cancelled trigger. Optionally (config-gated)
        # use the full velocity-inclusive predicted translation, which the
        # gravity-cancelled estimate under-counts for steady motion (it drops the
        # v_i*dt term, so keyframes fire late and IMU windows stretch).
        if self.config["keyframe"].get("use_velocity_in_trigger", False):
            if not self._should_trigger_with_velocity():
                return
        else:
            # Default path: should_trigger reads the PIM internally, so run it
            # under _imu_lock to serialize against on_imu (now on the poll
            # thread). Its read+decision is trivial; the lock never spans ICP,
            # add_keyframe, loop closure, or the worker spawn below. Printing is
            # done OUTSIDE the lock (fire-only, or per-eval under verbose).
            R_i = self.backend.current_pose.rotation()  # backend read, not PIM
            with self._imu_lock:
                fired, dt, raw_norm, cancelled_norm, rot_angle = should_trigger(
                    self.preintegrator, self.config, R_i
                )
            if verbose:
                print(
                    f"[TRIGGER] deltaTij={dt:.3f}s raw|dP|={raw_norm:.3f}m "
                    f"cancelled|dP|={cancelled_norm:.3f}m "
                    f"rot={np.degrees(rot_angle):.1f}deg",
                    flush=True,
                )
            if not fired:
                return
            if not verbose:
                self._log_trigger_fire(dt, raw_norm, cancelled_norm, rot_angle)

        # Stationary gate: even if the motion threshold was crossed, if the
        # camera isn't actually moving (bias drift alone tripped it), refuse
        # to fire a keyframe. KF 0 (backend.index == 0) is always allowed so
        # the prior gets seeded.
        with self._imu_lock:
            stationary = self._is_stationary
        if stationary and self.backend.index > 0:
            return

        with self._state_lock:
            if self._backend_busy:
                # Backend still chewing on the previous keyframe. DISCARD this
                # trigger. Do NOT reset the preintegrator — motion keeps
                # accumulating and we will trigger again (over a bigger gap)
                # once the backend is free. See module docstring.
                return
            self._backend_busy = True
            cloud = self._latest_cloud  # freshest frame, not a queued stale one

        threading.Thread(
            target=self._process_keyframe, args=(cloud,), daemon=True
        ).start()

    # -------------------------------------------------------------- worker
    def _process_keyframe(self, cloud: CloudFrame) -> None:
        """Worker: run ICP + back-end + loop closure + map + viz for one KF."""
        try:
            i = self.backend.index
            t_kf_start = time.perf_counter()

            if i == 0:
                # First keyframe: seed priors only — no ICP edge, no IMU factor.
                # Use the gravity-derived initial pose (tilt) rather than
                # identity, so IMU gravity subtraction is correct from the start.
                _t = time.perf_counter()
                pose = self.backend.add_keyframe(self._initial_pose, 1.0, None, None)
                t_isam = time.perf_counter() - _t
                # Handoff (item 4): anchor the trigger's window-start rotation to
                # the optimized X(0). KF 0 does not reset the PIM.
                self._seed_trigger_rotation(pose.rotation())
                t = pose.translation()
                print(
                    f"[KF {i}] prior seeded | pose "
                    f"x={t[0]:.3f} y={t[1]:.3f} z={t[2]:.3f} "
                    f"yaw={np.degrees(pose.rotation().yaw()):.1f}deg",
                    flush=True,
                )
                self.loop_closure.add_keyframe(i, pose, cloud)
                # OctoMap dormant: not used for handheld mapping (kept importable
                # for re-enable). self.map_builder.insert_cloud(cloud, pose)
                _t = time.perf_counter()
                self._safe_publish_keyframe(pose, cloud, i)
                t_map = time.perf_counter() - _t
                print(
                    f"[TIMING] KF={i} icp=0.0ms loop=0.0ms "
                    f"isam={t_isam * 1e3:.1f}ms map={t_map * 1e3:.1f}ms "
                    f"total={(time.perf_counter() - t_kf_start) * 1e3:.1f}ms",
                    flush=True,
                )
                with self._state_lock:
                    self._prev_keyframe_cloud = cloud
                return

            # Close the preintegration window and build the IMU factor. Do this
            # under the IMU lock so no sample is integrated mid-build/reset.
            with self._imu_lock:
                imu_factor = self.preintegrator.make_factor_and_reset(
                    i - 1, i, self.backend.current_bias
                )
                # Also reset the stationary-reset timer so on_imu doesn't
                # double-reset the PIM immediately after the keyframe worker
                # just reset it as part of factor construction.
                self._last_pim_reset_time = time.perf_counter()
            # The factor carries the pre-reset measurements; reuse them to
            # predict the initial velocity for the new state.
            pim_snapshot = imu_factor.preintegratedMeasurements()

            # Trigger rotation source at THIS reset: carry forward by the closed
            # deltaRij (the window that just ended, i-1 -> i), NOT a backend read
            # — the optimized X(i) isn't computed until add_keyframe() below, so
            # gyro propagation is the best rotation available right now. It's
            # re-anchored to the optimized X(i) at the handoff after add_keyframe.
            self._carry_trigger_rotation(pim_snapshot.deltaRij())

            # [DEBUG-TEMP] Window duration this PIM covers, read right before it
            # is consumed. If resetIntegrationAndSetBias() is working, this
            # should reset to ~0 each keyframe; if it keeps growing, the PIM is
            # never being reset. Remove after diagnosis.
            print(
                f"[KF {i}] DEBUG pim.deltaTij = {pim_snapshot.deltaTij():.4f}s",
                flush=True,
            )

            # Seed ICP with the IMU preintegration instead of blind identity.
            # deltaRij/deltaPij express the motion from state i-1 (prev) to i
            # (curr) in the prev/i-1 frame, i.e. ^prev T_curr. align() expects
            # its initial_transform as source(curr)->target(prev), which is the
            # same ^prev T_curr convention, so NO inversion is needed.
            delta_R = pim_snapshot.deltaRij()  # gtsam.Rot3
            delta_p = np.asarray(pim_snapshot.deltaPij(), dtype=np.float64)
            imu_guess = np.eye(4)
            imu_guess[:3, :3] = delta_R.matrix()

            # Translation seed: rotation-only by default. Optionally seed the
            # full IMU-predicted translation (clean now that the bias fix landed),
            # gated by config. A seed farther than the correspondence distance is
            # worse than none, so reject an over-large one and fall back to zero.
            icp_cfg = self.config["icp"]
            seed_trans = np.zeros(3)
            if icp_cfg.get("use_imu_translation_seed", False):
                seed = self.preintegrator.predicted_translation(
                    self.backend.current_pose.rotation(),
                    self.backend.current_velocity,
                    meas=pim_snapshot,  # self.pim was reset by make_factor_and_reset
                )
                seed_norm = float(np.linalg.norm(seed))
                max_seed = float(icp_cfg.get("max_imu_translation_seed_m", 0.5))
                if seed_norm > max_seed:
                    # Crude safety net, NOT a correctness check: a large seed is
                    # fine if it's ACCURATE. The real criterion is seed error vs
                    # the ICP result ([SEED-ERR] below). Revisit / likely drop
                    # this magnitude cap once [SEED-ERR] data exists.
                    print(
                        f"[SEED-REJECT] KF {i}: |seed|={seed_norm:.3f}m > "
                        f"{max_seed}m (deltaTij={pim_snapshot.deltaTij():.3f}s) — "
                        f"falling back to zero translation",
                        flush=True,
                    )
                else:
                    seed_trans = seed
            imu_guess[:3, 3] = seed_trans

            guess_trans = float(np.linalg.norm(delta_p))
            guess_rot_deg = float(np.degrees(np.linalg.norm(gtsam.Rot3.Logmap(delta_R))))
            print(
                f"[KF {i}] IMU initial guess: trans={guess_trans:.3f}m "
                f"rot={guess_rot_deg:.1f}deg",
                flush=True,
            )

            # ICP odometry: current cloud (source) -> previous keyframe (target).
            with self._state_lock:
                target = self._prev_keyframe_cloud
            _t = time.perf_counter()
            transform, fitness, _rmse = align(
                cloud, target, self.config, initial_transform=imu_guess
            )
            t_icp = time.perf_counter() - _t

            # Seed error is the real seed-quality criterion (not seed magnitude):
            # how far the seeded translation was from what ICP converged to. Both
            # are in the same ^prev T_curr frame.
            icp_trans = np.asarray(transform, dtype=np.float64)[:3, 3]
            seed_err = float(np.linalg.norm(seed_trans - icp_trans))
            print(
                f"[SEED-ERR] KF {i}: err={seed_err:.3f}m "
                f"seed|t|={float(np.linalg.norm(seed_trans)):.3f}m "
                f"icp|t|={float(np.linalg.norm(icp_trans)):.3f}m "
                f"fitness={fitness:.3f}",
                flush=True,
            )

            _t = time.perf_counter()
            pose = self.backend.add_keyframe(transform, fitness, imu_factor, pim_snapshot)
            t_isam = time.perf_counter() - _t

            # Loop closure: proximity search + ICP verification.
            _t = time.perf_counter()
            matches = self.loop_closure.check(
                i, pose, cloud, self.backend.get_all_poses(),
                lambda s, t, init, corr=None, iters=None: align(
                    s, t, self.config, initial_transform=init,
                    max_corr_override=corr, max_iter_override=iters,
                ),
                self.config,
            )
            t_loop = time.perf_counter() - _t
            for matched_index, lc_transform, lc_fitness in matches:
                _t = time.perf_counter()
                self.backend.add_loop_closure_factor(
                    i, matched_index, lc_transform, lc_fitness
                )
                t_isam += time.perf_counter() - _t  # loop-closure iSAM2 update
                pose = self.backend.current_pose  # may have shifted post-closure
                print(
                    f"[KF {i}] LOOP CLOSURE -> matched KF {matched_index} "
                    f"(fitness={lc_fitness:.3f})",
                    flush=True,
                )

            # Handoff (item 4): re-anchor the trigger's window-start rotation to
            # the optimized X(i) (post loop closure) — replaces the provisional
            # gyro-carried value from the reset above. Non-blocking to the poll
            # thread (short lock).
            self._seed_trigger_rotation(pose.rotation())

            t = pose.translation()
            print(
                f"[KF {i}] icp_fitness={fitness:.3f} | pose "
                f"x={t[0]:.3f} y={t[1]:.3f} z={t[2]:.3f} "
                f"yaw={np.degrees(pose.rotation().yaw()):.1f}deg",
                flush=True,
            )

            self.loop_closure.add_keyframe(i, pose, cloud)
            # OctoMap dormant: not used for handheld mapping (kept importable
            # for re-enable). self.map_builder.insert_cloud(cloud, pose)
            _t = time.perf_counter()
            self._safe_publish_keyframe(pose, cloud, i)
            t_map = time.perf_counter() - _t

            # [TIMING] Per-stage wall-clock. map= is now just the cheap inline
            # store+Path+signal (heavy map rebuild runs on the publisher's own
            # thread). Watch loop=/isam= as the graph grows.
            print(
                f"[TIMING] KF={i} icp={t_icp * 1e3:.1f}ms loop={t_loop * 1e3:.1f}ms "
                f"isam={t_isam * 1e3:.1f}ms map={t_map * 1e3:.1f}ms "
                f"total={(time.perf_counter() - t_kf_start) * 1e3:.1f}ms",
                flush=True,
            )

            with self._state_lock:
                self._prev_keyframe_cloud = cloud
        finally:
            # Release the gate no matter what, so a failed keyframe never
            # deadlocks the pipeline.
            with self._state_lock:
                self._backend_busy = False

    # ------------------------------------------------------------- viz guard
    def _safe_publish_keyframe(self, pose, cloud, index) -> None:
        """Hand the keyframe's LOCAL cloud + current optimized poses to the publisher.

        The publisher stores each keyframe's local-frame cloud by index and
        FULLY REBUILDS the world map from the CURRENT optimized poses every
        publish, so the map tracks iSAM2 revisions (loop closures) instead of
        freezing at insertion time. No world transform happens here anymore.
        The publisher self-disables if ROS is absent, so the pipeline runs fine
        headless.
        """
        local_pts = np.asarray(cloud.cloud.points)
        # RGB is view-independent; pass per-point colors straight through in the
        # same row order as the local points (additive sidecar, viz only).
        colors = np.asarray(cloud.cloud.colors) if cloud.cloud.has_colors() else None
        self.viz.publish_keyframe(
            pose, index, local_pts, self.backend.get_all_poses(), colors=colors
        )

    # ------------------------------------------------------- static IMU init
    def _initialize_static_imu(self) -> None:
        """Measure ONE static window; derive + seed BOTH tilt and IMU bias.

        Tilt (KF 0 orientation) and turn-on bias must come from the SAME
        stationary samples, so a single ``collect_static_imu`` call feeds both.
        Turn-on bias regenerates every power-on, so it is measured at runtime,
        never hard-coded. Must run BEFORE driver.start() (self-contained stream)
        AND before the first keyframe: ordering is load-bearing — both setters
        run here, ahead of any add_keyframe.

        All estimation math lives here (§3): the driver returns only means/stds.
        Raises with the measured values if the rig was not actually stationary.
        """
        imu_cfg = self.config["imu"]
        window = float(imu_cfg.get("startup_bias_window_s", 10.0))
        g_mag = float(imu_cfg.get("gravity_magnitude", 9.81))
        mean_accel, mean_gyro, std_accel, std_gyro = self.driver.collect_static_imu(
            window
        )

        # --- Stationarity validation (raise with measured values on failure) --
        val = imu_cfg.get("startup_validation", {})
        max_accel_std = float(val.get("max_accel_std", 0.1))       # m/s^2
        max_gyro_std = float(val.get("max_gyro_std", 0.02))        # rad/s
        max_gravity_err = float(val.get("max_gravity_err", 0.5))   # m/s^2
        max_gyro_bias = float(val.get("max_gyro_bias", 0.05))      # rad/s
        accel_std_max = float(np.max(std_accel))
        gyro_std_max = float(np.max(std_gyro))
        gravity_err = abs(float(np.linalg.norm(mean_accel)) - g_mag)
        gyro_bias_mag = float(np.linalg.norm(mean_gyro))
        failures = []
        if accel_std_max >= max_accel_std:
            failures.append(f"accel_std_max={accel_std_max:.4f}>={max_accel_std}")
        if gyro_std_max >= max_gyro_std:
            failures.append(f"gyro_std_max={gyro_std_max:.4f}>={max_gyro_std}")
        if gravity_err >= max_gravity_err:
            failures.append(f"gravity_err={gravity_err:.4f}>={max_gravity_err}")
        if gyro_bias_mag >= max_gyro_bias:
            failures.append(f"gyro_bias_mag={gyro_bias_mag:.4f}>={max_gyro_bias}")
        if failures:
            raise RuntimeError(
                "Startup IMU calibration rejected — rig not stationary? "
                + "; ".join(failures)
            )

        # --- Tilt: seed KF 0 orientation (existing math) ---------------------
        T0, roll_deg, pitch_deg = _gravity_to_initial_pose(mean_accel)
        self._initial_pose = T0
        # One-time startup seed of the trigger's window-start rotation from the
        # measured tilt (best rotation available before KF 0). Re-anchored to the
        # optimized X(0) once KF 0's add_keyframe completes.
        self._seed_trigger_rotation(gtsam.Rot3(np.asarray(T0)[:3, :3]))
        print(
            f"[INIT] measured gravity tilt: roll={roll_deg:.1f}deg "
            f"pitch={pitch_deg:.1f}deg",
            flush=True,
        )

        # --- Bias: derive + seed PIM and backend -----------------------------
        # Ported from diagnostics/measure_bias.py (validated 15.5->2.2cm): gyro
        # bias is the mean gyro; accel bias is the mean accel minus a
        # gravity-magnitude vector along the measured up direction.
        up_dir = mean_accel / np.linalg.norm(mean_accel)
        accel_bias = mean_accel - g_mag * up_dir
        gyro_bias = mean_gyro
        # accel first — matches gtsam.imuBias.ConstantBias(biasAcc, biasGyro),
        # verified: ConstantBias.vector() == [acc; gyro].
        bias = gtsam.imuBias.ConstantBias(accel_bias, gyro_bias)
        self.preintegrator.set_initial_bias(bias)
        self.backend.set_initial_bias(bias)
        print(
            f"[BIAS-INIT] accel bias (m/s^2): "
            f"[{accel_bias[0]:.4f}, {accel_bias[1]:.4f}, {accel_bias[2]:.4f}]  "
            f"gyro bias (rad/s): "
            f"[{gyro_bias[0]:.6f}, {gyro_bias[1]:.6f}, {gyro_bias[2]:.6f}]",
            flush=True,
        )

    # ---------------------------------------------------------- final dump
    def _dump_final_poses(self) -> None:
        """Print every keyframe's final optimized pose at shutdown.

        Reads the backend's current optimized poses and prints one line per
        keyframe (translation + roll/pitch/yaw in degrees), in sorted index
        order, so the whole trajectory can be inspected after a Ctrl-C.

        TEMPORARY diagnostic (greppable via [FINAL-POSES], per §9 of
        project_info.md). Runs in run()'s finally block, so it must never
        raise — the empty-graph case and any backend read error are handled
        gracefully.
        """
        try:
            poses = self.backend.get_all_poses()
        except Exception as exc:  # noqa: BLE001 - never raise from a finally
            print(f"[FINAL-POSES] backend read failed ({exc})", flush=True)
            return

        if not poses:
            print("[FINAL-POSES] no keyframes in the graph", flush=True)
            return

        for idx in sorted(poses):
            pose = poses[idx]
            t = pose.translation()
            r = pose.rotation()
            print(
                f"[FINAL-POSES] KF {idx}: "
                f"x={t[0]:+.4f} y={t[1]:+.4f} z={t[2]:+.4f} "
                f"roll={np.degrees(r.roll()):+.2f}deg "
                f"pitch={np.degrees(r.pitch()):+.2f}deg "
                f"yaw={np.degrees(r.yaw()):+.2f}deg",
                flush=True,
            )

    # ---------------------------------------------------------------- run
    def run(self) -> None:
        """Start streaming and block until interrupted."""
        # Measure one static window and seed BOTH KF 0 tilt and IMU bias before
        # the real pipeline starts streaming (ordering is load-bearing).
        self._initialize_static_imu()
        self.driver.start(depth_callback=self.on_depth, imu_callback=self.on_imu)
        try:
            threading.Event().wait()  # block forever; callbacks drive the work
        except KeyboardInterrupt:
            pass
        finally:
            self.driver.stop()
            self._dump_final_poses()
            self.viz.shutdown()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run SachiSLAM.")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Path to YAML config.")
    args = parser.parse_args()

    SlamOrchestrator(args.config).run()


if __name__ == "__main__":
    main()