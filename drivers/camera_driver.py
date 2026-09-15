"""Orbbec Gemini 435Le driver — the ONLY module that imports pyorbbecsdk.

This is the hardware boundary. It repackages every piece of raw SDK output into
the sensor-agnostic ``core.types`` contract before anything in ``core/`` sees it:

  * Depth frames -> metric Open3D point clouds wrapped in
    :class:`~core.types.CloudFrame`.
  * IMU (accel + gyro) -> :class:`~core.types.ImuSample`.

Threading / timing model:
  The pipeline runs in CALLBACK mode: the SDK's own thread invokes a frameset
  callback (:meth:`_on_frameset`) as each frameset arrives, so IMU delivery is
  not capped by any poll cadence (``wait_for_frames`` returned one frameset per
  iteration, bounding the rate at ~poll speed). The callback does only cheap
  work — pair and emit IMU, and hand any depth+color frameset to a dedicated
  ``gemini-depth`` worker thread via a bounded (size-1, drop-when-busy) queue.
  The heavy point-cloud build runs on that worker, never in the callback. Frames
  carry device timestamps and are associated downstream (preintegrator /
  keyframe logic), not here.

  The frame aggregate mode is read from config (``camera.frame_aggregate_mode``,
  applied in :meth:`start`); under per-sample modes accel and gyro arrive in
  separate framesets and are paired in :meth:`_emit_imu`.

What this module does NOT do:
  * No estimation, ICP, preintegration, or mapping (that is core/).
  * No visualization (viz/).
  * It never leaks a pyorbbecsdk type across its API surface — callers only
    ever receive CloudFrame / ImuSample.

pyorbbecsdk is imported lazily inside :meth:`start` so that importing this
module (for wiring in run_slam.py, type checking, etc.) does not require the SDK
or a connected camera.
"""

from __future__ import annotations

import queue
import threading
import time
from typing import Callable, Optional

import numpy as np
import open3d as o3d
import yaml

from core.types import CloudFrame, ImuSample


class GeminiDriver:
    """Thin adapter over the Orbbec SDK producing CloudFrame / ImuSample."""

    def __init__(self, config_path: str) -> None:
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        depth_cfg = self.config.get("camera", {}).get("depth", {})
        # Depth values from PointCloudFilter come out in millimeters; scale to
        # meters. depth_scale of 0.001 == "depth reported in mm".
        self._depth_scale = float(depth_cfg.get("depth_scale", 0.001))
        self._min_range = float(depth_cfg.get("min_range_m", 0.0))
        self._max_range = float(depth_cfg.get("max_range_m", float("inf")))

        self._depth_callback: Optional[Callable[[CloudFrame], None]] = None
        self._imu_callback: Optional[Callable[[ImuSample], None]] = None

        # Latest accel/gyro caches for IMU pairing under ANY_SITUATION (accel and
        # gyro arrive in separate framesets). Written ONLY on the poll thread, so
        # no locking is needed. Timestamps are kept for new-vs-repeat dedup.
        self._accel_xyz: Optional[np.ndarray] = None
        self._gyro_xyz: Optional[np.ndarray] = None
        self._last_accel_ts: Optional[int] = None
        self._last_gyro_ts: Optional[int] = None

        # [IMU-COUNT] poll-thread-only counters (read in stop() after the poll
        # thread is joined, so no locking is needed). Diagnostic for pairing loss.
        self._imu_accel_seen = 0
        self._imu_gyro_seen = 0
        self._imu_emitted = 0
        self._imu_count_t0: Optional[float] = None  # streaming start (perf_counter)

        self._pipeline = None  # pyorbbecsdk Pipeline
        self._pcf = None  # pyorbbecsdk PointCloudFilter
        self._running = False

        # Depth processing runs on a DEDICATED worker thread so the heavy
        # point-cloud build never blocks the SDK frameset callback (which must
        # return fast to keep IMU flowing at ~200 Hz). A bounded queue (size 1)
        # hands framesets over; when the worker is busy the callback DROPS the
        # frameset rather than blocking or growing the queue — dropping depth is
        # acceptable, stalling IMU is not.
        self._depth_queue = None  # queue.Queue, created in start()
        self._depth_thread: Optional[threading.Thread] = None
        self._depth_drops = 0

        # SDK enum handles, populated lazily in start() so the background loop
        # can reference them without re-importing.
        self._OBFrameType = None

        # Depth->color alignment mode actually used (set in start()).
        self._align_mode_used: Optional[str] = None
        # One-shot guard for the RGB_POINT layout verification print.
        self._layout_logged = False

    # ---------------------------------------------------------------- lifecycle
    def start(
        self,
        depth_callback: Callable[[CloudFrame], None],
        imu_callback: Callable[[ImuSample], None],
    ) -> None:
        """Open the device and begin streaming to the given callbacks.

        Args:
            depth_callback: Invoked (on the gemini-depth worker thread) with
                each CloudFrame.
            imu_callback: Invoked (on the SDK callback thread) with each
                ImuSample.
        """
        # Lazy import: this is the one and only pyorbbecsdk import site.
        from pyorbbecsdk import (
            Config,
            OBAlignMode,
            OBFormat,
            OBStreamType,
            OBFrameAggregateOutputMode,
            OBFrameType,
            Pipeline,
            PointCloudFilter,
        )

        self._depth_callback = depth_callback
        self._imu_callback = imu_callback
        self._OBFrameType = OBFrameType

        # Same config-driven lookup collect_static_imu() uses, so the main
        # pipeline's aggregate mode is changeable from camera.frame_aggregate_mode
        # (default FULL_FRAME_REQUIRE -> unchanged behaviour until set).
        mode_name = self.config.get("camera", {}).get(
            "frame_aggregate_mode", "FULL_FRAME_REQUIRE"
        )
        aggregate_mode = getattr(OBFrameAggregateOutputMode, mode_name)

        def _make_config(align_mode):
            cfg = Config()
            cfg.enable_video_stream(OBStreamType.DEPTH_STREAM)
            cfg.enable_video_stream(OBStreamType.COLOR_STREAM)
            cfg.enable_accel_stream()
            cfg.enable_gyro_stream()
            # Depth->color alignment so per-point RGB is registered to depth.
            cfg.set_align_mode(align_mode)
            cfg.set_frame_aggregate_output_mode(aggregate_mode)
            return cfg

        # Point cloud filter: colored output (OBFormat.RGB_POINT) -> (N, 6)
        # [x, y, z, r, g, b]. RGB as 0-255 (normalization off). Fed the aligned
        # frameset so color is registered to depth. Must exist BEFORE the
        # pipeline starts — the SDK callback can fire immediately and the depth
        # worker (which uses it) may run right away.
        self._pcf = PointCloudFilter()
        self._pcf.set_create_point_format(OBFormat.RGB_POINT)
        self._pcf.set_color_data_normalization(False)

        # Bring up state + the depth worker BEFORE starting the pipeline: the SDK
        # now owns the streaming thread and _on_frameset may fire the instant
        # start() is called.
        self._running = True
        # [IMU-COUNT] reset counters and mark streaming start for rate math.
        self._imu_accel_seen = 0
        self._imu_gyro_seen = 0
        self._imu_emitted = 0
        self._imu_count_t0 = time.perf_counter()
        # Dedicated depth worker + bounded hand-off queue (size 1 -> drop when
        # busy). Depth is processed here, never in the callback.
        self._depth_drops = 0
        self._depth_queue = queue.Queue(maxsize=1)
        self._depth_thread = threading.Thread(
            target=self._depth_worker, name="gemini-depth", daemon=True
        )
        self._depth_thread.start()

        # Start the pipeline in CALLBACK mode (no poll thread): the SDK's own
        # thread invokes self._on_frameset per frameset, uncapping IMU delivery
        # from poll cadence. Prefer hardware D2C alignment; fall back to software
        # if the device / profile combo rejects it at start().
        self._pipeline = Pipeline()
        try:
            self._pipeline.start(_make_config(OBAlignMode.HW_MODE), self._on_frameset)
            self._align_mode_used = "HW_MODE"
        except Exception as exc:  # noqa: BLE001 - fall back to software align
            print(
                f"[GeminiDriver] HW_MODE depth->color align failed ({exc}); "
                f"falling back to SW_MODE.",
                flush=True,
            )
            self._pipeline = Pipeline()
            self._pipeline.start(_make_config(OBAlignMode.SW_MODE), self._on_frameset)
            self._align_mode_used = "SW_MODE"
        print(
            f"[GeminiDriver] depth->color align mode: {self._align_mode_used}",
            flush=True,
        )

    def stop(self) -> None:
        """Stop streaming and release the device.

        We no longer own the streaming thread (the SDK does), so stop the
        pipeline FIRST — that halts the frameset callback — then drain and join
        the depth worker, then report counters.
        """
        self._running = False
        # 1) Stop the SDK pipeline first: halts the callback thread, so no new
        #    framesets are emitted or enqueued after this returns.
        if self._pipeline is not None:
            self._pipeline.stop()
            self._pipeline = None
        # 2) Drain + join the depth worker (it exits once _running is False; any
        #    frameset still queued at shutdown is simply dropped).
        if self._depth_thread is not None:
            self._depth_thread.join(timeout=2.0)
            self._depth_thread = None
        self._depth_queue = None
        # [IMU-COUNT] one-shot summary (callback stopped + worker joined -> the
        # counters are quiescent). Elapsed lets you compute per-second rates.
        _imu_elapsed = (
            time.perf_counter() - self._imu_count_t0
            if self._imu_count_t0 is not None
            else 0.0
        )
        print(
            f"[IMU-COUNT] accel_seen={self._imu_accel_seen} "
            f"gyro_seen={self._imu_gyro_seen} emitted={self._imu_emitted} "
            f"elapsed={_imu_elapsed:.2f}s",
            flush=True,
        )
        self._pcf = None

    # ------------------------------------------------------------ static IMU
    def collect_static_imu(self, duration_s: float):
        """Collect ~``duration_s`` of accel+gyro with the rig held still.

        Runs a short, self-contained accel+gyro stream BEFORE the main pipeline
        starts. This is the SINGLE stationary window used for BOTH tilt and bias
        init, so those estimates come from the same samples. The frame aggregate
        mode is read from config (``camera.frame_aggregate_mode``) so §7 step 2
        has one place to change it.

        This method does NO estimation math (§3) — it returns only descriptive
        statistics of the raw samples: ``(mean_accel, mean_gyro, std_accel,
        std_gyro)``, each a float64 ``(3,)`` array in the IMU/body frame. Tilt
        and bias derivation live in the orchestrator. Blocks ~``duration_s``;
        raises if no synchronized samples arrive.
        """
        from pyorbbecsdk import (
            Config,
            OBFrameType,
            OBFrameAggregateOutputMode,
            Pipeline,
        )

        mode_name = self.config.get("camera", {}).get(
            "frame_aggregate_mode", "FULL_FRAME_REQUIRE"
        )
        aggregate_mode = getattr(OBFrameAggregateOutputMode, mode_name)

        config = Config()
        config.enable_accel_stream()
        config.enable_gyro_stream()
        config.set_frame_aggregate_output_mode(aggregate_mode)

        pipeline = Pipeline()
        pipeline.start(config)

        accels = []
        gyros = []
        t0 = time.perf_counter()
        try:
            while time.perf_counter() - t0 < duration_s:
                frames = pipeline.wait_for_frames(100)
                if frames is None:
                    continue
                accel_frame = frames.get_frame(OBFrameType.ACCEL_FRAME)
                gyro_frame = frames.get_frame(OBFrameType.GYRO_FRAME)
                if accel_frame is None or gyro_frame is None:
                    continue
                a = accel_frame.as_accel_frame()
                g = gyro_frame.as_gyro_frame()
                accels.append([a.get_x(), a.get_y(), a.get_z()])
                gyros.append([g.get_x(), g.get_y(), g.get_z()])
        finally:
            pipeline.stop()

        if not accels:
            raise RuntimeError("No IMU samples collected during static init.")

        accel = np.asarray(accels, dtype=np.float64)
        gyro = np.asarray(gyros, dtype=np.float64)
        return (
            accel.mean(axis=0),
            gyro.mean(axis=0),
            accel.std(axis=0),
            gyro.std(axis=0),
        )

    def collect_static_gravity(self, duration_s: float = 1.0) -> np.ndarray:
        """Mean specific-force vector (m/s^2, body frame) over a static window.

        Thin wrapper over :meth:`collect_static_imu` kept for callers that only
        need the accel mean for tilt (e.g. diagnostics/tilt_stability.py).
        Returns shape ``(3,)``. Raises if no samples arrive.
        """
        mean_accel, _mean_gyro, _std_accel, _std_gyro = self.collect_static_imu(
            duration_s
        )
        return mean_accel

    # ------------------------------------------------------------ SDK callback
    def _on_frameset(self, frames) -> None:
        """SDK-thread frameset callback: emit IMU, enqueue depth. Returns fast.

        Invoked by the pipeline's internal thread once per frameset (callback
        mode), so IMU delivery is no longer capped by a poll loop. Does only
        cheap work — pair/emit IMU and hand any depth+color frameset to the
        gemini-depth worker. The WHOLE body is wrapped in try/except: an
        exception must never propagate into the SDK-owned thread.
        """
        try:
            if frames is None or not self._running:
                return

            # IMU: emitted on every frameset (cheap).
            self._emit_imu(frames)

            # Depth: only when BOTH depth and color are present in this frameset.
            # PointCloudFilter (RGB_POINT) needs both; under ANY_SITUATION they
            # can arrive in separate framesets, and calling it without color
            # raises "depth frame or color frame not found in frameset!".
            depth = frames.get_frame(self._OBFrameType.DEPTH_FRAME)
            color = frames.get_frame(self._OBFrameType.COLOR_FRAME)
            if depth is not None and color is not None:
                # Hand depth work to the dedicated worker and return immediately.
                # Bounded queue (size 1): if the worker is still busy, DROP this
                # frameset rather than blocking the SDK thread.
                try:
                    self._depth_queue.put_nowait(frames)
                except queue.Full:
                    self._depth_drops += 1
                    if self._depth_drops % 30 == 0:  # counted, not per-frame
                        print(
                            f"[DEPTH-DROP] dropped {self._depth_drops} depth "
                            f"framesets total (worker busy)",
                            flush=True,
                        )
        except Exception as exc:  # noqa: BLE001 - never escape into the SDK thread
            print(f"[GeminiDriver] frameset callback error: {exc}", flush=True)

    # --------------------------------------------------------------- depth worker
    def _depth_worker(self) -> None:
        """Process depth framesets off the poll loop.

        Holding a pyorbbecsdk frameset across the poll iteration and processing
        it here is safe: frames are reference-counted (holding the Python object
        keeps the underlying buffer alive). This mirrors the SDK's own async
        pattern — examples/advanced/15_high_performance_pipeline.py and
        14_two_devices_sync.py queue frames to a worker thread with a bounded,
        drop-when-full queue.
        """
        while self._running:
            try:
                frames = self._depth_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            self._emit_depth(frames)

    # ------------------------------------------------------------------- emit
    def _emit_imu(self, frames) -> None:
        """Pair the latest accel + gyro into an ImuSample.

        Under ANY_SITUATION a frameset often carries only accel, only gyro, or
        neither, so requiring both in one frameset drops most samples. Instead we
        cache the most recent of each type and emit whenever a NEW frame of
        either type arrives and the other is already available, using the
        newly-arrived frame's timestamp. Repeated (stale) frames are ignored via
        timestamp dedup, so no duplicate ImuSamples are emitted.

        The caches are written ONLY on the poll thread, so no locking is needed.
        """
        OBFrameType = self._OBFrameType
        accel_frame = frames.get_frame(OBFrameType.ACCEL_FRAME)
        gyro_frame = frames.get_frame(OBFrameType.GYRO_FRAME)

        # SDK timestamps are in microseconds; core speaks nanoseconds.
        emit_ts = None  # ts of the newly-arrived frame that triggers emission

        if accel_frame is not None:
            self._imu_accel_seen += 1  # [IMU-COUNT]
            accel = accel_frame.as_accel_frame()
            ts = accel.get_timestamp_us() * 1000
            if ts != self._last_accel_ts:  # new sample, not a repeat
                self._last_accel_ts = ts
                self._accel_xyz = np.array(
                    [accel.get_x(), accel.get_y(), accel.get_z()], dtype=np.float64
                )
                emit_ts = ts

        if gyro_frame is not None:
            self._imu_gyro_seen += 1  # [IMU-COUNT]
            gyro = gyro_frame.as_gyro_frame()
            ts = gyro.get_timestamp_us() * 1000
            if ts != self._last_gyro_ts:  # new sample, not a repeat
                self._last_gyro_ts = ts
                self._gyro_xyz = np.array(
                    [gyro.get_x(), gyro.get_y(), gyro.get_z()], dtype=np.float64
                )
                emit_ts = ts

        # Emit only on a new arrival, and only once both types are available.
        if emit_ts is None or self._accel_xyz is None or self._gyro_xyz is None:
            return

        self._imu_emitted += 1  # [IMU-COUNT]
        sample = ImuSample(
            timestamp_ns=emit_ts,
            gyro=self._gyro_xyz,
            accel=self._accel_xyz,
        )
        try:
            self._imu_callback(sample)
        except Exception as exc:  # noqa: BLE001
            print(f"[GeminiDriver] imu_callback error: {exc}", flush=True)

    def _emit_depth(self, frames) -> None:
        """Build and dispatch a CloudFrame from an aligned depth+color frameset."""
        depth_frame = frames.get_frame(self._OBFrameType.DEPTH_FRAME).as_depth_frame()
        timestamp_ns = depth_frame.get_timestamp_us() * 1000
        cloud = self._depth_to_cloud(frames)
        if cloud is None:
            return
        try:
            self._depth_callback(CloudFrame(cloud=cloud, timestamp_ns=timestamp_ns))
        except Exception as exc:  # noqa: BLE001
            print(f"[GeminiDriver] depth_callback error: {exc}", flush=True)

    # -------------------------------------------------------------- conversion
    def _depth_to_cloud(self, frames) -> Optional[o3d.geometry.PointCloud]:
        """Deproject an aligned depth+color frameset into a colored O3D cloud.

        Uses PointCloudFilter (RGB_POINT format) to produce an (N, 6) array
        [x, y, z, r, g, b] — XYZ in millimeters, RGB in 0-255. XYZ is scaled to
        meters, range-gated, and wrapped in an Open3D cloud EXACTLY as before;
        RGB rides along as an additive sidecar (``cloud.colors``, 0-1 float) and
        is carried through the SAME row masks so point[i] and color[i] stay
        married. ICP reads only ``.points``/``.normals`` and ignores colors.
        """
        point_cloud_frame = self._pcf.process(frames)
        if point_cloud_frame is None:
            return None

        raw = np.asarray(self._pcf.calculate(point_cloud_frame), dtype=np.float32)
        if raw.size == 0:
            return None
        raw = raw.reshape(-1, 6)

        # ---- ONE-TIME layout verification (keep until confirmed on-device) ----
        # Confirm cols 0-2 are metric XYZ in mm (~-3500..3600) and cols 3-5 are
        # RGB in 0-255, BEFORE this data is trusted. Fires only on the 1st frame.
        if not self._layout_logged:
            self._layout_logged = True
            print(
                f"[PCF-LAYOUT] shape={raw.shape} "
                f"col_min={np.nanmin(raw, axis=0)} col_max={np.nanmax(raw, axis=0)}",
                flush=True,
            )

        xyz = raw[:, :3]
        rgb = raw[:, 3:6]

        # Drop non-finite points (inf/nan from invalid depth) and near-zero-norm
        # points (invalid-depth markers parked at the origin). Apply the SAME row
        # mask to RGB so rows stay aligned. Promote surviving XYZ to float64.
        finite_mask = np.isfinite(xyz).all(axis=1)
        nonzero_mask = np.linalg.norm(xyz, axis=1) > 1e-6
        keep = finite_mask & nonzero_mask
        xyz = xyz[keep].astype(np.float64)
        rgb = rgb[keep]

        # Too few valid points to be worth a keyframe attempt; skip this frame.
        if xyz.shape[0] < 100:
            return None

        # SDK emits millimeters; convert to meters.
        xyz = xyz * self._depth_scale

        # Range gate on Euclidean distance from the camera origin (same row mask
        # applied to RGB).
        dist = np.linalg.norm(xyz, axis=1)
        rng = (dist >= self._min_range) & (dist <= self._max_range)
        xyz = xyz[rng]
        rgb = rgb[rng]
        if xyz.shape[0] == 0:
            return None

        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(xyz)
        # RGB sidecar: 0-255 -> 0-1 float, clipped. Additive only; never fed to ICP.
        cloud.colors = o3d.utility.Vector3dVector(
            np.clip(rgb.astype(np.float64) / 255.0, 0.0, 1.0)
        )
        return cloud
