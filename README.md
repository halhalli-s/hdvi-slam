# HDVI-SLAM

**Handheld Dense Visual-Inertial SLAM** — a real-time dense RGBD-inertial SLAM system for handheld 3D room scanning. You walk around a room holding an Orbbec Gemini 435Le RGBD camera (with onboard IMU) and the system builds a live camera trajectory and a dense, colored 3D point-cloud map, both continuously corrected by a factor-graph backend as loop closures land.

Built solo, from scratch, in Python on ROS 2 Jazzy.

![HDVI-SLAM demo](assets/demo.gif)

> More demos: [live demos and walkthroughs](YOUR_WEBSITE_URL)

---

## What it does

- **Live trajectory** of the camera, published as a ROS `nav_msgs/Path`.
- **Dense colored point-cloud map**, published as `sensor_msgs/PointCloud2`.
- **Loop closure** — the map and trajectory retro-correct when the factor graph is revised on revisit, keeping the reconstruction metrically consistent.

Everything reflects the current optimized estimate from the backend, not a raw dead-reckoned path.

---

## Results

Closed-loop circle walks, room-scale, returning to a marked start:

| Configuration        | Position error | Heading error |
|----------------------|----------------|---------------|
| Loop closure OFF     | ~9.5 cm        | ~7.6 deg      |
| Loop closure ON      | ~0.8 - 3.3 cm  | ~0.4 - 5.5 deg|

Loop closure delivers a consistent ~3-5x reduction in position error. On rectangle and pivot runs the position error lands around 1-2 cm.

Per-keyframe cost is dominated by ICP registration (~85-95% of the cycle); the iSAM2 backend update is ~1 ms and never the bottleneck.

---

## How it works

The system is split into three strictly-separated layers so the estimation core can be tested with no hardware and no ROS:

```
drivers/   ->  Orbbec SDK: RGBD frames + IMU. The only place the camera SDK is imported.
core/      ->  Sensor- and middleware-agnostic estimation. numpy / Open3D / GTSAM only.
viz/       ->  ROS 2 publishers (Path + colored map). The only place ROS is imported.
```

Everything crossing from hardware into the core passes through two dataclasses (`CloudFrame`, `ImuSample`), so the core never depends on the camera or ROS.

**Pipeline:**

1. **Driver** streams IMU via SDK push callbacks at native rate (202 Hz) and depth frames on a dedicated worker thread, decoupled so heavy depth-to-cloud conversion never stalls IMU delivery.
2. **IMU preintegration** (GTSAM) accumulates samples between keyframes; turn-on bias is measured at startup and seeded with an anisotropic prior.
3. **Keyframe trigger** fires on gravity-cancelled translation, with a hysteresis-banded stationary detector.
4. **ICP frontend** (Open3D point-to-plane) registers consecutive keyframe clouds, using a rotation-only IMU initial guess and an adaptive point cap for bounded cost.
5. **Backend** (GTSAM iSAM2) fuses ICP between-factors, IMU factors, and constant-bias random-walk edges into a factor graph, solved incrementally.
6. **Loop closure** verifies revisits with seeded coarse-to-fine ICP and adds them as constraints; the graph re-solves and the map rebuilds from the updated poses.
7. **Visualization** publishes the trajectory and a colored map rebuilt from current optimized poses on its own thread.

---

## Tech stack

- **Language:** Python 3.12
- **Estimation:** GTSAM (iSAM2, IMU preintegration), Open3D (point-to-plane ICP), NumPy
- **Middleware:** ROS 2 Jazzy (visualization only)
- **Hardware:** Orbbec Gemini 435Le RGBD camera with onboard IMU (GigE)

---

## Repository layout

```
config/          Hardware and tuning parameters (single source of truth)
core/            Estimation: types, ICP frontend, IMU preintegration,
                 keyframe trigger, iSAM2 backend, loop closure, map builder
drivers/         Orbbec camera driver (SDK callbacks, RGBD + IMU)
viz/             ROS 2 publishers (trajectory + colored map)
scripts/         Orchestrator (run_slam.py)
calibration/     IMU recording + Allan-variance noise characterization
diagnostics/     Standalone sensor/rate/bias/rotation diagnostic tools
tests/           Unit tests for core estimation (no hardware required)
```

---

## Running it

> Requires the Orbbec Gemini 435Le, ROS 2 Jazzy, and a Python venv with the
> project dependencies. Estimation-core unit tests run without any hardware.

**Network (fresh session, GigE camera):**
```bash
sudo ip addr add 192.168.1.10/24 dev enp45s0
```

**Terminal 1 - static transform:**
```bash
source /opt/ros/jazzy/setup.bash
ros2 run tf2_ros static_transform_publisher 0 0 0 0 0 0 world map
```

**Terminal 2 - RViz2:**
```bash
source /opt/ros/jazzy/setup.bash
rviz2
# Fixed Frame = map; add /slam/trajectory (Path) and /slam/map (PointCloud2, Color Transformer = RGB8)
```

**Terminal 3 - pipeline:**
```bash
cd hdvi-slam && source venv/bin/activate
PYTHONPATH=".:/opt/ros/jazzy/lib/python3.12/site-packages" \
  python3 -u scripts/run_slam.py > run.log 2>&1
```

**Tests (no hardware):**
```bash
PYTHONPATH="." pytest
```

---

## Engineering highlights

A few of the harder problems solved along the way:

- **IMU delivery, 9.4 Hz -> 202 Hz.** The IMU was being delivered bundled inside depth-paced framesets via a polling interface, capping it near the depth rate. Fixed by switching to the SDK's callback interface (IMU as its own native-rate stream) and moving depth-to-cloud conversion onto a dedicated worker thread so it never blocks IMU delivery.
- **Loop-closure verification.** Seeding verification ICP from live optimized poses (not identity) and running coarse-to-fine (wide radius for reach, narrow radius for an honest score) raised accepted-closure fitness from ~0.5 to ~0.85-0.96.
- **Adaptive ICP point cap.** Re-voxelizing wide-open views instead of random thinning cut worst-case registration from ~2900 ms to ~400 ms.
- **Turn-on bias seeding.** Measuring MEMS turn-on bias at startup (instead of assuming zero) removed ~15 cm of position drift per 2 s.
- **Weak-edge reject handling.** Failed ICP registrations are added as inflated-sigma constraints rather than dropped, eliminating chain holes where loop-closure corrections used to pool and tear the map.

---

## Status & known limits

Working end to end, including loop closure. Active areas: heading-error repeatability across runs, sub-frame IMU/depth timestamp interpolation, and eigenvalue-based ICP degeneracy detection. A residual map seam at closure keyframes is structural to the rigid keyframe-cloud map representation (documented, not a bug).

---

## License

Released under the [MIT License](LICENSE).
