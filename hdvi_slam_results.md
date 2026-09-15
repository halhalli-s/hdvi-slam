# HDVI-SLAM — Handheld Dense Visual-Inertial SLAM

A real-time dense RGBD SLAM system built from scratch in Python. A handheld
Orbbec Gemini 435Le is walked around a scene; the system produces a live
camera trajectory and a dense colored point-cloud map, both continuously
retro-corrected by a GTSAM iSAM2 factor-graph backend when a loop closes.

**Stack:** Python · GTSAM (iSAM2, IMU preintegration) · Open3D (point-to-plane
ICP) · ROS 2 Jazzy (RViz2 visualization) · Orbbec SDK
**Hardware:** Orbbec Gemini 435Le RGBD camera with onboard IMU (202 Hz)

---

## Results

Five runs on a fixed configuration, single session, room lights on. Ground
truth established by returning the camera to a marked start position and
heading; re-seating repeatability approximately ±1°.

### Loop closure on vs. off — 0.63 × 0.92 m rectangular path

| Run | Path length | Position error | Heading error | Closures accepted |
|---|---|---|---|---|
| Loop closure **off** | 2.99 m | 9.0 cm | 6.82° | 0 (gated) |
| Loop closure **on** (run 1) | 2.94 m | 1.7 cm | 0.79° | 3 @ fitness 0.929–0.930 |
| Loop closure **on** (run 2) | 2.92 m | 1.0 cm | 1.89° | 2 @ fitness 0.974–0.985 |

**Backend correction: 5–9× reduction in position error, 3.6–8.6× in heading.**

The control run is a strict comparison, not an absence of data: loop-closure
verification ran normally and *found* the same closures the enabled runs used —
six candidates scoring 0.923–0.928 — but the acceptance threshold was raised to
0.95 so none were applied. Path lengths across the three runs agree within 2%,
and walked perimeter (3.10 m) matches the trajectory-integrated length (2.92–2.99 m)
to 95%, confirming there is no translation-scale error.

### 360° in-place rotation — ~0.9 m pivot path

| Run | Path length | Position error | Heading error | Closures accepted |
|---|---|---|---|---|
| Pivot 1 | 0.96 m | 0.81 cm | 1.26° | 3 @ fitness 0.976–0.989 |
| Pivot 2 | 0.87 m | 1.03 cm | 0.75° | 3 @ fitness 0.979–0.986 |

Isolates rotational accuracy with minimal translation: a full 360° turn with
the camera returned to its marked start pose. Both runs close to roughly a
centimetre and within about a degree.

### Measurement method

Position error is the Euclidean distance between the first and last optimized
keyframe pose. Heading error is the rotation about the world vertical axis
between the same two poses, computed as the world-Z component of
`R_last · R_0ᵀ` — verified to agree with the reported yaw difference to two
decimal places across all five runs.

On rectangle run 1 the operator walked past the mark before stopping, so error
is measured at the keyframe of closest approach to the marked origin (KF 99)
rather than the final keyframe. All other runs ended stationary on the mark.

---

## Loop closure: how it works and why it needed to

Closure detection is a proximity search over past keyframes, followed by
**coarse-to-fine ICP verification** on each candidate. Stage 1 runs at a 0.3 m
correspondence radius for reach; stage 2 re-runs from stage 1's transform at
the normal 0.09 m radius for an honest score. Gating uses stage 2 only, since
stage 1's fitness is inflated by construction at the loose radius.

Two design decisions made closure work at all:

**1. The verification ICP is seeded with `Tᵢ⁻¹ · Tⱼ` from live optimized poses.**
Originally identity, which started ICP a full drift-distance away with a 9 cm
correspondence radius. It found nothing and returned fitness 0.000 every time.

**2. Coarse-to-fine, not a single wide pass.** Rotation error displaces points
in proportion to range: 6.5° of yaw error moves a 3 m wall point 34 cm but a
0.5 m floor point only 5.7 cm. At a 9 cm radius, near points find partners and
far points don't, capping fitness near 0.6 regardless of how good the alignment
actually is. Before this change candidates topped out at 0.510–0.699 and
nothing was ever accepted; after it, accepted closures score 0.93–0.99.

The rectangle runs capture the whole progression in a single log, as the
operator walks back toward the start:

| Seed distance | Coarse fitness | Fine fitness | Outcome |
|---|---|---|---|
| 0.90 m | 0.000 | 0.000 | reject |
| 0.77 m | 0.000 | 0.000 | reject |
| 0.66 m | 0.436 | 0.270 | reject |
| 0.54 m | 0.908 | 0.651 | reject |
| **0.11 m** | **0.998** | **0.929** | **accept** |

The two failure modes are distinguishable by fitness alone: a seed far outside
the coarse radius gives fitness ≈ 0.0, because every point is displaced equally
and all fail together. A rotation-dominated failure gives 0.3–0.7, because near
points match and far points don't. Widening the correspondence distance does
not fix the second case — it lets points on genuinely different surfaces pair
up, inflating fitness without improving alignment.

---

## Performance

Median per-keyframe cost, measured over a full run:

| Stage | Median | Note |
|---|---|---|
| ICP registration | ~250 ms | dominant cost |
| Map rebuild | ~80 ms | own thread, non-blocking |
| iSAM2 update | 1–2 ms | never the bottleneck |
| **Total** | **~348 ms** | |
| Closure keyframes | 1.4–1.8 s | verification ICP on each candidate |

Runs sustain ~100 keyframes over a 4-minute walk with zero to one weak ICP
edge per run.

**Profiling changed where the optimization effort went.** iSAM2 was the
intuitive suspect and measures at 1–2 ms — targeting it would have achieved
nothing. Within ICP, the registration solver itself is only 27–79 ms; voxel
downsampling and normal estimation account for roughly 80% of ICP time. That
is the next optimization target, and it is not where the search started.

---

## Engineering notes

**IMU delivery: 9.4 Hz → 202 Hz.** Three compounding causes. The SDK's frame
aggregate mode was set to require every stream before emitting a frameset,
pinning IMU delivery to the depth rate. The driver used a polling loop, where
each call returns one frameset, capping delivery at ~84 Hz. And depth-to-cloud
conversion (~120 ms) ran on the delivery thread, blocking it. Fixing all three
— push callbacks, permissive aggregation, and moving depth conversion to a
worker thread behind a drop-on-full queue — reached 202 Hz with an inter-sample
standard deviation of 0.022 ms. Allan variance was then re-derived at the new
rate, since noise densities are rate-dependent.

**Turn-on bias is measured at startup, not assumed zero.** Allan variance
characterizes how bias *drifts*; it says nothing about the current offset.
Zero-initialized bias forced ~0.075 m/s² of real MEMS offset into pose
estimates — roughly 15 cm of phantom translation per 2-second window.

**Failed ICP registrations become weak edges, not dropped edges.** A rejected
registration used to be discarded entirely, leaving that keyframe constrained
by the IMU factor alone — the one sensor that physically cannot supply
position. That created a hole in the chain, and loop-closure corrections pooled
at the hole, producing a localized map tear. Rejected edges are now added with
sigmas scaled 15×: a poor measurement is still a measurement.

**Sensor and middleware isolation.** The estimation core imports numpy, Open3D,
and GTSAM but never a sensor SDK or ROS. All sensor data crosses a two-dataclass
boundary. The camera SDK is imported in one file, ROS in one other. The core is
unit-testable with neither hardware nor a ROS installation present.

---

## Known limits

**Residual map seam at the closure keyframe.** Structural, not a bug. The map
is per-keyframe point clouds bolted rigidly to poses. A closure moves the
closure keyframe's pose; its neighbours were not part of that measurement and
move less; where their surfaces overlap, they disagree. Removing it entirely
requires a deformable map representation — surfels with a deformation graph
(ElasticFusion), frame de-integration and re-integration (BundleFusion), or a
landmark map optimized jointly with poses. All are substantial rewrites.

**Depth quality is lighting-dependent, and it constrains rotation.** The Gemini
435Le uses active stereo, which needs visible texture. In a dim room the depth
map degrades and ICP is left under-constrained in rotation while still
reporting acceptable fitness. Measured directly: identical loops in low light
ended at −42°, −46°, and −21° of heading error; the same loop with the same
code and the lights on ended at −2.8°. All results above were collected with
room lights on.

**Point-to-plane ICP is degenerate against a flat wall** for motion parallel to
the plane — fitness reads 0.95 while translation is under-measured roughly 5×.
Feature-rich views with visible corners are a requirement, not a preference.
Eigenvalue-based degeneracy detection on the ICP information matrix is the
principled fix and is not yet implemented.

**Scale.** Results above are on 0.6 × 0.9 m and ~0.9 m closed paths. Larger
loops have not been characterized under this protocol.

**Reproducibility caveat.** The loop-closure acceptance threshold was 0.87 on
rectangle run 1 and 0.91 on rectangle run 2. Accepted closures scored
0.929–0.985 in both, above either threshold; one candidate at 0.879 was
accepted under the lower gate and would not have been under the higher.
