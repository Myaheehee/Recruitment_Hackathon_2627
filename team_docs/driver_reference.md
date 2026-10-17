# `team_driver` — our race driver

A ROS 2 node that drives the F1TENTH car around the hackathon track:

1. **At startup** it builds its own **racing line** from the map: the smoothest
   path that keeps 20 cm from every wall.
2. **Each lap point** gets a target speed from a **speed table** tuned with
   reinforcement learning (PPO).
3. **On every LiDAR scan** it steers with **pure pursuit**. A small learned
   policy chooses how far ahead to aim.

It laps the official track in about **13.7 s**.

> **New to the project?** Read the two-page [overview](README.md) first.
> This page is the reference: how to run it, what every setting does, and how
> the files fit together.

---

## Quick start

All commands run **inside the hackathon container**, with ROS sourced.

```sh
cd /hackathon/race_ws
colcon build --symlink-install          # also installs models/*.npz
source install/local_setup.bash
ros2 run team_driver driver             # exactly how the judges run it
```

### Example: what a healthy start looks like

The driver logs what it built and loaded:

```text
racing line: 380 points, 75.9 m per lap (centreline ...), closest approach to a wall 0.3x m
speed table tuned, from .../models/speed_table.npz, scaled by 1.0: 3.7x to 9.2x m/s, 12.28 s per lap if driven exactly
lookahead policy loaded from .../models/lookahead.npz: L = 0.004 +0.249*v +0.003*|k0| ...
pure pursuit settings changed since the policy was trained: ['table_scale', 'speed_lead'] ...   <- expected, harmless
reaction delay over 200 scans [steps]: age mean ... max ...; gap mean ... max ...
```

- If you see **"following the centreline instead"**, the racing line came too
  close to a wall.
- If you see **"tuned on a different racing line"**, the tuned speed table was
  rejected (see [Safety and fallbacks](#safety-and-fallbacks)).

### Changing a setting

```sh
ros2 run team_driver driver --ros-args -p table_scale:=0.95      # 5% slower everywhere
ros2 run team_driver driver --ros-args -p use_lookahead:=false   # plain pure pursuit lookahead
ros2 launch team_driver driver.launch.py                         # uses config/driver_params.yaml (see note below)
```

### Handy while testing

```sh
tools/reset_car.sh     # (repo root) kill the driver, stop the car, put it back on the grid
ros2 topic pub --once /drive ackermann_msgs/msg/AckermannDriveStamped "{drive: {speed: 0.0, steering_angle: 0.0}}"
ros2 topic echo /ego_racecar/odom --no-arr     # watch odometry
pkill -f team_driver                           # lost the driver?
```

---

## How it works

| File | What it does | Main functions |
| --- | --- | --- |
| [team_driver/driver.py](../race_ws/src/team_driver/team_driver/driver.py) | The ROS node: parameters, startup, callbacks, publishing | [`Driver.build_path()`](../race_ws/src/team_driver/team_driver/driver.py#L99), [`Driver.build_speed_table()`](../race_ws/src/team_driver/team_driver/driver.py#L123), [`Driver.load_policy()`](../race_ws/src/team_driver/team_driver/driver.py#L152), [`Driver.plan()`](../race_ws/src/team_driver/team_driver/driver.py#L230), [`Driver.scan_callback()`](../race_ws/src/team_driver/team_driver/driver.py#L199) |
| [team_driver/pure_pursuit.py](../race_ws/src/team_driver/team_driver/pure_pursuit.py) | Steering and speed for one tick. No ROS, numpy or JAX | [`pure_pursuit()`](../race_ws/src/team_driver/team_driver/pure_pursuit.py#L51), [`preprocess()`](../race_ws/src/team_driver/team_driver/pure_pursuit.py#L45), [`DEFAULTS`](../race_ws/src/team_driver/team_driver/pure_pursuit.py#L16) |
| [team_driver/raceline.py](../race_ws/src/team_driver/team_driver/raceline.py) | Clearance map, racing line, physics speed table | [`load_clearance_map()`](../race_ws/src/team_driver/team_driver/raceline.py#L146), [`build_raceline()`](../race_ws/src/team_driver/team_driver/raceline.py#L30), [`speed_table()`](../race_ws/src/team_driver/team_driver/raceline.py#L110), [`limit_speed_changes()`](../race_ws/src/team_driver/team_driver/raceline.py#L120) |
| [team_driver/policies.py](../race_ws/src/team_driver/team_driver/policies.py) | Everything learned: lookahead policy, speed-table loader, PPO losses | [`lookahead_inputs()`](../race_ws/src/team_driver/team_driver/policies.py#L45), [`lookahead()`](../race_ws/src/team_driver/team_driver/policies.py#L52), [`load_lookahead()`](../race_ws/src/team_driver/team_driver/policies.py#L99), [`load_speed_table()`](../race_ws/src/team_driver/team_driver/policies.py#L133) |
| [team_driver/units.py](../race_ws/src/team_driver/team_driver/units.py) | Unit labels for type hints (`Metres`, `Radians`, ...) | — |
| [models/](../race_ws/src/team_driver/models/) | Trained weights, produced by [training/train.py](../training/train.py) | `lookahead.npz`, `speed_table.npz` |
| [config/driver_params.yaml](../race_ws/src/team_driver/config/driver_params.yaml) | Parameter file, used **only** by the launch file | — |
| [launch/driver.launch.py](../race_ws/src/team_driver/launch/driver.launch.py) | Starts the node with the yaml | — |

**Why there is no ROS in three of the four files.** The training code in
[training/](../training/) runs `pure_pursuit.py`, `raceline.py` and
`policies.py` *unchanged* inside a GPU simulator written in JAX. Every function
takes `xp` = `numpy` or `jax.numpy`. One copy of the maths means training and
racing cannot drift apart.
[training/test_parity.py](../training/test_parity.py) checks that they
agree.

### One control tick, step by step

This is [`Driver.plan()`](../race_ws/src/team_driver/team_driver/driver.py#L230), called from [`Driver.scan_callback()`](../race_ws/src/team_driver/team_driver/driver.py#L199).

1. **No odometry yet?** Send speed 0 and steering 0. ([`Driver.odom_callback()`](../race_ws/src/team_driver/team_driver/driver.py#L191) stores each odometry message.)
2. **Choose the lookahead L.** If the lookahead policy is loaded, [`lookahead_inputs()`](../race_ws/src/team_driver/team_driver/policies.py#L45) and [`lookahead()`](../race_ws/src/team_driver/team_driver/policies.py#L52) compute
   `L = b + w · [v, |κ0|, |κ1|, |κ2|, |κ1|−|κ0|]`, clipped to 0.35–4 m.
   Otherwise pure pursuit uses `max(min_lookahead, lookahead_gain·v)`.
3. **[`pure_pursuit()`](../race_ws/src/team_driver/team_driver/pure_pursuit.py#L51):**
   - find the nearest line point
   - find the target L ahead
   - compute the curvature `κ = 2y/L²`
   - compute the steering `atan(wheelbase·κ)`, clipped to ±`max_steer`
   - read the speed from the table, at the nearest point (+ `speed_lead`
     term, off by default)
4. **Publish** an `AckermannDriveStamped` on `/drive` ([`Driver.publish()`](../race_ws/src/team_driver/team_driver/driver.py#L253)). Every 10th scan, also
   publish RViz markers on `/driver/markers` ([`Driver.publish_markers()`](../race_ws/src/team_driver/team_driver/driver.py#L264)).
   Every 200 scans, [`Driver.measure_delay()`](../race_ws/src/team_driver/team_driver/driver.py#L211) logs the reaction delay.

---

## How the code works, file by file

The code itself keeps only one-line docstrings and type hints with units
(`yaw: Radians`, from [units.py](../race_ws/src/team_driver/team_driver/units.py)).
The background lives here.

### `pure_pursuit.py`

- **Why `xp`.** numpy and jax.numpy share function names, so each function takes
  the library as an argument: numpy in the driver, jax.numpy in training. For the
  same reason there is no `while` or `if` on an array value (JAX cannot compile
  them); `if lookahead is None` is fine, because Python decides it before JAX
  compiles.
- **Steps of `pure_pursuit()`:**
  1. nearest line point to the rear axle;
  2. lookahead L = `max(min_lookahead, lookahead_gain·v)` unless the policy chose it;
  3. target = the first point, going round from the nearest one, at least L away
     (`argmax` of "distance ≥ L" over the indices in lap order);
  4. how far left the target is, in the car's frame (rotate dx, dy by −yaw);
  5. steering: the circle through car and target has curvature κ = 2·left/L²,
     and a car with wheelbase b drives κ with steering atan(b·κ) (bicycle model);
  6. free distance ahead = shortest beam within ±`ahead_angle`;
  7. speed = table speed at the nearest point, plus the lead
     `speed_lead · (1 − |steering|/max_steer) · (v_1m_ahead − v_here)`: the car's
     speed always lags its command, so it brakes and accelerates a little early,
     but not mid-corner. Without a table: the smallest of the braking limit
     √(2·a_brake·(ahead − margin)), the cornering limit √(μg/κ) and `v_max`.
- **History of `DEFAULTS`.** A 2026-10-12 sweep found every `lookahead_gain`
  from 0.2 to 0.35 laps cleanly, while 0.4 cuts the infield hairpin.

### `raceline.py`

Slide each centreline point sideways so the lap bends as little as possible,
keeping the car's side `wall_gap` from the walls; a gently bending line can be
driven faster (corner speed √(μg/|κ|)), and "wide in, clip the apex, wide out" is
what a bend-minimising line looks like.

- **`load_clearance_map()`**: a pixel is wall when (255 − value)/255 >
  `occupied_thresh`. Cone rows are sealed by a morphological *closing* with a
  0.4 m disc: every wall grows by the disc and shrinks back, so gaps under
  0.8 m fill in (cones are 0.25–0.5 m apart; corridors are over 1 m). A distance
  transform then gives each pixel's distance to the nearest wall.
- **`build_raceline()`**, four steps:
  1. `left_directions()`: unit vector to the left of travel (next − previous
     point via `np.roll`, which wraps round the loop, turned 90°);
  2. `slide_limits()`: how far each point may slide each way, testing every
     0.025 m up to 3 m (`cumprod` keeps the run of positions that fit);
  3. `minimise_bending()`: projected gradient descent on Σ bend², where
     bend = previous − 2·point + next. The gradient for one point is the same
     pattern applied to `bend`; each point slides against it by `NUDGE` (0.12;
     bigger makes it blow up), then is clamped to its limits. 20,000 steps;
  4. `resample()`: redraw the line with points 0.2 m apart.

  If the result is closer than `margin − 0.05` m to a wall, the centreline is
  used instead.
- **`speed_table()`**: rule 1, each point no faster than √(μg/|κ|); rule 2,
  neighbours d apart obey v_next² ≤ v² + 2·a_brake·d (the distance to change
  speed at `a_brake`). `limit_speed_changes()` enforces rule 2 by walking round
  the lap forwards (speeding up) and backwards (braking), repeating until
  nothing changes, since the lap is a loop. Training reuses it to keep the tuned
  table within the rule.

### `policies.py`

The two learned parts, both trained with PPO in [training/](../training/README.md).

- **`path_geometry()`**: heading = direction from the previous point to the
  next; curvature = how fast that heading turns per metre (positive = left).
- **Lookahead policy** (after Elgouhary & El-Wakeel, arXiv 2603.28625): linear,
  L = b + w·s, so what it learned can be read off. Inputs s: speed, |κ| at the
  nearest point, 1 m and 3 m ahead (our choice; the paper gives no horizons),
  and dk = |κ1| − |κ0| (bend tightening or opening). L is clipped to 0.35–4 m.
  Training starts from L = 0.25·v, the hand rule, and ended at L ≈ 0.249·v.
- **Speed table**: the "policy" is the 380 speeds. The driver uses a tuned
  table only if the line stored with it matches the line built at startup
  (same length, within 1 mm); otherwise it falls back to the physics table.
- **Training-only maths**: `gaussian_log_prob()` (how likely a randomly
  explored choice was), `ppo_policy_loss()` (make better-than-expected choices
  more likely, but clip the change per update; `kl` measures how far the policy
  moved, `weight` limits learning to cars inside the tuned stretch),
  `speed_table_inputs()` (what the critic sees: the 5 lookahead inputs plus the
  car's position round the lap as sin/cos).

### `driver.py`

- **ROS.** Nodes talk by publishing messages on topics; a callback is the
  function ROS calls when a message arrives. `rclpy.spin()` runs callbacks one
  at a time (odometry never changes in the middle of `plan()`) and only after
  `__init__` returns, so the car waits on the grid while the line is built.
- **Topics**: in `/scan` (819 beams over 270°) and `/ego_racecar/odom`; out
  `/drive` and `/driver/markers` (RViz: 0 steering arrow, 1 centreline,
  2 racing line). The inputs have queue length 1: headless, the simulator runs
  faster than real time, and answering an old scan is exactly the reaction
  delay that made the car late into corners.
- **Settings.** Each `DEFAULTS` key becomes a ROS parameter whose default fixes
  its type: `-p v_max:=6.0` works, `-p v_max:=6` stops the node.
- **Odometry.** The position is the rear axle, which pure pursuit's geometry
  assumes. Yaw comes from the quaternion: for a car turning only about the
  vertical axis, z = sin(yaw/2) and w = cos(yaw/2).
- **`measure_delay()`.** Scans and odometry carry the simulator's clock. *Age*
  = steps between the scan and the newest odometry when the command goes out
  (the true delay is at least this); *gap* = steps between answered scans
  (1 = keeping up). Compare with `training/evaluate.py --max-delay N`.

---

## Parameters

Each one is a ROS parameter (`-p name:=value`). The race values are the
defaults in [`DEFAULTS`](../race_ws/src/team_driver/team_driver/pure_pursuit.py#L16) and [`Driver.setup_parameters()`](../race_ws/src/team_driver/team_driver/driver.py#L77). The judges
pass **no parameter file**.

### Driving (`pure_pursuit.DEFAULTS`)

Each name links to its line in [`DEFAULTS`](../race_ws/src/team_driver/team_driver/pure_pursuit.py#L16).

| Name | Default | Unit | What it controls | Used when |
| --- | --- | --- | --- | --- |
| [`v_max`](../race_ws/src/team_driver/team_driver/pure_pursuit.py#L17) | 6.8 | m/s | Top speed | Speed table off |
| [`a_brake`](../race_ws/src/team_driver/team_driver/pure_pursuit.py#L18) | 6.05 | m/s² | Braking/acceleration we plan with | Physics table, or speed table off |
| [`margin`](../race_ws/src/team_driver/team_driver/pure_pursuit.py#L19) | 0.55 | m | Stop this far short of a wall | Speed table off |
| [`mu`](../race_ws/src/team_driver/team_driver/pure_pursuit.py#L20) | 1.0489 | — | Tyre grip for the corner speed `√(μg/κ)` | Physics table, or speed table off |
| [`ahead_angle`](../race_ws/src/team_driver/team_driver/pure_pursuit.py#L21) | 10.0 | deg | Half-width of the "straight ahead" LiDAR cone | Speed table off |
| [`max_range`](../race_ws/src/team_driver/team_driver/pure_pursuit.py#L22) | 8.0 | m | LiDAR readings clipped to this | Always |
| [`lookahead_gain`](../race_ws/src/team_driver/team_driver/pure_pursuit.py#L23) | 0.25 | s | L = gain × speed | No lookahead policy |
| [`min_lookahead`](../race_ws/src/team_driver/team_driver/pure_pursuit.py#L24) | 0.5 | m | Shortest L | No lookahead policy |
| [`wheelbase`](../race_ws/src/team_driver/team_driver/pure_pursuit.py#L25) | 0.33 | m | Car geometry for steering | Always |
| [`max_steer`](../race_ws/src/team_driver/team_driver/pure_pursuit.py#L26) | 0.4189 | rad | Steering limit (±24°) | Always |
| [`use_speed_table`](../race_ws/src/team_driver/team_driver/pure_pursuit.py#L27) | `True` | — | Table speeds vs the old "minimum of three limits" | Always |
| [`table_scale`](../race_ws/src/team_driver/team_driver/pure_pursuit.py#L28) | 1.0 | — | Multiplies every table speed (<1 = safety margin) | Speed table on |
| [`speed_lead`](../race_ws/src/team_driver/team_driver/pure_pursuit.py#L29) | 0.0 | — | Brake/accelerate earlier by k × (speed change over the next 1 m); *experimental* | Speed table on |

### Files, topics and switches (`Driver.setup_parameters`)

All of these are declared in [`Driver.setup_parameters()`](../race_ws/src/team_driver/team_driver/driver.py#L77).

| Name | Default | What it does |
| --- | --- | --- |
| `scan_topic` / `odom_topic` / `drive_topic` | `/scan`, `/ego_racecar/odom`, `/drive` | Topics |
| `centerline_csv`, `map_yaml` | `/hackathon/maps/icra26_centerline.csv`, `/hackathon/maps/icra26.yaml` | Track inputs |
| `wall_gap` | 0.2 m | Air between the car's side and a wall on the racing line. **Changing it changes the line, so the tuned table is rejected** |
| `raceline_iters` | 20000 | Smoothing iterations: more gives a smoother line and a slower startup |
| `raceline_csv` | `/tmp/raceline.csv` | Where a copy of the line is saved for plotting |
| `use_raceline` | `True` | `False` = drive the plain centreline |
| `use_lookahead` / `lookahead_weights` | `True` / `''` | Use `models/lookahead.npz` (or this path) |
| `speed_table_weights` | `''` | Path to a tuned table (blank = `models/speed_table.npz`) |

> **Note on `config/driver_params.yaml`.** It sets different values (e.g.
> `lookahead_gain: 0.4`, `v_max: 6.0`). It is only read by
> `ros2 launch team_driver driver.launch.py`, **not** by `ros2 run`. With the
> shipped models loaded, most of its values aren't used anyway: the policy picks
> L, and the tuned table picks the speed.

---

## Models (`models/`)

| File | What is inside | Made by |
| --- | --- | --- |
| `lookahead.npz` | `w` (5), `b`, `lookahead_range`, critic weights, settings used | `train.py --method lookahead`, run `20261013-030026` |
| `speed_table.npz` | `speeds` (380), the `line` it was tuned on, critic weights, settings used | `train.py --method speed_table`, run `20261013-021003` |

Every file also stores the [`DEFAULTS`](../race_ws/src/team_driver/team_driver/pure_pursuit.py#L16) it was trained with. If they differ from
the current settings, the driver prints a warning. How each model was trained
is in [training/README.md](../training/README.md).

---

## Safety and fallbacks

| Situation | What the driver does |
| --- | --- |
| No odometry received yet | Sends speed 0 and steering 0 |
| Racing line passes closer than `margin − 0.05` m to a wall | Follows the centreline instead |
| Tuned table was made for a different line | Ignores it and uses the physics table from `mu` and `a_brake` |
| A model file is missing, or `use_*:=false` | Skips it; plain pure pursuit still drives |
| Lookahead policy outputs something silly | Clipped to 0.35–4 m |
| Scans pile up | Subscriptions have queue depth 1, so old scans are dropped and we always answer the newest |

## Known limitations

- **No obstacle braking with the speed table.** In table mode, speed ignores
  the LiDAR and `v_max`. This is fine on the empty race track, but it is not
  safe with obstacles.
- **The tuned table is tied to one racing line.** Changing the map, `wall_gap`
  or `raceline_iters` rejects it.
- **A "settings changed" warning appears at every startup.** It names
  `table_scale` and `speed_lead`, which were added after the models were saved.
  It is harmless.
- **No extra packages need declaring.** `raceline.py` imports `scipy`,
  `Pillow` (`PIL`) and `PyYAML`, and rule 33 lists all three as already
  installed in the image, so `requirements.txt` is empty.
- **Startup takes a few seconds** while the line is smoothed.

## Contributing (for teammates)

- **Keep `pure_pursuit.py`, `raceline.py` and `policies.py` free of ROS**, and
  write them with `xp.` instead of `np.`. Avoid `if`/`while` on array values,
  because JAX cannot compile them.
- **After changing any of them, run the parity tests:**
  `~/venvs/rl/bin/python training/test_parity.py`.
- **After changing [`DEFAULTS`](../race_ws/src/team_driver/team_driver/pure_pursuit.py#L16), the racing line or the reward, retrain the
  models.** They learned to work with the old settings.
- **Let the code explain itself:** type hints with units from `units.py`
  (`speed: MetresPerSecond`), a one-line docstring, and short comments only
  for *why*. Longer background goes in this file.

## References

- Pure pursuit: R. C. Coulter, *Implementation of the Pure Pursuit Path Tracking
  Algorithm*, CMU-RI-TR-92-01, 1992.
- Learned lookahead: Elgouhary & El-Wakeel, arXiv:2603.28625, 2026.
- PPO: Schulman et al., arXiv:1707.06347, 2017. GAE: Schulman et al.,
  arXiv:1506.02438, 2016.

## Licence and contact

This package is part of our entry to the NTU hackathon, built on the
organisers' template repository. That repository has no licence file, so no
open-source licence is granted. Questions: contact the team (see
[SUBMISSION.md](../SUBMISSION.md)).
