# Residual RL training

The driver is **pure pursuit plus a learned correction**. Pure pursuit follows
the racing line built from the map and picks a speed from its braking and
cornering limits. A small neural network, trained here with reinforcement
learning (PPO), adds a capped correction on top: at most ±0.10 rad of
steering and ±1.5 m/s of speed. Without trained weights the driver is plain
pure pursuit, exactly as before.

```
           odometry + LiDAR
                  │
     ┌────────────┴────────────┐
     ▼                         ▼
 pure pursuit ──(its own command, as input)──► network (14 numbers in, 2 out)
     │                         │
     │ (steering, speed)       │ correction, capped by LIMITS
     └──────────► + ◄──────────┘
                  │
                /drive
```

## Where everything is

| File | What it does |
| --- | --- |
| `race_ws/src/team_driver/team_driver/pure_pursuit.py` | The controller and its tuned defaults (`DEFAULTS`). Used by the driver **and** by training. |
| `race_ws/src/team_driver/team_driver/raceline.py` | Builds the racing line from the map (moved out of `driver.py`, unchanged). |
| `race_ws/src/team_driver/team_driver/residual.py` | The network's inputs (`features`), the network (`actor`), `LIMITS`, and how the correction is added. |
| `race_ws/src/team_driver/models/residual.npz` | The trained network the driver loads. Absent = pure pursuit only. |
| `training/sim.py` | The GPU simulator set up like the judged race, with the controller running inside it. |
| `training/train.py` | PPO training, with Adam written out. `--method lookahead` (default) or `--method residual`. |
| `training/evaluate.py` | Races 11 laps in the simulator at every reaction delay and prints lap times and contacts. |
| `training/test_parity.py` | Checks training and racing compute exactly the same thing. |
| `training/runs/` | Output of each training run (not committed). |

The pure pursuit, features and network code is written once, with `xp`
standing for either `numpy` (driver) or `jax.numpy` (training). That is how
the network races exactly the controller it was trained to correct.

## One-time setup

Training runs on your PC, not in the Docker container: it needs the GPU and
JAX with CUDA, which the judging image does not have and does not need.

```sh
training/setup_venv.sh                       # creates ~/venvs/rl (already done on this PC)
~/venvs/rl/bin/python training/test_parity.py
```

## The loop

All commands are run from the repository root.

```sh
# 1. Baseline: how does pure pursuit alone do in the simulator?  (~1 min, CPU)
~/venvs/rl/bin/python training/evaluate.py

# 2. Check training starts, then train for real.  (~2 min, then ~20-25 min, GPU)
~/venvs/rl/bin/python training/train.py --method residual --smoke
~/venvs/rl/bin/python training/train.py --method residual

# 3. Is it better than the baseline?
~/venvs/rl/bin/python training/evaluate.py --weights training/runs/<run>/residual.npz

# 4. If yes: give it to the driver, rebuild, and race it for real.
mkdir -p race_ws/src/team_driver/models
cp training/runs/<run>/residual.npz race_ws/src/team_driver/models/
#    inside the container:
cd /hackathon/race_ws && colcon build --symlink-install && source install/local_setup.bash
#    on the host:
./scripts/evaluate.sh --runs 3 --headless
```

The driver logs `residual network loaded from ...` at startup when it uses
the network, and `no residual network ... pure pursuit only` when it does not.
To race without it, delete the file or pass `-p use_residual:=false`.

### Reading the training output

| Column | Meaning | Want |
| --- | --- | --- |
| `track speed` | average metres of track covered per second, including restarts from standstill | up |
| `crashes/lap` | wall contacts per 78.5 m driven | down, towards 0 |
| `\|steer fix\|`, `\|speed fix\|` | average size of the network's correction | whatever works; at the cap (0.10 rad, 1.5 m/s) means the limits are too tight |
| `noise` | exploration noise; it shrinks as the policy gets sure of itself | down, slowly |

`log.csv` in the run folder has every update, for plotting.

## Dynamic lookahead (the second method)

After Elgouhary & El-Wakeel 2026, *Dynamic Lookahead Distance via
Reinforcement Learning-Based Pure Pursuit for Autonomous Racing*
(arXiv 2603.28625). Instead of correcting pure pursuit's output, the policy
chooses pure pursuit's lookahead distance L every step. Here the policy is
linear, so after training you can read off what it learned:

```
L = b + w_v*v + w_0*|k0| + w_1*|k1| + w_2*|k2| + w_d*dk     clipped to 0.35..4.0 m
```

| Input | Meaning |
| --- | --- |
| `v` | speed [m/s] |
| `|k0|`, `|k1|`, `|k2|` | how sharply the racing line bends at the nearest point, 1 m and 3 m further on [1/m] |
| `dk` | `|k1| - |k0|`: is the bend ahead tightening (+) or opening (-)? |

Training starts from `L = 0.4*v`, close to the hand-made rule, and PPO
adjusts the 5 weights and the bias. Where the code is:

- `residual.py`: the inputs (`lookahead_inputs`), the policy (`lookahead`),
  and the loss (`lookahead_loss`, which calls `ppo_policy_loss`: the paper's
  eq. 24-25).
- `train.py`: the gradient descent. `adam_step` (Adam, written out),
  `clip_by_global_norm`, the linear learning-rate decay, and the target-KL
  early stop. The defaults are the paper's PPO settings (`SETTINGS`).

```sh
~/venvs/rl/bin/python training/train.py --smoke          # 2 minutes
~/venvs/rl/bin/python training/train.py                  # 20M steps, about 10 minutes
~/venvs/rl/bin/python training/evaluate.py --lookahead training/runs/<run>/lookahead.npz
```

Training prints the current equation every 5 updates, and the mean L on
straight and on curved parts of the track. The paper's finding is that the
policy lengthens L on straights and shortens it in curves; check whether
yours does.

The reward is metres of progress, minus 10 per crash, plus three of the
paper's eq. 13 terms, each weighted by a command-line option:
`+ w_speed * v` (`--w-speed`, default 0.002), `- w_jerk * |L - L_before|`
(`--w-jerk`, 0.01) and `- w_stall` for each step after 0.5 s below
0.05 m/s (`--w-stall`, 0.05). The paper does not publish its weights; these
are ours.

Where this differs from the paper: the policy is linear rather than a
neural network; the reward leaves out the paper's ideal-lookahead and
curvature terms; its curvature horizons are not given, so 1 m and 3 m are
ours; L is not smoothed before use; and every episode
gets a random reaction delay (below). The driver does not load
`lookahead.npz` yet.

## How the simulator matches the race, and where it does not

Matches: car model (ST, RK4, 0.01 s), vehicle parameters, speed + steering
angle control, the icra26 map and centreline, LiDAR mounting and beam angles,
odometry at the rear axle. Checked on 2026-10-11: the GPU simulator and the
race bridge's own engine put the car into the infield hairpin at the same
point, (5.09, 9.64).

**Reaction delay is the big unknown.** In ROS a command reaches the physics
a few hundredths of a second after the scan it answers, and it varies. Pure
pursuit is right on the edge in the hairpin around (5.1, 9.7), so this
matters a lot:

| Delay | Pure pursuit alone, race engine without ROS |
| --- | --- |
| 0 or 1 step | crashes in the hairpin and stays stuck |
| 2 or 3 steps | laps in about 17.2 s, with a few wall contacts |
| real ROS run (2026-10-11) | 16.68 s, no contacts |

So each training episode gets a random delay of 0–3 steps, and
`evaluate.py` reports every delay separately. A network worth shipping
should be clean at all of them. Not modelled: the delay changing within a
lap, and the driver occasionally skipping a scan.

## Things to tune

- `LIMITS` in `residual.py`: how much the network may correct. Start small;
  widen once a small residual laps cleanly, and continue from the previous
  weights with `--init`.
- `COLLISION_PENALTY`, `RESIDUAL_COST` in `train.py`: the reward.
- `--steps`, `--envs`, `--lr`, `--gamma` on the command line.
- If you change anything in `pure_pursuit.DEFAULTS`, retrain: the network
  learned to correct the old tuning. The driver warns when they differ.

## For SUBMISSION.md and the interview

Rule 38 asks you to explain the method, not the weights:

- **In:** 14 numbers (`residual.FEATURE_NAMES`): speed, yaw rate, last
  steering, offset and heading error from the racing line, pure pursuit's
  own command, free distance ahead, and the line's curvature 0.6–6 m ahead.
- **Out:** two numbers, squashed by tanh to −1..1 and scaled by `LIMITS`.
- **Trained:** PPO, in f1tenth_gym's JAX simulator on icra26, 2048 cars in
  parallel from random points, random reaction delay 0–3 steps.
- **Reward:** metres of progress per step, −10 for touching a wall (which
  ends the episode), and a small cost for large corrections.
- **Validated:** `evaluate.py` at every delay, then `scripts/evaluate.sh`.
- **Where it can fail:** wherever the simulator differs from ROS (timing
  above all), and anywhere the capped correction is not enough to save a
  bad pure pursuit decision.

The PPO code follows `external/f1tenth_gym/examples/jax_ppo_training.py`
(MIT licence), which follows PureJaxRL. Credit both in `SUBMISSION.md`.
