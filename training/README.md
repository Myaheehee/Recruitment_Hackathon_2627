# Training: teaching pure pursuit with reinforcement learning

Our car is driven by **pure pursuit**: a simple, hand-written rule that steers
towards a point a little way ahead on the racing line. This folder trains
small **learned pieces** that plug into pure pursuit, using reinforcement
learning (RL) in a GPU simulator. Training happens on your PC; the car only
ever loads the finished numbers from a `.npz` file and runs them with numpy.

There are two methods, chosen with `train.py --method`:

| `--method` | What it learns | Size | Shipped? |
| --- | --- | --- | --- |
| `lookahead` (default) | how far ahead pure pursuit aims, `L` | 5 weights + 1 bias | **yes**, `models/lookahead.npz` |
| `speed_table` | the target speed at every point of the racing line | 380 speeds | **yes**, `models/speed_table.npz` |

(An earlier version also had a residual network that corrected pure
pursuit's steering and speed; it was moved into `unused/residual/` because it
was never trained to a shippable model.)

The short version of the result: **the speed table did the work.** PPO
tuned it from a 13.51 s lap to a 12.28 s lap (if driven exactly). The
lookahead policy, trained afterwards on top of that table, ended up almost
exactly where it started (the hand rule `L = 0.25 × speed`). See
[What ships, honestly](#1-what-ships-honestly).

For the bigger picture (how the whole driver works) read the two-page
[overview](../team_docs/README.md) and the
[driver reference](../team_docs/driver_reference.md) in `team_docs/`.

**Contents**
1. [What ships, honestly](#1-what-ships-honestly)
2. [RL in 5 minutes (for this code)](#2-rl-in-5-minutes-for-this-code)
3. [How the learned pieces plug into pure pursuit](#3-how-the-learned-pieces-plug-into-pure-pursuit)
4. [Where everything is](#4-where-everything-is)
5. [Install](#5-install)
6. [Worked example: your first run](#6-worked-example-your-first-run)
7. [Method 1: the speed table](#7-method-1-the-speed-table)
8. [Method 2: dynamic lookahead](#8-method-2-dynamic-lookahead)
9. [How PPO is written here, step by step](#9-how-ppo-is-written-here-step-by-step)
10. [Reaction delay, and how the simulator matches the race](#10-reaction-delay-and-how-the-simulator-matches-the-race)
11. [Reference: every script and flag](#11-reference-every-script-and-flag)
12. [Reading the training output](#12-reading-the-training-output)
13. [Reproduce the shipped models](#13-reproduce-the-shipped-models)
14. [test_parity.py: the four checks](#14-test_paritypy-the-four-checks)
15. [Where it can fail](#15-where-it-can-fail)
16. [Things to tune](#16-things-to-tune)
17. [References and credits](#17-references-and-credits)

---

## 1. What ships, honestly

| File in `race_ws/src/team_driver/models/` | Made by run | What it is |
| --- | --- | --- |
| `speed_table.npz` | `20261013-021003` (continued from `20261013-014155`) | 380 speeds, 3.74 to 9.27 m/s. Lap time if driven exactly: **12.28 s** (the untuned table: 13.51 s). |
| `lookahead.npz` | `20261013-030026` | `L = 0.004 + 0.249·v + 0.003·\|k0\| + 0.001·\|k1\| + 0.004·\|k2\| − 0.001·dk` |

**The speed table.** Compared with the table the driver would build by
itself, PPO raised 255 of the 380 speeds and lowered 123, by at most about
1.9 m/s either way (on average +0.39 m/s). On paper that is 1.2 s per lap.
The real car is slower than "driven exactly" because it reacts late; the
commit that shipped these files (`f9e6c43`) is titled "13.7 second laptime".

**The lookahead policy.** It started at `L = 0.25·v` (the hand rule) and
finished at `w_v = 0.249`, with every other weight between −0.001 and
+0.004. At 6 m/s in a tight bend (curvature 1 /m) the curvature terms change
L by under 1 cm out of about 1.5 m. In practice the car steers with the
hand rule. What that means:

- The hand rule was already close to a *local optimum* for this track: small
  changes to the weights did not make laps measurably better or worse.
- The reward gave PPO very little to go on. The difference that a few
  centimetres of L makes to "metres of progress" is tiny next to the noise
  from random start points, random delays and the odd crash, so the
  averaged gradient was close to zero. (The run was also short: 2M steps,
  30 updates, about 2 minutes.)
- An earlier 20M-step run on the *untuned* table did move the weights
  (`w_v` 0.32, `w_|k2|` 0.10: run `20261012-014600`), so the
  method can learn; on the tuned table it found nothing worth changing.

So: the lap-time gain came from the speed table. Say that plainly in the
interview; it is a good answer, not a weak one.

---

## 2. RL in 5 minutes (for this code)

**Reinforcement learning** = learning by trial and error. Something tries
actions, gets a score, and slowly changes itself to get higher scores.

| Word | Meaning here |
| --- | --- |
| **agent** | our controller: pure pursuit plus the learned piece |
| **environment** | the simulator: the icra26 track and the car's physics ([sim.py](sim.py)) |
| **step** | 0.01 s of simulated time. The agent acts once per step, like the real driver acts once per LiDAR scan. |
| **observation** | what the agent sees each step. Lookahead: 5 numbers (speed and how much the line bends ahead). |
| **action** | what it decides. Lookahead: L in metres. Speed table: a speed. |
| **reward** | the score for one step: metres of track covered, minus 10 for touching a wall (details per method below). |
| **episode** | one attempt, from a start to a crash or the 30 s time limit. Then the car is put back at a random place on the track. |
| **policy** | the thing being learned: observation in, action out. |
| **critic** (value function) | a helper network used only in training. It guesses "how much reward is still to come from here". It never drives. |

**Why 2048 cars at once?** One car learns slowly: each lap gives a few
thousand noisy examples. JAX (Google's numpy-for-GPUs) runs the same
simulator code on 2048 cars side by side on the graphics card, each from a
random place on the track and with a random reaction delay. That is about
90,000 simulated steps per second (15 minutes of driving per real second),
so a 20M-step run takes about 5 minutes.

**PPO in plain words.** PPO (Proximal Policy Optimisation) repeats three
things:

1. **Collect experience.** Every car drives 128 steps (1.28 s). During
   training the policy adds random noise to its action (exploration), so it
   tries things it would not otherwise. 2048 × 128 = 262,144 samples.
2. **Estimate advantages.** For each step: did things go *better or worse
   than the critic expected*? That number is the *advantage*. It is worked
   out with **GAE** (generalised advantage estimation), which blends
   "the very next reward" with "everything after it", trusting far-away
   rewards a bit less (`gamma` 0.99 per step: a reward 1 s away counts about
   0.37 as much).
3. **Nudge the policy, but not too far.** Make actions with positive
   advantage a bit more likely and negative ones a bit less likely, using
   gradient descent. The **clip** (0.2) stops any action becoming more than
   20 % more or less likely in one update, so one lucky batch cannot wreck
   the policy. That is the "proximal" (= nearby) in PPO.

Two safety and exploration knobs:

- **KL early stop.** KL divergence measures how different the new policy is
  from the one that collected the data. If it passes 1.5 × `target_kl`
  (1.5 × 0.015) mid-update, the rest of that update is skipped.
- **Entropy bonus.** Entropy measures how random the policy still is. A
  small bonus for keeping it (`--entropy`, 0.02 for lookahead) stops the
  policy from becoming sure of itself too early and stopping exploring.

Then repeat, about 76 times for a 20M-step lookahead run.

---

## 3. How the learned pieces plug into pure pursuit

```
            odometry (position, heading, speed)  +  LiDAR scan,  every 0.01 s
                                     │
                                     ▼
 ┌─ lookahead.npz ─────────┐   ┌──────────────────────────────────────┐
 │ L = b + w·(v,|k0|,...)  │──►│ PURE PURSUIT  (pure_pursuit.py)       │
 │ "aim L metres ahead"    │ L │  1. nearest point of the racing line  │
 └─────────────────────────┘   │  2. target point L ahead              │
 ┌─ speed_table.npz ───────┐   │  3. steering = arc through target     │
 │ 380 speeds, one per     │──►│  4. speed = table at nearest point    │
 │ point of the line       │ v │     (+ optional speed_lead)           │
 └─────────────────────────┘   └───────────────────┬──────────────────┘
                                                   │ steering, speed
                                                   ▼
                                                /drive
```

Delete a file and that piece falls back to the hand-made rule: no
`lookahead.npz` → `L = max(0.5, 0.25·v)`; no `speed_table.npz` → the
table built from `mu` and `a_brake`.

The key trick: pure pursuit, the policy inputs and the policies are written
**once**, with `xp` standing for either `numpy` (driver) or `jax.numpy`
(training). So the car races exactly the controller it was trained with.
[test_parity.py](#14-test_paritypy-the-four-checks) checks this.

---

## 4. Where everything is

| File | What it does |
| --- | --- |
| [`../race_ws/src/team_driver/team_driver/pure_pursuit.py`](../race_ws/src/team_driver/team_driver/pure_pursuit.py) | The controller and its race settings ([`DEFAULTS`](../race_ws/src/team_driver/team_driver/pure_pursuit.py#L31)). Used by the driver **and** by training. |
| [`../race_ws/src/team_driver/team_driver/raceline.py`](../race_ws/src/team_driver/team_driver/raceline.py) | Builds the racing line from the map, and the starting speed table (`speed_table`, `limit_speed_changes`, `lap_time`). |
| [`../race_ws/src/team_driver/team_driver/policies.py`](../race_ws/src/team_driver/team_driver/policies.py) | Both policies: their inputs, the policy functions, PPO's loss, and the loaders the driver uses. |
| `../race_ws/src/team_driver/models/*.npz` | The trained files the driver loads. Absent = hand-made rule. |
| [`sim.py`](sim.py) | The GPU simulator set up like the judged race, with the controller inside it, and the reaction delay. |
| [`train.py`](train.py) | PPO training (Adam written out by hand). `--method lookahead` (default) or `speed_table`. |
| [`evaluate.py`](evaluate.py) | Races 1 warm-up + 10 timed laps in the simulator, once per reaction delay, and prints lap times and wall contacts. |
| [`speed_table.py`](speed_table.py) | Prints and draws the *starting* (untuned) speed table: `speed_table.csv`, `speed_table.png`. |
| [`test_parity.py`](test_parity.py) | Checks training and racing compute exactly the same thing. |
| `requirements.txt`, `setup_venv.sh` | The training environment. |
| `runs/<date-time>/` | One folder per training run: `args.json`, `log.csv`, the policy `.npz`. Not committed (`.gitignore`). |

---

## 5. Install

Training runs on your PC, **not** in the Docker container: it needs an
NVIDIA GPU and JAX with CUDA, which the judging image does not have and
does not need.

```sh
training/setup_venv.sh                       # makes ~/venvs/rl (Python 3.12), ~3 GB the first time
~/venvs/rl/bin/python training/test_parity.py   # should end "All parity checks passed."
```

`setup_venv.sh` needs [uv](https://docs.astral.sh/uv/) and an NVIDIA driver
new enough for CUDA 13. It installs (`requirements.txt`):

- `./external/f1tenth_gym`: the simulator, from the repo's pinned submodule, so training uses the race's physics.
- `jax[cuda13]==0.11.2`: the same JAX version as the judging image.
- `optax==0.2.8`: listed but **not used** any more; Adam is written out in `train.py`. Safe to remove.

No GPU? Everything also runs with `--device cpu`, just far slower.

---

## 6. Worked example: your first run

All commands from the repository root.

```sh
# 1. A tiny run (64 cars, 10 updates) to check everything works. About 1 minute,
#    most of it compiling.
~/venvs/rl/bin/python training/train.py --smoke
```

It prints one line every 5 updates. From the smoke run `20261012-012215`
(yours will differ a little):

```
   10    0.04M steps    1934/s  speed 4.29 m/s  crashes/lap 0.000  noise 0.30  L straight 1.27 m, curve 0.82 m  stalled 0.0%
       L = 0.006 + 0.261*v + 0.003*|k0| - 0.008*|k1| - 0.000*|k2| - 0.014*dk
```

Read it as: after 10 updates (40,960 simulated steps), the cars cover 4.29 m
of track per second, never crashed, still explore with ±0.30 m of noise on
L, aim 1.27 m ahead on straights and 0.82 m in bends, and the policy is the
equation on the second line (still close to `0.25·v`).

```sh
# 2. Race what it learned in the simulator, at every reaction delay 0..3.
~/venvs/rl/bin/python training/evaluate.py --lookahead training/runs/<run>/lookahead.npz

# 3. Race the shipped setup the same way (both files!):
~/venvs/rl/bin/python training/evaluate.py \
    --lookahead race_ws/src/team_driver/models/lookahead.npz \
    --speed-table race_ws/src/team_driver/models/speed_table.npz
```

`evaluate.py` prints, for each delay, laps driven, best timed lap, the 10-lap
total, and wall contacts. A policy worth shipping is clean at every delay.
Note: without `--speed-table`, `evaluate.py` uses the *untuned* table built
from [`DEFAULTS`](../race_ws/src/team_driver/team_driver/pure_pursuit.py#L31), not `models/speed_table.npz`.

To put a new policy in the car:

```sh
cp training/runs/<run>/lookahead.npz race_ws/src/team_driver/models/
#   inside the container:
cd /hackathon/race_ws && colcon build --symlink-install && source install/local_setup.bash
#   on the host:
./scripts/evaluate.sh --runs 3 --headless
```

At startup the driver logs `lookahead policy loaded from ...: L = ...`
and `speed table tuned, from ..., scaled by 1.0: ...`. With no lookahead policy it
logs `no trained lookahead policy: lookahead_gain * speed`. It also warns `pure pursuit
settings changed since the policy was trained: [...]` if [`DEFAULTS`](../race_ws/src/team_driver/team_driver/pure_pursuit.py#L31) differ
from those saved in the file. The shipped files were saved before
`table_scale` and `speed_lead` existed, so expect that warning to list
those two; both are at their neutral values (1.0 and 0.0), so it is harmless.

---

## 7. Method 1: the speed table

`--method speed_table`. **Shipped.** The policy *is* a table: 380 numbers,
one target speed per racing-line point (points are 0.2 m apart, 75.9 m per
lap). Each step the car reads the speed at its nearest point.

**Where it starts.** `speed_table()` in [raceline.py](../race_ws/src/team_driver/team_driver/raceline.py)
builds it from two rules, with no top speed:

1. Corners: no faster than `sqrt(mu · g / |curvature|)`. With `mu` 1.0489, a
   bend of radius 1 m allows 3.2 m/s.
2. Neighbours: between points `d` metres apart, speed changes by at most
   `2 · a_brake · d / (v + v_next)`, i.e. `v_next² ≤ v² + 2·a_brake·d`, the
   physics of braking/accelerating at `a_brake` (6.05 m/s²).
   ([`limit_speed_changes()`](../race_ws/src/team_driver/team_driver/raceline.py#L275) in [raceline.py](../race_ws/src/team_driver/team_driver/raceline.py))

With today's [`DEFAULTS`](../race_ws/src/team_driver/team_driver/pure_pursuit.py#L31) that table laps in 13.51 s if driven exactly.

**How PPO tunes 380 numbers.** Changing all 380 at once gives a signal too
noisy to learn from, so each update:

- Picks a random stretch of [`WINDOW`](train.py#L146) = 60 consecutive points (12 m).
- Only cars inside the stretch explore. Each gets one random speed offset
  (spread 0.3 m/s) per [`NOISE_SEGMENT`](train.py#L147) = 10 points (2 m), **held** for the
  whole update. Why held: the simulated car eases towards its speed command,
  so a change lasting one 0.01 s step barely does anything; one held for 2 m does.
- Cars elsewhere drive the table as it is. About 1 car in 6 is inside the
  window (`inside` ≈ 0.12–0.19 in the logs).
- Only the stretch's samples count in the policy loss
  ([`speed_table_loss()`](../race_ws/src/team_driver/team_driver/policies.py#L333) in [policies.py](../race_ws/src/team_driver/team_driver/policies.py),
  PPO's loss with a `weight`), and only its 60 speeds may change. The
  critic (7 inputs: the 5 lookahead inputs + where on the lap, as sin/cos)
  learns from every car.
- After each update the table is clipped to [`SPEED_RANGE`](train.py#L148) (1–15 m/s) and
  lowered wherever neighbours break rule 2 (`train.keep_within_rule`), so
  the tuned table obeys the same physics it started with.
- Reward: the simulator's own, metres of progress per step, −10 for a wall.
  Steering is plain pure pursuit with the hand-made lookahead.
- The exploration spread is **fixed** at 0.3 m/s (`stop_gradient`). In the
  first run (`20261012-164457`) it was learned, and PPO shrank it to 0.05
  within one run, so the table stopped exploring.

Because each stretch starts at a random point, the 2 m pieces line up
differently each update, and over the run every point is tuned on its own.

**Results** (table lap = lap time if the table were driven exactly):

| Run | `--max-delay` | Started from | Table lap at end |
| --- | --- | --- | --- |
| `20261012-164457` | 3 | built-in table | 12.60 s (spread learned, shrank to 0.05) |
| `20261013-014155` | 8 | built-in table (13.51 s) | 12.36 s |
| `20261013-021003` | 12 | `--init` 014155 | **12.28 s** (shipped) |

40M steps take 8–13 minutes (152 updates, 56,000–93,000 steps/s).

```sh
~/venvs/rl/bin/python training/speed_table.py                       # look at the starting table
~/venvs/rl/bin/python training/train.py --method speed_table --smoke
~/venvs/rl/bin/python training/train.py --method speed_table        # 40M steps, 8-13 min
~/venvs/rl/bin/python training/evaluate.py --speed-table training/runs/<run>/speed_table.npz
cp training/runs/<run>/speed_table.npz race_ws/src/team_driver/models/   # then rebuild (section 6)
```

The driver refuses a table tuned on a different racing line (different
`wall_gap` or map) and logs why.

---

## 8. Method 2: dynamic lookahead

`--method lookahead` (the default). **Shipped**, but see section 1: it
learned almost nothing beyond the hand rule.

After Elgouhary & El-Wakeel 2026, *Dynamic Lookahead Distance via
Reinforcement Learning-Based Pure Pursuit for Autonomous Racing*
(arXiv 2603.28625). **Lookahead distance** L = how far ahead on the line
pure pursuit aims. Short L follows the line tightly but wobbles at speed;
long L is smooth but cuts corners. The policy picks L every step:

```
L = b + w_v·v + w_0·|k0| + w_1·|k1| + w_2·|k2| + w_d·dk        clipped to 0.35 .. 4.0 m
```

| Input | Meaning |
| --- | --- |
| `v` | speed [m/s] |
| `\|k0\|`, `\|k1\|`, `\|k2\|` | curvature (how sharply the line bends, 1/radius, in 1/m) at the nearest point, 1 m and 3 m ahead |
| `dk` | `\|k1\| − \|k0\|`: is the bend tightening (+) or opening (−)? |

It is **linear** (5 weights + bias), so after training you can read off what
it learned. Training starts from `L = 0.25·v` ([`START_WEIGHTS`](../race_ws/src/team_driver/team_driver/policies.py#L87) in [policies.py](../race_ws/src/team_driver/team_driver/policies.py)),
the hand rule. During training L gets ±0.3 m of Gaussian noise; the spread
is learned too.

**Reward** (`drive_lookahead()` in [train.py](train.py)), per 0.01 s step:

```
reward =  progress (m along the centreline this step)          ≈ 0.06 at 6 m/s
        − 10        if the car touched a wall (episode ends)
        + 0.002 · v                    (--w-speed)              ≈ 0.012 at 6 m/s
        − 0.01  · |L − previous L|     (--w-jerk)   a jumpy L makes jumpy steering
        − 0.05  for every step stalled (--w-stall)  < 0.05 m/s for more than 0.5 s
```

The last three are three of the paper's eq. 13 terms; the paper does not
publish its weights, so these are ours.

**Where this differs from the paper:** the policy is linear rather than a
neural network; the reward leaves out the paper's ideal-lookahead and
curvature terms; the paper does not give its curvature horizons, so 1 m and
3 m are ours; L is not smoothed before use; and every episode gets a random
reaction delay (section 10). The PPO settings are the paper's (section III-J):
discount 0.99, GAE λ 0.98, 5 epochs, clip 0.2, target KL 0.015, entropy 0.02,
value weight 0.6, learning rate 2.3927e-4 falling linearly to 0
([`SETTINGS`](train.py#L152) in [train.py](train.py)).

Training prints the equation every 5 updates and the mean L on straight
(`L straight`) and curved (`L curve`) track. The paper finds L grows on
straights and shrinks in bends; that is mostly just `w_v·v` (you go faster
on straights).

A 20M-step run takes 4–7 minutes (76 updates, ~57,000–91,000 steps/s).

---

## 9. How PPO is written here, step by step

One update (`update()` in [train.py](train.py)), with the default lookahead numbers:

1. **Drive** 128 steps × 2048 cars = 262,144 samples. Each sample keeps what
   the policy saw, what it picked (noise included), how likely that pick
   was (`log_prob`), the critic's guess, and the reward.
2. **Advantages with GAE** ([`advantages_and_returns()`](train.py#L359) in [train.py](train.py)). For
   each step, `delta = reward + gamma · V(next) − V(now)`: one step of
   "better than expected?". GAE adds up later deltas, each shrunk by
   `gamma · lambda` per step. A crash gets no future value; hitting the 30 s
   time limit does (the car would have kept driving). Advantages are then
   rescaled to mean 0, spread 1.
3. **Learn.** 5 passes (epochs) over the samples, each split into 32
   shuffled minibatches of 8,192: up to 160 gradient steps.
   - Loss = PPO policy loss ([`ppo_policy_loss()`](../race_ws/src/team_driver/team_driver/policies.py#L172) in [policies.py](../race_ws/src/team_driver/team_driver/policies.py))
     + 0.6 × the critic's squared error − entropy bonus.
   - Gradients longer than 0.5 are shrunk ([`clip_by_global_norm()`](train.py#L502) in [train.py](train.py)),
     so one bad minibatch cannot throw the weights far.
   - **Adam** ([`adam_step()`](train.py#L456) in [train.py](train.py)) takes the step: it keeps running
     averages of the gradient and its square, so every weight moves about
     `lr` per step whether its gradient is huge or tiny.
   - The learning rate falls in a straight line to 0 over the run.
   - KL early stop: if KL > 1.5 × 0.015, the rest of this update is skipped
     (`one_minibatch()` inside `update()` in [train.py](train.py)).
4. Log, print every 5 updates, save every 20 updates and at the end
   (so a stopped run still leaves usable weights).

---

## 10. Reaction delay, and how the simulator matches the race

**Matches:** car model (ST single-track with tyre slip, RK4 integrator,
0.01 s steps), f1tenth vehicle parameters, speed + steering-angle control,
the icra26 map and centreline, LiDAR mounting and beam angles (noise-free),
odometry at the rear axle. Checked on 2026-10-11: the GPU simulator and the
race bridge's own engine put the car into the infield hairpin at the same
point, (5.09, 9.64).

**Differs on purpose:** only the LiDAR beams in the driver's ±10° "ahead"
cone are simulated (the driver uses no others; all 819 is ~4× slower), and
training episodes start at a random point, standing still.

**Reaction delay is the big unknown.** In ROS, a command reaches the physics
a few hundredths of a second after the scan it answers, and how long varies
with the computer. Measured on 2026-10-11 with the *old* speed rule (before
the speed table), pure pursuit was on the edge in the hairpin near (5.1, 9.7):

| Delay | Pure pursuit alone, race engine without ROS |
| --- | --- |
| 0 or 1 step | crashes in the hairpin and stays stuck |
| 2 or 3 steps | laps in about 17.2 s, with a few wall contacts |
| real ROS run (2026-10-11) | 16.68 s, no contacts |

So `sim.delay_commands` ([sim.py](sim.py)) keeps a queue of each car's
last commands and applies the one from `delay` steps ago. Every episode gets
a random delay from 0 to `--max-delay` (default [`MAX_DELAY`](sim.py#L83) = 3 steps). The
shipped runs used larger ranges (speed table 8 then 12, lookahead 10) to be
safe against a slower reaction. `evaluate.py` races one car per delay and
reports each separately. Not modelled: the delay changing within a lap, and
the driver occasionally skipping a scan.

---

## 11. Reference: every script and flag

### `train.py`

Defaults marked "per method" come from [`SETTINGS`](train.py#L152) in [train.py](train.py)
(lookahead / speed_table).

| Flag | Default | What it does |
| --- | --- | --- |
| `--method` | `lookahead` | `lookahead` or `speed_table` |
| `--steps` | per method: 20M / 40M | total simulated steps. Updates = steps ÷ (envs × rollout). |
| `--envs` | 2048 | cars driving in parallel |
| `--rollout` | 128 | steps each car drives between updates |
| `--epochs` | 5 / 5 | passes over each batch |
| `--minibatches` | 32 / 32 | pieces each pass is cut into. envs × rollout must divide by it. |
| `--hidden` | 128 | neurons per hidden layer of the critic |
| `--lr` | 2.3927e-4 / 1e-3 | learning rate |
| `--lr-decay` / `--no-lr-decay` | on / on | fall linearly to 0 over the run |
| `--gamma` | 0.99 / 0.99 | discount per step |
| `--gae-lambda` | 0.98 / 0.95 | GAE's blend |
| `--clip` | 0.2 | PPO clip |
| `--entropy` | 0.02 / 0 | entropy bonus (does nothing for speed_table: its spread is fixed) |
| `--value-coef` | 0.6 / 0.5 | weight of the critic's error in the loss |
| `--target-kl` | 0.015 / 0.015 | KL early stop threshold |
| `--episode-steps` | 3000 | time limit per episode (30 s) |
| `--w-speed`, `--w-jerk`, `--w-stall` | 0.002, 0.01, 0.05 | lookahead reward terms (section 8) |
| `--init FILE` | none | continue from a saved `.npz` of the same method (policy, critic and noise; Adam and the lr schedule restart). This is how runs are chained, e.g. 014155 → 021003. For `speed_table` it is also the only way to start from a tuned table. |
| `--speed-table FILE` | built from [`DEFAULTS`](../race_ws/src/team_driver/team_driver/pure_pursuit.py#L31) | lookahead only: drive by this tuned table (must match the racing line). Ignored by `--method speed_table`. |
| `--max-delay N` | 3 | each car's reaction delay is random in 0..N steps |
| `--seed` | 0 | random seed |
| `--device` | `gpu` | `gpu` or `cpu` |
| `--smoke` | off | tiny run: 64 cars × 64 steps, 10 updates (overrides steps/envs/rollout) |

### `evaluate.py`

| Flag | Default | What it does |
| --- | --- | --- |
| `--lookahead FILE` | none | let this policy choose L |
| `--speed-table FILE` | built from [`DEFAULTS`](../race_ws/src/team_driver/team_driver/pure_pursuit.py#L31) | drive by this tuned table, with or without `--lookahead` |
| `--max-delay N` | 3 | race one car per delay 0..N |
| `--laps N` | 11 | laps to drive, the first is a warm-up (like the referee) |
| `--max-time S` | 300 | give up after this many *simulated* seconds |
| `--device` | `cpu` | cpu is faster for a handful of cars |

### `speed_table.py`

`--mu` and `--a-brake` (default from [`DEFAULTS`](../race_ws/src/team_driver/team_driver/pure_pursuit.py#L31)): print and draw the
untuned table those values give. Writes `speed_table.csv` and `speed_table.png`.

### `pure_pursuit.DEFAULTS` knobs that matter here

[`DEFAULTS`](../race_ws/src/team_driver/team_driver/pure_pursuit.py#L31) in [pure_pursuit.py](../race_ws/src/team_driver/team_driver/pure_pursuit.py) are
the race settings and also ROS parameters (`-p name:=value`).

| Knob | Value | Meaning |
| --- | --- | --- |
| `use_speed_table` | `True` | speed from the table; `False` = the old three limits (LiDAR braking, cornering, `v_max`), and no tuned table |
| `table_scale` | 1.0 | every table speed (tuned or not) is multiplied by this, in the driver, `train.py` (lookahead) and `evaluate.py`. Below 1 leaves room for a slow ROS reaction. |
| `speed_lead` | 0.0 (off) | **new, uncommitted.** Adds `k × (table speed 1 m ahead − table speed here)`, scaled by straightness (`1 − \|steering\| / max_steer`). The car's real speed lags its command, so this makes it brake slightly earlier before a corner and speed up slightly earlier after one; it fades to nothing at full lock. |
| `mu`, `a_brake` | 1.0489, 6.05 | the starting table's cornering grip and braking |
| `lookahead_gain`, `min_lookahead` | 0.25, 0.5 | the hand lookahead rule `L = max(0.5, 0.25·v)` |

Constants in [train.py](train.py): [`COLLISION_PENALTY`](train.py#L140) 10,
[`MAX_GRAD_NORM`](train.py#L141) 0.5, [`STALL_SPEED`](train.py#L142) 0.05 m/s,
[`STALL_STEPS`](train.py#L143) 50, [`WINDOW`](train.py#L146) 60, [`NOISE_SEGMENT`](train.py#L147) 10, [`SPEED_RANGE`](train.py#L148) (1, 15).

---

## 12. Reading the training output

Every update writes one row of `runs/<run>/log.csv`
(the `metrics` dict at the end of `update()` in [train.py](train.py)); the screen shows some of them every 5 updates.

| Column | Meaning | Want |
| --- | --- | --- |
| `update` | update number | |
| `steps_M` | simulated steps so far, in millions | |
| `steps_per_s` | speed of training (0 on the first row, which includes compiling) | |
| `track_speed` | average metres of track per second, restarts from standstill included (screen: `speed`) | up |
| `crashes_per_lap` | wall contacts per 78.5 m driven (one centreline lap) | down, towards 0 |
| `noise_std` | exploration spread (m for L, m/s for speeds) (screen: `noise`) | slowly down |
| `value_loss` | critic's squared error | down, then flat |
| `kl` | how far the policy moved this update | below ~0.02 |
| `stopped_early` | 1 if the KL early stop fired | mostly 0 |
| **lookahead** | | |
| `L_straight`, `L_curve` | mean L where the sharpest of `\|k0\|,\|k1\|,\|k2\|` is < 0.2 /m, and > 0.8 /m | straight > curve |
| `L_mean`, `L_change` | mean L, mean step-to-step change of L | |
| `stalled` | share of steps stalled | 0 |
| `w_v`, `w_\|k0\|`, `w_\|k1\|`, `w_\|k2\|`, `w_dk`, `b` | the policy's weights | |
| **speed_table** | | |
| `window_start` | first point of the stretch tuned this update | |
| `inside` | share of samples inside the stretch (≈ 0.16) | |
| `window_change`, `window_shift` | mean size of the change to the stretch's speeds, and its signed mean (m/s) | small |
| `table_lap` | lap time of the whole table if driven exactly | down |

---

## 13. Reproduce the shipped models

From each run's `args.json` (everything not shown is the default):

```sh
# 1. Speed table, first run (8-13 min).                 -> was 20261013-014155, table lap 12.36 s
~/venvs/rl/bin/python training/train.py --method speed_table --max-delay 8

# 2. Continue it with a wider delay range.             -> was 20261013-021003, table lap 12.28 s
~/venvs/rl/bin/python training/train.py --method speed_table --max-delay 12 \
    --init training/runs/<run from step 1>/speed_table.npz
cp training/runs/<run from step 2>/speed_table.npz race_ws/src/team_driver/models/

# 3. Lookahead on top of the tuned table (about 2 min). -> was 20261013-030026
~/venvs/rl/bin/python training/train.py --steps 2e6 --envs 512 --max-delay 10 \
    --speed-table race_ws/src/team_driver/models/speed_table.npz
cp training/runs/<run from step 3>/lookahead.npz race_ws/src/team_driver/models/

# 4. Check, then rebuild and race (section 6).
~/venvs/rl/bin/python training/evaluate.py --max-delay 12 \
    --lookahead race_ws/src/team_driver/models/lookahead.npz \
    --speed-table race_ws/src/team_driver/models/speed_table.npz
```

Expect close, not identical, numbers: the seed is fixed (0) but GPU maths is
not bit-for-bit repeatable, and [`DEFAULTS`](../race_ws/src/team_driver/team_driver/pure_pursuit.py#L31) have changed slightly since (the
shipped lookahead file records a since-removed `table_v_max` setting, and
neither file knows `table_scale` or `speed_lead`).

### All runs in `runs/`

| Run | Method | Notes |
| --- | --- | --- |
| `20261012-012215` | lookahead | `--smoke` |
| `20261012-012505` | lookahead | 20M, delay 0–3, untuned table → `w_v` 0.33 |
| `20261012-014600` | lookahead | 20M, same → `w_v` 0.32, `w_\|k2\|` 0.10 |
| `20261012-015943` | lookahead | 20M, `--init` 014600 |
| `20261012-161751` | lookahead | 20M, `--init` 015943; 1.4–1.9 crashes/lap |
| `20261012-164457` | speed_table | 40M, delay 0–3, learned spread → 12.60 s |
| `20261013-014155` | speed_table | 40M, delay 0–8 → 12.36 s |
| `20261013-021003` | speed_table | 40M, delay 0–12, `--init` 014155 → 12.28 s, **shipped** |
| `20261013-030026` | lookahead | 2M, 512 cars, delay 0–10, tuned table, **shipped** |

---

## 14. `test_parity.py`: the four checks

```sh
~/venvs/rl/bin/python training/test_parity.py
```

If any check fails, a policy trained here would race a different controller
from the one it learned with. Each prints the largest difference it found.

1. **Moving the code changed nothing.** It fetches the original [`plan()`](../race_ws/src/team_driver/team_driver/driver.py#L533)
   from `driver.py` at commit `0c2a770` (with `git show`) and compares it
   with [`pure_pursuit()`](../race_ws/src/team_driver/team_driver/pure_pursuit.py#L81) on 1000 random situations: must agree to 1e-9.
   **Needs git history**: it fails in a copy without `.git` or a shallow
   clone that lacks `0c2a770`.
2. **numpy (driver) = JAX (training)** for pure pursuit's command, on 300
   random situations.
   Tolerance 1e-3, because JAX uses 32-bit floats and numpy 64-bit (expect ~1e-6).
3. **Dynamic lookahead:** numpy = JAX for its 5 inputs, L and pure pursuit
   aiming L ahead; passing the hand rule's own L changes nothing; the loss
   agrees; JAX's gradient matches a numerical one (nudge each weight by 1e-6);
   a saved policy loads back.
4. **Speed table:** it obeys both rules (neighbours, corners); pure pursuit
   reads it the same in numpy and JAX, with and without `speed_lead`; its
   loss agrees; its gradient matches a numerical one and lands only on points
   inside the window; a saved table loads back.

---

## 15. Where it can fail

- **Timing.** Anything where ROS differs from the simulator, the reaction
  delay above all. Mitigation: random delay in training, `evaluate.py` at
  every delay, `table_scale` below 1 as a safety margin.
- **A tuned table is tied to its racing line.** Change `wall_gap`, the map or
  the line code and the driver refuses the table (and falls back to the
  built-in one).
- **Retuning [`DEFAULTS`](../race_ws/src/team_driver/team_driver/pure_pursuit.py#L31) invalidates policies.** They learned with the old
  tuning; the driver warns. Retrain.
- **Weak reward signal** (the lookahead result): small policy changes barely
  move "metres of progress", so PPO can sit still. Longer runs, more cars or
  a sharper reward would be needed to learn more.
- **Learned exploration collapsing** (speed table run 1): PPO shrank the
  noise to nothing. Fixed by holding the spread constant.

---

## 16. Things to tune

- Speed table: [`WINDOW`](train.py#L146), [`NOISE_SEGMENT`](train.py#L147), [`SPEED_RANGE`](train.py#L148) in `train.py`;
  `mu`, `a_brake` in [`DEFAULTS`](../race_ws/src/team_driver/team_driver/pure_pursuit.py#L31) (the starting table); `--max-delay`.
- In the car without retraining: `table_scale` (slower, safer) and
  `speed_lead` (earlier braking/acceleration), e.g. `-p table_scale:=0.95`.
- Lookahead: `--w-speed`, `--w-jerk`, `--w-stall`, `--steps`, `--envs`.
- Everywhere: `--lr`, `--gamma`, `--entropy`; [`COLLISION_PENALTY`](train.py#L140) in `train.py`.
- If you change anything in `pure_pursuit.DEFAULTS`, retrain.

---

## 17. References and credits

Code we built on (credit both in `SUBMISSION.md`):

- `external/f1tenth_gym/examples/jax_ppo_training.py` (MIT licence): the
  simulator API usage, GAE and loss code our `train.py` follows.
- PureJaxRL, which that example follows: C. Lu et al., "Discovered Policy
  Optimisation", NeurIPS 2022, https://github.com/luchris429/purejaxrl

Papers:

- J. Schulman, F. Wolski, P. Dhariwal, A. Radford, O. Klimov, "Proximal Policy
  Optimization Algorithms", arXiv:1707.06347, 2017. (PPO)
- J. Schulman, P. Moritz, S. Levine, M. Jordan, P. Abbeel, "High-Dimensional
  Continuous Control Using Generalized Advantage Estimation", ICLR 2016,
  arXiv:1506.02438. (GAE)
- M. Elgouhary, A. El-Wakeel, "Dynamic Lookahead Distance via Reinforcement
  Learning-Based Pure Pursuit for Autonomous Racing", arXiv:2603.28625, 2026.
  (the lookahead method and its PPO settings)
- D. Kingma, J. Ba, "Adam: A Method for Stochastic Optimization", ICLR 2015,
  arXiv:1412.6980. (`adam_step`)
- The target-KL early stop follows Stable-Baselines3 (A. Raffin et al., JMLR 2021).
