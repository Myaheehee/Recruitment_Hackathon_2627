# Team DeepSpeed Team 2

> **TODO before submitting:** fill in every `TODO` (team name, members, results,
> machine). The results must come from your own `./scripts/evaluate.sh` runs:
> copy them, unedited, into `results/submitted/`
> ([§7.1](docs/07-submission.md#71-what-to-submit)).

## Members
| Name | Matric number | Email |
| --- | --- | --- |
| Hridyan Deshpande | --- | --- |
| Leonard | M2622241F | Lcheng016@e.ntu.edu.sg |

## Approach

Our driver follows a **racing line it builds from the map at startup**, steers
with **pure pursuit**, and takes its speed from a **per-point speed table tuned
with reinforcement learning**.

**At startup:**
- The driver reads the occupancy map and seals the cone rows into solid walls
  (rule 11). It then computes every pixel's distance to the nearest wall.
- It slides each centreline point sideways to minimise the line's total
  bending, never letting the car's side come within 0.2 m of a wall. This is
  projected gradient descent on the second differences of the points.
- It works out a target speed for each of the line's 380 points. The starting
  point is the cornering limit √(μg/κ) together with a braking rule enforced by
  forward and backward passes round the lap. PPO then tuned those 380 numbers
  in a GPU simulator.

**While racing,** on each LiDAR scan, a learned linear policy chooses pure
pursuit's lookahead distance from the speed and the curvature ahead. Pure
pursuit steers along the circle through the point that far ahead, and the
speed is read from the table.

**Why this design:** pure pursuit is simple, stable and explainable, but it
cuts corners and has no idea how fast to go. A smooth line fixes the first
problem, and a planned speed table that brakes before corners fixes the
second. We used RL only to *tune* parts of a controller that already worked.
A bad policy therefore still drives, and every learned piece is small enough to
explain.

**What we tried that did not work:**
- A residual neural network that corrects pure pursuit's steering and speed.
  We built and tested it, but never trained it to a shippable model, so it was
  moved out of the driver into `unused/residual/` (kept for reference, not used
  by the driver or training).
- Letting PPO learn the speed table's exploration noise: it collapsed to almost
  nothing mid-run.
- Exploration noise lasting only one 0.01 s step: it barely changed the car's
  speed, so there was nothing to learn from.
- `lookahead_gain` 0.4: it hit the infield hairpin.

We are also open about one result: the learned lookahead policy converged to
almost exactly our hand rule (L ≈ 0.249 × speed). The lap-time gain came from
the racing line and the tuned speed table.

Lap time progression on the official referee: constant speed ~27 s → three
speed limits <18 s → racing line 16.68 s → tuned table + learned lookahead
~13.7 s.

A two-page overview of the whole project is in
[team_docs/](team_docs/README.md), next to the
[driver reference](team_docs/driver_reference.md).

## How it uses the inputs
- **LiDAR (`/scan`):** cleaned (NaN → 0, ∞ → 8 m), and the shortest beam
  within ±10° ahead becomes "free space ahead". That value only sets the speed
  in the fallback mode without a speed table. With the speed table (the
  default), the LiDAR does not change the command.
- **Odometry (`/ego_racecar/odom`):** **ground truth**, as rule 31 permits.
  Position (rear axle), yaw from the quaternion, speed. We built no
  localisation of our own.
- **Map:** `maps/icra26.yaml`/`.png` and `maps/icra26_centerline.csv`, read at
  startup to build the clearance map, the racing line and the physics speed
  table. Nothing about the line is typed in by hand (rule 32).

## Results on our own machine
The runs committed in `results/submitted/`, summarised.

| Best lap | 10-lap total | Collisions | Runs attempted |
| --- | --- | --- | --- |
| 13.790 | 137.900 | 1 | 5 |

### The machine that produced them
| CPU | GPU | RAM | OS | Typical real-time factor |
| --- | --- | --- | --- | --- |
| TODO (e.g. AMD Ryzen 7 260) | TODO | TODO (e.g. 16 GB) | TODO (e.g. Windows 11 + WSL2) | TODO |

**Things that might explain a gap between your numbers and ours:**
- The speed table was trained to tolerate reaction delays of 0–12 simulator
  steps (0–0.12 s). A much slower machine could exceed that.
- The driver logs its measured delay every 200 scans ("reaction delay over 200
  scans …"), so the judges' logs will show whether that happened.
- Every startup prints a harmless warning: "pure pursuit settings changed since
  the policy was trained: ['table_scale', 'speed_lead']". Those two settings
  were added after the model files were saved, and both are at neutral values.

## Anything precomputed

Nothing is precomputed from the map: the racing line and the physics speed
table are rebuilt from the map on every start
([raceline.py](race_ws/src/team_driver/team_driver/raceline.py)). Two trained
files are shipped in `race_ws/src/team_driver/models/`:

**`speed_table.npz`: the PPO-tuned speed table (380 target speeds plus the line they belong to).**
- **Environment:** f1tenth_gym's JAX simulator, configured to match the race:
  single-track model with tyre slip, 0.01 s steps, the icra26 map, the race's
  LiDAR beam angles. It ran 2048 cars in parallel, each starting at a random
  track point with a random reaction delay.
- **Policy:** the table itself. Each update tunes one random 12 m window. Cars
  inside it try random speed offsets held over 2 m pieces, with a fixed spread
  of 0.3 m/s.
- **Training:** PPO, hand-written in JAX. After each update the table is kept
  within 1–15 m/s and within the braking rule a = 6.05 m/s².
- **Reward:** metres of progress along the centreline per step, minus 10 per
  wall contact (which also ends that car's episode).
- **Steps and time:** two runs of 40M simulated steps each, about 10 minutes
  each on a GPU. The second (`training/runs/20261013-021003`) continued from
  the first, and widened the random delay from 0–8 to 0–12 steps.
- **How we decided it was good:** the "table lap" (the lap time if the table
  were driven exactly) fell from 13.51 s (untuned) to 12.36 s, then 12.28 s.
  It stayed clean across delays in `training/evaluate.py`, and the official
  referee lap fell to ~13.7 s.
- **Safety check:** the driver uses this table only if its stored line matches
  the line built at startup. Otherwise it falls back to the physics table.

**`lookahead.npz`: a linear lookahead policy, L = b + w·[v, |κ0|, |κ1|, |κ2|, |κ1|−|κ0|], clipped to 0.35–4 m.**
- **Source:** after Elgouhary & El-Wakeel 2026 (arXiv:2603.28625), with the
  paper's PPO settings and reward terms.
- **Reward:** progress, plus a speed bonus, minus costs for a jumpy L and for
  standing still.
- **Steps and time:** 2M steps with 512 cars and delays 0–10, about 2 minutes
  (run `20261013-030026`), trained on top of the tuned table.
- **Result:** L ≈ 0.0035 + 0.249·v, with negligible curvature weights. It
  confirms rather than improves our hand rule of 0.25·v.

**How to regenerate both:** see
[training/README.md](training/README.md#13-reproduce-the-shipped-models). In
short:

```sh
training/setup_venv.sh
~/venvs/rl/bin/python training/train.py --method speed_table --max-delay 8
~/venvs/rl/bin/python training/train.py --method speed_table --max-delay 12 --init training/runs/<first run>/speed_table.npz
~/venvs/rl/bin/python training/train.py --method lookahead --steps 2e6 --envs 512 --max-delay 10 --speed-table <tuned table>
```

Training details for both are in
[training/README.md](training/README.md).

## Dependencies we added
None. `requirements.txt` and `apt-packages.txt` are unchanged.

Our driver imports `numpy`, `scipy` (`ndimage`), `Pillow` and `PyYAML`. Rule 33
lists all four as already installed in the image, so there was nothing to
declare.

The training code (not used during judging) has its own
`training/requirements.txt`, installed into a separate virtual environment:
the pinned f1tenth_gym submodule and `jax[cuda13]`. `optax` is also listed
there but is unused.

| Package | pip or apt | What we use it for |
| --- | --- | --- |
| none | — | — |

## Third-party code and references
- **Pure pursuit:** R. C. Coulter, *Implementation of the Pure Pursuit Path
  Tracking Algorithm*, CMU-RI-TR-92-01, 1992.
- **Learned lookahead:** Elgouhary & El-Wakeel, arXiv:2603.28625, 2026
  (linear lookahead policy, reward terms, PPO settings).
- **PPO:** J. Schulman et al., *Proximal Policy Optimization Algorithms*,
  arXiv:1707.06347, 2017.
- **GAE:** J. Schulman et al., *High-Dimensional Continuous Control Using
  Generalized Advantage Estimation*, arXiv:1506.02438, 2016.
- **Training code structure:** our advantage and loss code follows
  `external/f1tenth_gym/examples/jax_ppo_training.py` (MIT licence), which
  follows PureJaxRL. The simulator is f1tenth_gym's `JaxSimulator` (pinned
  submodule).
- **Image operations:** `scipy.ndimage` (`binary_closing`,
  `distance_transform_edt`, `map_coordinates`).
- **AI-written code:** much of the code was written with Claude Code
  (Anthropic). See [AI assistance](#ai-assistance) for which parts.

## AI assistance
We used **Claude Code** (Anthropic's coding agent, run in VS Code and the
terminal) throughout the project, as rule 36 permits. What it did, by part:

| Part | Files | What Claude Code did |
| --- | --- | --- |
| Racing line | `raceline.py` | Implemented the clearance map, the cone sealing and the minimum-bending line optimiser to our design, and checked the geometry in the container without starting the simulator. |
| Pure pursuit and speed | `pure_pursuit.py`, `driver.py` | Implemented and refactored pure pursuit, the physics speed table (cornering limit plus braking passes) and the ROS node around them. |
| Learned policies | `policies.py` | Wrote the loaders for the tuned speed table and the lookahead policy, the line-match safety check and the fallback to the physics table. It also built the residual network we later dropped (`unused/residual/`). |
| Training | `training/train.py`, `sim.py`, `evaluate.py`, `speed_table.py` | Wrote the JAX PPO code and the simulator wrapper that matches the race, and helped run and read the tuning experiments. |
| Tests | `training/test_parity.py` | Wrote the tests that check the training code (JAX) and the driver (numpy) compute the same steering, lookahead and table speed, and that the speed table keeps to its braking and cornering limits. |
| Documentation | docstrings, `team_docs/`, `training/README.md`, this file | Drafted the explanations of the code and maths, and `team_docs/check_links.py`, which keeps the doc links pointing at the right lines. |
| Debugging | — | Helped track down build and environment problems (e.g. `verify_judging_env.sh` failing inside the container when run from a git worktree). |

We made the design decisions (racing line plus pure pursuit, using RL only to
tune parts of a controller that already worked, dropping the residual
network), chose and ran the experiments, and judged the results on the
official referee. 

TODO: list any other tools (ChatGPT, Copilot, …) and what they were used for.

## Known issues
- **No obstacle braking with the speed table.** Speed ignores the LiDAR, which
  is fine on the empty race track. The fix would be
  `min(table speed, √(2·a_brake·(ahead − margin)))`, which the code already
  computes for its fallback mode.
- **The tuned table is tied to one racing line.** Changing the map, `wall_gap`
  or `raceline_iters` makes the driver fall back to the slower physics table
  until it is retrained.
- **Reaction delays beyond 12 steps are untested.** `table_scale` (e.g. 0.95)
  is the quick safety margin.
- **`config/driver_params.yaml` holds older values** (e.g.
  `lookahead_gain: 0.4`). It is only read by `ros2 launch`; the judged
  `ros2 run` uses the defaults in `pure_pursuit.py`.
- **Next steps with more time:**
  - obstacle-aware speed and steering
  - our own localisation (particle filter)
  - a minimum-time line instead of minimum bending
  - measuring whether the learned lookahead beats the hand rule at all
