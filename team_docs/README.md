# Our F1TENTH driver: a two-page overview

> For how to run the driver and what every setting does, see the
> [driver reference](driver_reference.md).

Our car laps the `icra26` track in about **13.7 s** on the official referee,
with no wall contacts. It follows a **racing line built from the map**, steers
with **pure pursuit**, and takes its speed from a **speed table tuned with
reinforcement learning (RL)**.

## 1. The idea in one paragraph

Pure pursuit is a simple, stable steering method, but on its own it cuts
corners and has no idea how fast to go. We fixed the first problem with a
smooth racing line, and the second with a table of target speeds that brakes
before corners. RL is only used to *tune* parts of a controller that already
worked, so a bad policy still drives, and every learned piece is small enough
to explain (one is a table of 380 numbers, the other is 6 numbers).

## 2. What happens when the car starts

The driver ([`driver.py`](../race_ws/src/team_driver/team_driver/driver.py))
does all the slow work once, before the car moves:

1. **Read the map.** Rows of cones are sealed into solid walls (rule 11), and
   every pixel gets its distance to the nearest wall.
2. **Build the racing line** ([`raceline.py`](../race_ws/src/team_driver/team_driver/raceline.py)).
   Start from the official centreline and slide each point sideways, a little
   at a time (20,000 times), so the line bends as little as possible, while
   the car's side stays at least 0.2 m from any wall. Result: 380 points,
   0.2 m apart, 75.9 m per lap.
3. **Build the speed table.** For each point, the fastest safe speed in the
   corner, √(μ·g / curvature). Then walk round the lap forwards and backwards
   so that neighbouring speeds never differ by more than braking at
   6.05 m/s² allows. If the trained table (`models/speed_table.npz`) was made
   for this same line, use that instead.
4. **Load the lookahead policy** (`models/lookahead.npz`), if it is there.

## 3. What happens on every LiDAR scan

1. Find the point of the racing line nearest the car (position from ground-truth
   odometry, which the rules allow).
2. **How far ahead to aim**, L: the learned policy gives L ≈ 0.25 × speed,
   between 0.35 and 4 m ([`policies.py`](../race_ws/src/team_driver/team_driver/policies.py)).
3. **Steering** ([`pure_pursuit.py`](../race_ws/src/team_driver/team_driver/pure_pursuit.py)):
   steer along the circle that passes through the car and the point L metres
   ahead, limited to ±0.42 rad.
4. **Speed**: read from the table at the nearest point.
5. Send the command on `/drive`. Every 200 scans the driver also logs how late
   it is reacting (see section 6).

## 4. The two learned parts

| | Speed table | Lookahead policy |
| --- | --- | --- |
| What it decides | the target speed at each of the 380 points | how far ahead pure pursuit aims |
| Size | 380 numbers | 5 weights + 1 bias |
| Starts from | the physics table of section 2 | our hand rule, L = 0.25 × speed |
| Training | PPO, 2 runs × 40 M simulated steps, ~10 min each on a GPU | PPO, 2 M steps, ~2 min |
| Result | ideal lap time 13.51 s → 12.28 s | L ≈ 0.249 × speed: it confirmed our hand rule |

**How the speed table is trained.** 2,048 simulated cars drive at once. Each
update picks a random 12 m stretch of the line; cars inside it try speeds a
little above or below the table (±0.3 m/s, held for 2 m at a time, because
the car cannot react to a change that lasts one 0.01 s step). The reward is
metres of progress, minus 10 for touching a wall. PPO, the RL method we use,
then makes the speeds that went better than expected more likely. After every
update the table is pulled back within the braking rule.

## 5. How we know it works

- **Training simulator** (`training/sim.py`): f1tenth_gym's GPU simulator, set
  up to match the race (car model, 0.01 s steps, LiDAR angles, odometry).
- **`training/test_parity.py`**: checks that the driver (numpy) and the
  training code (JAX) compute exactly the same thing, so we race what we
  trained.
- **`training/evaluate.py`**: races the driver in the simulator at every
  reaction delay and counts wall contacts.
- **The official referee** (`./scripts/evaluate.sh`), which is what the judges
  run.

## 6. The biggest problem we hit: reaction delay

In ROS the simulator does not wait for our driver, so each command lands a few
0.01 s steps after the scan it answers. The slower car never noticed, but at
9 m/s the delay mattered: runs with the visualiser on were clean, while the
judges' headless mode (faster, so more delay) hit walls. We fixed it by:

- training the speed table with random delays of up to 12 steps;
- keeping only the newest LiDAR scan, so the driver never steers from an old
  one;
- logging the measured delay, so we can see it instead of guessing.

## 7. What we tried that did not work

- **A residual neural network** that corrected pure pursuit's steering and
  speed. Built and tested, never trained to a model good enough to race; it
  is kept in [`unused/`](../unused/README.md).
- **The untuned physics speed table**: fast (14.5 s) but it hit walls once the
  delay grew.
- **Learning the exploration noise**: it shrank to almost nothing mid-run, so
  we fixed it at 0.3 m/s.
- **A LiDAR emergency brake**: it made laps slower and was no safer.
- **`lookahead_gain` 0.4**: the car cut the infield hairpin.

## 8. Lap time progression

Constant speed ~27 s → three speed limits < 18 s → racing line 16.68 s →
tuned speed table ~13.7 s.

## 9. Limits

- The speed ignores obstacles: fine on an empty track, not in traffic.
- The tuned table belongs to one racing line; change the map or `wall_gap`
  and the driver falls back to the slower physics table.
- Delays longer than 12 steps are untested; `table_scale` (e.g. 0.95) slows
  the whole table down as a safety margin.

## 10. Where things are

| Path | What it is |
| --- | --- |
| `race_ws/src/team_driver/team_driver/` | the driver: `driver.py` (ROS node), `raceline.py`, `pure_pursuit.py`, `policies.py` |
| `race_ws/src/team_driver/models/` | the two trained files |
| `training/` | simulator, training, evaluation and tests ([README](../training/README.md)) |
| `unused/` | code we kept but do not use |
| [`SUBMISSION.md`](../SUBMISSION.md) | what we hand in |

Run the race: `./scripts/evaluate.sh --runs 3 --headless`.
Train: `~/venvs/rl/bin/python training/train.py --method speed_table`.
