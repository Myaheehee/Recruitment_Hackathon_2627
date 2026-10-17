# Interview preparation

Every question we expect, then an answer to each one at four depths. Start at
the top of an answer and keep going only as far as the judge pushes:

| Level | What it is | How long to say it |
| --- | --- | --- |
| **In one line** | The answer itself. Always say this first. | 5 seconds |
| **Explain** | The answer with its reason, in plain words. | 30–60 seconds |
| **Go deeper** | The maths, the code, the numbers. Only if they ask "how exactly?" | as needed |
| **If they push** | The follow-ups judges tend to chain on, with short answers. | — |

Three rules for every answer (from [docs/07-submission.md §7.4](../docs/07-submission.md#74-the-interview), which lists what they ask):

1. **Answer, then reason, then evidence.** "We do X. Because Y. We know it works because Z (a number)."
2. **Be honest about weak results.** They will find them anyway. "The learned lookahead ended up equal to our hand rule" is a strong answer, not a weak one.
3. **Point at the code.** "That's `limit_speed_changes()` in `raceline.py`." Any member may be asked about any part.

Numbers, the "what if we double it" table and the list of failures are at the end, in [Part C](#part-c-quick-reference).

---

## Part A: the questions

**1. The big picture**
- [1.1 Walk us through your driver.](#11-walk-us-through-your-driver)
- [1.2 Why this approach and not gap-following, MPC or end-to-end RL?](#12-why-this-approach-and-not-gap-following-mpc-or-end-to-end-rl)
- [1.3 What does the car sense, and what does it control?](#13-what-does-the-car-sense-and-what-does-it-control)
- [1.4 What is the most interesting thing you did?](#14-what-is-the-most-interesting-thing-you-did)
- [1.5 Who wrote the code? How did you use AI?](#15-who-wrote-the-code-how-did-you-use-ai)

**2. The racing line**
- [2.1 How do you build the racing line?](#21-how-do-you-build-the-racing-line)
- [2.2 What is curvature, and how do you compute it?](#22-what-is-curvature-and-how-do-you-compute-it)
- [2.3 Why minimise bending and not lap time?](#23-why-minimise-bending-and-not-lap-time)
- [2.4 How do you make sure the line never hits a wall?](#24-how-do-you-make-sure-the-line-never-hits-a-wall)
- [2.5 What about the gaps between the cones?](#25-what-about-the-gaps-between-the-cones)
- [2.6 Why build it at startup instead of shipping a file?](#26-why-build-it-at-startup-instead-of-shipping-a-file)

**3. Steering: pure pursuit**
- [3.1 How does pure pursuit work?](#31-how-does-pure-pursuit-work)
- [3.2 Derive the steering angle.](#32-derive-the-steering-angle)
- [3.3 How do you choose the lookahead distance?](#33-how-do-you-choose-the-lookahead-distance)
- [3.4 Why measure from the rear axle?](#34-why-measure-from-the-rear-axle)

**4. Speed**
- [4.1 How does the car decide how fast to go?](#41-how-does-the-car-decide-how-fast-to-go)
- [4.2 Explain the two rules of the speed table.](#42-explain-the-two-rules-of-the-speed-table)
- [4.3 Where do μ = 1.0489 and a_brake = 6.05 come from?](#43-where-do-μ--10489-and-a_brake--605-come-from)
- [4.4 What are table_scale and speed_lead?](#44-what-are-table_scale-and-speed_lead)

**5. Reinforcement learning basics**
- [5.1 What is reinforcement learning, in our project's terms?](#51-what-is-reinforcement-learning-in-our-projects-terms)
- [5.2 What is PPO, and why clip the ratio?](#52-what-is-ppo-and-why-clip-the-ratio)
- [5.3 What does the critic do? What is an advantage? What is GAE?](#53-what-does-the-critic-do-what-is-an-advantage-what-is-gae)
- [5.4 Why add random noise to the actions?](#54-why-add-random-noise-to-the-actions)
- [5.5 What are Adam, the learning rate and the KL early stop for?](#55-what-are-adam-the-learning-rate-and-the-kl-early-stop-for)
- [5.6 Why JAX, and why 2,048 cars at once?](#56-why-jax-and-why-2048-cars-at-once)

**6. How we trained our two learned parts**
- [6.1 How did you train the speed table?](#61-how-did-you-train-the-speed-table)
- [6.2 Is tuning a table really reinforcement learning?](#62-is-tuning-a-table-really-reinforcement-learning)
- [6.3 Why PPO and not a simpler search?](#63-why-ppo-and-not-a-simpler-search)
- [6.4 What is the reward? Could the car exploit it?](#64-what-is-the-reward-could-the-car-exploit-it)
- [6.5 How did you train the lookahead policy? Is it useless?](#65-how-did-you-train-the-lookahead-policy-is-it-useless)
- [6.6 How much of your improvement is reinforcement learning?](#66-how-much-of-your-improvement-is-reinforcement-learning)
- [6.7 How did you decide a model was good enough to ship?](#67-how-did-you-decide-a-model-was-good-enough-to-ship)

**7. Simulation, testing and the real race**
- [7.1 Why trust the training simulator?](#71-why-trust-the-training-simulator)
- [7.2 Why did the car crash headless but not with the GUI?](#72-why-did-the-car-crash-headless-but-not-with-the-gui)
- [7.3 What do your parity tests check?](#73-what-do-your-parity-tests-check)
- [7.4 The judging machine is different. What changes?](#74-the-judging-machine-is-different-what-changes)
- [7.5 Why not train in ROS directly?](#75-why-not-train-in-ros-directly)

**8. The ROS node and the engineering**
- [8.1 What happens at startup and on every scan?](#81-what-happens-at-startup-and-on-every-scan)
- [8.2 Why are the maths files free of ROS?](#82-why-are-the-maths-files-free-of-ros)
- [8.3 What if a model file is missing or wrong?](#83-what-if-a-model-file-is-missing-or-wrong)
- [8.4 Why a queue length of 1?](#84-why-a-queue-length-of-1)

**9. What-ifs and limits**
- [9.1 How would you handle a track twice as fast?](#91-how-would-you-handle-a-track-twice-as-fast)
- [9.2 What about an obstacle mid-corner?](#92-what-about-an-obstacle-mid-corner)
- [9.3 What happens on a different track?](#93-what-happens-on-a-different-track)
- [9.4 What if we double this parameter?](#94-what-if-we-double-this-parameter)
- [9.5 Where does it break?](#95-where-does-it-break)
- [9.6 What would you do with another week?](#96-what-would-you-do-with-another-week)

**10. Things that did not work**
- [10.1 Show us a change that didn't work, and how you knew.](#101-show-us-a-change-that-didnt-work-and-how-you-knew)
- [10.2 What happened to the residual network?](#102-what-happened-to-the-residual-network)
- [10.3 Tell us about a bug you found.](#103-tell-us-about-a-bug-you-found)

---

## Part B: the answers

### 1. The big picture

#### 1.1 Walk us through your driver.

**In one line:** We build a smooth racing line from the map at startup, steer
along it with pure pursuit, and take our speed from a table of 380 target
speeds that we tuned with reinforcement learning; it laps in about 13.7 s.

**Explain:** When the driver starts, it reads the map, seals the cone rows
into walls, and slides the official centreline sideways until it bends as
little as possible while staying 0.2 m from every wall. It then works out a
safe speed for each of the line's 380 points from grip and braking physics,
and loads a version of that table which PPO tuned in a GPU simulator. While
racing, on every LiDAR scan, it finds the nearest point of the line, aims at a
point about 0.25 s ahead, steers along the circle through that point, and
sends the table's speed. Lap times went ~27 s → under 18 s → 16.68 s → ~13.7 s.

**Go deeper:**
- Startup: [`load_clearance_map()`](../race_ws/src/team_driver/team_driver/raceline.py#L146) → [`build_raceline()`](../race_ws/src/team_driver/team_driver/raceline.py#L30) → [`speed_table()`](../race_ws/src/team_driver/team_driver/raceline.py#L110) → [`load_speed_table()`](../race_ws/src/team_driver/team_driver/policies.py#L133) and [`load_lookahead()`](../race_ws/src/team_driver/team_driver/policies.py#L99).
- Every scan: [`Driver.plan()`](../race_ws/src/team_driver/team_driver/driver.py#L230) → [`lookahead()`](../race_ws/src/team_driver/team_driver/policies.py#L52) → [`pure_pursuit()`](../race_ws/src/team_driver/team_driver/pure_pursuit.py#L51) → publish on `/drive`.
- The whole tick costs about 0.1 ms of maths, about 1% of a 0.01 s physics step.

**If they push:**
- *"Which part made the biggest difference?"* The speed table: planning the
  speed for each point instead of reacting to the wall ahead (see [6.6](#66-how-much-of-your-improvement-is-reinforcement-learning)).
- *"Which part is learned?"* Two: the speed table's 380 numbers, and the
  lookahead policy's 6 numbers. Everything else is hand-built.

#### 1.2 Why this approach and not gap-following, MPC or end-to-end RL?

**In one line:** Pure pursuit on a planned line with planned speeds gives most
of the performance, stays explainable, and leaves RL to tune numbers inside a
controller that already works.

**Explain:**
- **Gap-following** reacts only to what the LiDAR sees now. It has no plan for
  the next corner, so it brakes late and drives slowly.
- **MPC** (optimise the next second with a car model, every step) is more
  powerful, but much harder to tune, explain and run 100 times a second in Python.
- **End-to-end RL** (a network from LiDAR straight to steering) needs huge
  amounts of training, can fail in strange ways, and is hard to explain.
- Our choice: hand-built parts we can reason about, plus RL only where it can
  safely help. A bad policy still drives, because pure pursuit is underneath.

**Go deeper:** Our learned parts are deliberately tiny: a table of 380 speeds
and a linear rule with 5 weights and a bias. Both can be read and checked by
eye. The speed table is also clipped to 1–15 m/s and to the braking rule after
every training update, so even training can't produce physically impossible
speeds.

**If they push:**
- *"So you avoided the hard stuff?"* We avoided *unexplainable* stuff. The
  hard parts are still there: a gradient-descent line optimiser, PPO written
  from scratch in JAX, and a sim-to-real delay problem we measured and fixed.

#### 1.3 What does the car sense, and what does it control?

**In one line:** It reads the LiDAR and ground-truth odometry, and it sends a
steering angle and a target speed.

**Explain:**
- **Odometry** (`/ego_racecar/odom`): exact position, heading and speed. The
  rules allow using it (rule 31), so we didn't build our own localisation.
- **LiDAR** (`/scan`, 819 beams over 270°): cleaned, then the shortest beam in
  a ±10° cone ahead gives the "free distance ahead". With the speed table on
  (the default), this value does **not** change the command; it only matters
  in the fallback speed mode.
- **Map**: the occupancy image and the official centreline, read once at startup.
- **Output** (`/drive`): steering angle (±0.4189 rad, 24°) and target speed. The
  simulator's own controller then accelerates or brakes towards that speed.

**If they push:**
- *"So the LiDAR is useless to you?"* For the empty race track, almost. That is
  a real limitation: see [9.2](#92-what-about-an-obstacle-mid-corner).

#### 1.4 What is the most interesting thing you did?

**In one line:** Pick one and know it deeply. The best candidate is the
reaction-delay story ([7.2](#72-why-did-the-car-crash-headless-but-not-with-the-gui)),
because it has a problem, a measurement and a fix.

**Explain:** Other good choices:
- the speed-table training trick: tune 60 points at a time, with noise held
  for 2 m so the car can actually feel it ([6.1](#61-how-did-you-train-the-speed-table));
- the racing line as gradient descent on "bending" ([2.1](#21-how-do-you-build-the-racing-line));
- the parity tests that prove training and racing compute the same thing ([7.3](#73-what-do-your-parity-tests-check)).

#### 1.5 Who wrote the code? How did you use AI?

**In one line:** We used Claude Code throughout, as the rules allow, and we
declare it in SUBMISSION.md; the design decisions, experiments and judgements
were ours, and we can explain every part.

**Explain:** Say it plainly, then show it: offer to walk through any function.
What was ours: choosing pure pursuit plus a planned line, the speed-change rule
for the table, deciding to tune the table with RL, every experiment and its
interpretation (for example finding the headless crash, and deciding what to
ship). The AI wrote and refactored much of the code to our design, and wrote
tests and documentation.

**If they push:**
- *"Then explain X line by line."* Do it. This guide and
  [driver_reference.md](driver_reference.md#how-the-code-works-file-by-file)
  cover every file.

---

### 2. The racing line

#### 2.1 How do you build the racing line?

**In one line:** Start from the official centreline and slide every point
sideways, a little at a time, until the lap bends as little as possible,
without letting the car come within 0.2 m of a wall.

**Explain:** Each of the ~392 centreline points can only move along the line
that is perpendicular to the track there (left or right). For each point we
first find how far it may move each way before the car would get too close to
a wall. Then we repeat 20,000 small steps: work out which way each point
should move to make the line straighter, move it a little, and clamp it within
its limits. Finally we respace the points evenly, 0.2 m apart: 380 points,
75.9 m per lap. This produces "wide in, clip the apex, wide out", the shape a
racing driver uses.

**Go deeper:** In [`build_raceline()`](../race_ws/src/team_driver/team_driver/raceline.py#L30):
1. [`left_directions()`](../race_ws/src/team_driver/team_driver/raceline.py#L53): unit vector to the left of travel, from (next point − previous point) turned 90°.
2. [`slide_limits()`](../race_ws/src/team_driver/team_driver/raceline.py#L60): test sideways positions every 0.025 m, up to 3 m each way, against the clearance map.
3. [`minimise_bending()`](../race_ws/src/team_driver/team_driver/raceline.py#L80): the "bend" at point i is `p[i−1] − 2·p[i] + p[i+1]` (zero on a straight). We minimise Σ |bend|². Its gradient for one point is the same pattern applied to the bends; each step moves the point against the gradient by `NUDGE` = 0.12, along its left direction only, then clamps it. This is **projected gradient descent**: a gradient step, then a projection back into the allowed range.
4. [`resample()`](../race_ws/src/team_driver/team_driver/raceline.py#L99): sliding bunches points on the inside of bends, so we redraw the loop with even 0.2 m spacing.

**If they push:**
- *"Why 20,000 iterations and 0.12?"* 0.12 is about the largest step that
  doesn't overshoot and blow up; with it, 20,000 steps let the line settle.
  It takes about 1.5 s at startup.
- *"Does it find the global best line?"* Σ bend² with box limits is a convex
  problem, so gradient descent heads to the one best answer for that measure.
  But "least bending" is not the same as "fastest" ([2.3](#23-why-minimise-bending-and-not-lap-time)).

#### 2.2 What is curvature, and how do you compute it?

**In one line:** Curvature κ is 1 / turn radius: 0 on a straight, larger in
tighter bends; we compute it as how fast the line's heading turns per metre.

**Explain:** A bend of radius 2 m has κ = 0.5 1/m. Our tightest point (the
hairpin) is about κ = 1.76, a radius of about 0.57 m. Curvature drives
everything: the cornering speed limit, the lookahead inputs, and how much
pure pursuit steers.

**Go deeper:** [`path_geometry()`](../race_ws/src/team_driver/team_driver/policies.py#L17):
heading at point i is the direction from point i−1 to i+1; curvature is the
heading change across the two neighbours (wrapped into −π..π) divided by the
distance between them. Positive means turning left.

#### 2.3 Why minimise bending and not lap time?

**In one line:** Least bending is simple, robust and convex, and it is a good
stand-in for "fast", because corner speed is limited by curvature; a true
minimum-time line would be better but much harder.

**Explain:** The corner speed limit is √(μg/κ), so a line with smaller
curvature can be driven faster. Minimising bending lowers curvature
everywhere at once. A minimum-time line also trades a slightly sharper bend for
a shorter path, and needs the speed profile inside the optimisation. It's on
our "another week" list.

**If they push:**
- *"How far from optimal are you?"* The table driven exactly would lap in
  12.28 s on this line; the car does about 13.7 s. Most of the gap is the car
  not tracking the table perfectly (it lags its speed command), not the line.

#### 2.4 How do you make sure the line never hits a wall?

**In one line:** Every point is only allowed to slide as far as keeps the
car's middle 0.355 m from any wall, and the finished line is checked against
the map; if it fails, the driver follows the centreline instead.

**Explain:** 0.355 m is the car's half-width (0.155 m) plus a 0.2 m gap
(`wall_gap`). The clearance map gives the distance to the nearest wall for any
point. After building, the line's closest approach is checked (it is 0.36 m on
icra26); if it were closer than 0.355 − 0.05 m, the driver logs it and uses the
centreline.

**Go deeper:** [`load_clearance_map()`](../race_ws/src/team_driver/team_driver/raceline.py#L146)
uses a **distance transform**: for every pixel, the distance in metres to the
nearest wall pixel. `clearance(points)` reads it at any (x, y), blending the 4
nearest pixels; off the map counts as wall.

#### 2.5 What about the gaps between the cones?

**In one line:** We seal every cone row into a solid wall before planning, so
the line can never go through a gap, which rule 11 forbids.

**Explain:** The gaps are 0.25–0.5 m, and the car is 0.31 m wide, so a
planner could squeeze through. We "close" the wall image with a 0.4 m disc:
grow every wall by 0.4 m, then shrink it back. Walls that touched while grown
stay joined, so every gap narrower than 0.8 m is filled, while the corridors
(over 1 m wide) stay open.

**If they push:**
- *"Why 0.8 m and not 0.4?"* Two walls each grow by 0.4 m towards each other,
  so they meet across any gap up to 0.8 m. (The code's comment used to say
  0.4 m; we found and fixed that.)

#### 2.6 Why build it at startup instead of shipping a file?

**In one line:** Rule 32 wants anything precomputed to be reproducible from
the map by our code, and building it at runtime goes further: it is a named
bonus item (rule 41), and it adapts to a new map automatically.

**Explain:** It costs about 1.5 s at startup, before the race starts. The
tuned speed table *is* shipped as a file, so it stores the line it was tuned
on, and the driver only uses it if the line built at startup matches.

---

### 3. Steering: pure pursuit

#### 3.1 How does pure pursuit work?

**In one line:** Pick a point on the racing line a short distance ahead, and
steer along the circle that passes through the car and that point.

**Explain:** Like a driver looking a little way down the road. Every scan:
find the nearest line point; walk forward until a point is at least L metres
away (the lookahead); work out how far that point is to the car's left; turn
that into a circle and a steering angle, limited to ±24°. Repeating this many
times a second keeps the car on the line.

**Go deeper:** The 7 steps are numbered in [`pure_pursuit()`](../race_ws/src/team_driver/team_driver/pure_pursuit.py#L51):
nearest point, lookahead, target, how far left, steering, free distance ahead,
speed. Target-finding has no loop (so JAX can compile it): list the path
indices in lap order from the nearest point, and `argmax` finds the first one
at least L away.

**If they push:**
- *"Why is pure pursuit known to cut corners?"* It aims at a point ahead, and
  the straight chord to that point cuts inside the curve. A longer L cuts more.
  Our smooth racing line and short lookahead keep this small.

#### 3.2 Derive the steering angle.

**In one line:** The circle through the car and the target has curvature
κ = 2y / L², and a car with wheelbase b drives that circle with steering angle
atan(b·κ).

**Explain:** Put the car at the origin facing along x. The target is at
(x, y), distance L away. A circle of radius R that touches the car's heading
has its centre at (0, R). The target is on it, so x² + (y − R)² = R². Expand:
x² + y² − 2yR = 0, and x² + y² = L², so R = L² / (2y), κ = 1/R = **2y / L²**.
For the steering: in the bicycle model, a car with wheelbase b and steering
angle δ turns on radius R = b / tan δ, so **δ = atan(b·κ)**. We use b = 0.33 m.

**Go deeper:** y ("how far left") is the target's position rotated into the
car's frame: `y = −sin(yaw)·dx + cos(yaw)·dy`. We use the chosen L in the
formula rather than the exact distance to the target point; they differ by at
most the 0.2 m point spacing.

**If they push:**
- *"What if y is negative?"* κ is negative, so we steer right. Same formula.
- *"What if the target is behind?"* It can't be: it's at least L ahead along
  the line, and the line passes close to the car.

#### 3.3 How do you choose the lookahead distance?

**In one line:** L ≈ 0.25 s × speed (at least 0.5 m), which our learned
policy confirmed: it learned L = 0.0035 + 0.249 × speed.

**Explain:** It is a trade-off. A short L follows the line tightly but zig-zags
at speed; a long L is smooth but cuts corners. Scaling L with speed means the
car always aims about a quarter of a second ahead. A sweep showed every gain
from 0.2 to 0.35 lapped cleanly, while 0.4 cut the infield hairpin and hit it.
The learned policy could also use curvature, but its curvature weights came out
near zero.

**Go deeper:** The policy is [`lookahead()`](../race_ws/src/team_driver/team_driver/policies.py#L52):
L = b + w·s with s = (speed, |κ| here, 1 m ahead, 3 m ahead, |κ1| − |κ0|),
clipped to 0.35–4 m. Learned weights: 0.249 on speed, 0.0026, 0.0015, 0.0044,
−0.0007 on the curvature terms, bias 0.0035.

#### 3.4 Why measure from the rear axle?

**In one line:** Pure pursuit's circle geometry assumes the point that moves
along the circle without sliding sideways, which is the rear axle, and that is
also what odometry reports.

**Explain:** In the bicycle model, the rear wheels roll straight along the
car's heading, so the rear axle traces the turning circle exactly. The
simulator internally tracks the centre of mass (0.17 m further forward), and
its odometry message converts back to the rear axle for us.

---

### 4. Speed

#### 4.1 How does the car decide how fast to go?

**In one line:** It reads a target speed from a table with one speed for each
of the racing line's 380 points; the table comes from physics and was then
tuned by reinforcement learning.

**Explain:** Physics gives the starting table: in every corner the fastest
speed the tyres can hold, then smoothed so neighbouring points never differ by
more than braking can achieve, so the car starts braking *before* a corner.
PPO then raised or lowered each speed based on how laps actually went. The
tuned speeds run 3.74–9.27 m/s. The old approach, still there as a fallback,
was reactive: the smallest of a braking limit to the wall ahead, a cornering
limit, and a top speed.

**If they push:**
- *"Why did the old approach lose time?"* Its braking limit looked at the wall
  straight ahead. In corners that is usually the outside wall, which the car
  never actually reaches, so it braked for nothing about two thirds of the lap.

#### 4.2 Explain the two rules of the speed table.

**In one line:** Rule 1: never faster than the tyres can hold in that corner,
√(μg/|κ|). Rule 2: between neighbouring points d metres apart, the speed can
only change as much as braking at a_brake allows, v_next² ≤ v² + 2·a_brake·d.

**Explain:**
- **Rule 1.** In a turn, the sideways acceleration is v²·κ, and the tyres
  grip at most μ·g sideways. So v²·κ ≤ μ·g, giving v ≤ √(μg/|κ|).
- **Rule 2.** To change speed from v to v_next at a constant rate a over a
  distance d takes v_next² − v² = 2·a·d (the standard constant-acceleration
  equation). We apply it both ways: speeding up out of corners and braking into
  them.

**Go deeper:** [`limit_speed_changes()`](../race_ws/src/team_driver/team_driver/raceline.py#L120)
walks round the lap forwards (cap each point at what the previous one can
reach) and then backwards (cap each point so the next one can still be reached
by braking). The lap is a loop, so a slow corner at the end also limits the
start: it repeats until nothing changes (2–3 rounds). Equivalent form:
(v_next − v)(v_next + v) = v_next² − v², so the speed change is at most
2·a·d / (v + v_next). The test in [`test_parity.py`](../training/test_parity.py)
checks this holds at every point.

**If they push:**
- *"Why no top speed?"* On a straight, rule 2 already limits it: the car must
  be able to brake for the next corner. The fastest point reaches 9.27 m/s.

#### 4.3 Where do μ = 1.0489 and a_brake = 6.05 come from?

**In one line:** μ is the simulator's own tyre-grip parameter for the
F1TENTH car; a_brake is a deliberately cautious braking value, below the car's
maximum of 9.51 m/s².

**Explain:** With these, the physics table alone would lap in 13.51 s if driven
perfectly. Using the full grip value is aggressive: on its own, the physics
table hit walls once ROS's reaction delay grew. PPO's tuning then
fixed that point by point, rather than us lowering μ everywhere (μ = 0.8 made it
clean but slow, about 15.9 s in the simulator).

**If they push:**
- *"Do they matter in the race?"* Only if the tuned table is rejected. With the
  tuned table loaded, the speeds come from the file, not from μ or a_brake.

#### 4.4 What are table_scale and speed_lead?

**In one line:** Two safety and tuning knobs on the table: `table_scale`
multiplies every speed (1.0 in the race), and `speed_lead` lets the car start
braking or accelerating slightly earlier (0, off).

**Explain:**
- **`table_scale`**: e.g. 0.95 drives everywhere at 95%. It is the quick
  safety margin if a machine reacts more slowly. Scaling all speeds down can't
  break rule 2.
- **`speed_lead`** (k): command `v_here + k × straightness × (v_1m_ahead − v_here)`,
  where straightness = 1 − |steering| / max_steer. The car's speed always lags
  its command, so it brakes a bit before the table drops; the straightness
  factor makes it do nothing at full lock, mid-corner. Off by default, because
  we haven't tuned it.

---

### 5. Reinforcement learning basics

#### 5.1 What is reinforcement learning, in our project's terms?

**In one line:** Learning by trial and error: the car tries slightly
different choices, gets a score (the reward), and the choices that scored
better than expected become more likely.

**Explain:**
- **Policy**: the rule that makes a decision. Ours: "the target speed at this
  point" (the table) and "how far ahead to aim" (the lookahead rule).
- **Action**: the decision itself, e.g. 6.2 m/s at point 140.
- **Reward**: the score per 0.01 s step: metres of progress along the track,
  minus 10 for touching a wall.
- **Episode**: one car's drive from a start until it crashes or 30 s pass.
- **Exploration**: trying something a bit different from the current best guess,
  so it can find something better.

#### 5.2 What is PPO, and why clip the ratio?

**In one line:** PPO (Proximal Policy Optimisation) makes better-than-expected
actions more likely, but caps how much one update can change the policy, so a
noisy batch of data can't wreck it.

**Explain:** After the cars drive, every action has an *advantage* (how much
better or worse than expected it went). PPO compares how likely each action is
under the new policy versus the one that drove: the **ratio**. It pushes the
ratio up for good actions and down for bad ones, but clips it to 1 ± 0.2, so no
action's probability changes by more than about 20% per update. That stability
is why PPO is the standard choice.

**Go deeper:** In [`ppo_policy_loss()`](../race_ws/src/team_driver/team_driver/policies.py#L67):
- ratio r = exp(log π_new(a) − log π_old(a));
- loss = −mean( min(r·A, clip(r, 0.8, 1.2)·A) ) − entropy bonus;
- the KL estimate mean((r − 1) − log r) measures how far the policy moved.

The `min` means: when an action was good, the gain stops counting once r > 1.2;
when bad, once r < 0.8. So there is no reward for moving further than the clip.

**If they push:**
- *"Why not plain policy gradient?"* Plain policy gradient takes one step per
  batch and can take a huge step on a lucky batch. PPO reuses each batch for
  several passes safely, because of the clip and the KL early stop.

#### 5.3 What does the critic do? What is an advantage? What is GAE?

**In one line:** The critic is a small network that predicts how much reward
is still to come from a situation; the advantage is how much better the actual
outcome was than that prediction; GAE is the recipe that blends real rewards
with the critic's predictions to compute it.

**Explain:** Raw rewards are noisy: a car scores more on a straight than in a
hairpin no matter what it does. Comparing against the critic's expectation for
*that* situation isolates the effect of the choice. Our critic sees the car's
speed, the curvature ahead, and (for the speed table) where it is on the lap.

**Go deeper:** In [`advantages_and_returns()`](../training/train.py#L136):
- one-step surprise: δ_t = r_t + γ·V(next) − V(now) (γ = 0.99, about 1 s of foresight);
- advantage: A_t = δ_t + γλ·A_{t+1}, summed backwards in time (λ = 0.95 for the
  table, 0.98 for the lookahead);
- a crash gets no future value; the 30 s time limit does (the car would have
  kept driving);
- the critic is trained towards the *return* = advantage + value.

The critic network: 2 hidden layers of 128 tanh neurons, about 17,500 weights.
It is only used in training; it does not ship.

#### 5.4 Why add random noise to the actions?

**In one line:** To explore: without trying slightly different speeds or
lookaheads, the policy can never find out that a different one is better.

**Explain:** During training each action is drawn from a bell curve (normal
distribution) centred on the policy's choice, with spread 0.3 (m/s for the
table, m for the lookahead). PPO needs to know how likely each drawn action
was: that is `gaussian_log_prob()`. In the race there is no noise: the car uses
the policy's best guess.

**If they push:**
- *"Who chooses 0.3?"* We do. For the table it is fixed, because when PPO was
  allowed to learn it, it shrank to 0.05 m/s mid-run and learning stopped
  ([10.3](#103-tell-us-about-a-bug-you-found)).

#### 5.5 What are Adam, the learning rate and the KL early stop for?

**In one line:** Adam is the gradient-descent method that actually changes the
weights; the learning rate sets the step size; the KL early stop ends an update
if the policy has already moved too far.

**Explain:**
- **Adam** keeps running averages of the gradient and of its square, so noisy
  minibatches cancel out and every weight moves about `lr` per step.
- **Learning rate**: 1e-3 for the table, 2.39e-4 for the lookahead (the paper's
  value), both falling linearly to 0 over the run, so late updates are small.
- **KL early stop**: once the KL estimate passes 1.5 × 0.015, the rest of that
  update's steps are skipped.
- **Gradient clipping**: gradients longer than 0.5 are shrunk, so one bad
  minibatch can't throw the weights far.

**Go deeper:** All of it is written out by hand in [train.py](../training/train.py):
[`adam_step()`](../training/train.py#L175) (with bias correction: both averages
start at 0, so early steps divide by 1 − β^step) and
[`clip_by_global_norm()`](../training/train.py#L188). No RL library is used.

#### 5.6 Why JAX, and why 2,048 cars at once?

**In one line:** JAX runs the simulator and the learning on the GPU, for
thousands of cars in parallel, so we get tens of millions of steps in minutes
instead of weeks.

**Explain:** f1tenth_gym has a JAX version of the race simulator. We write the
controller for one car; `jax.vmap` runs it for 2,048 cars at once; `jax.jit`
compiles each whole update into one GPU program. One update: 2,048 cars × 128
steps = 262,144 samples. About 55,000 simulated steps per second on our GPU.

**If they push:**
- *"Why does the controller code use `xp` everywhere?"* So the same function
  runs with numpy in the driver and jax.numpy in training ([8.2](#82-why-are-the-maths-files-free-of-ros)).

---

### 6. How we trained our two learned parts

#### 6.1 How did you train the speed table?

**In one line:** PPO in the GPU simulator, tuning one random 12 m stretch of
the table per update, with 2,048 cars, random reaction delays, and a reward of
metres of progress minus 10 per crash.

**Explain:** The policy *is* the table: 380 numbers, starting from the
physics table. Each update picks a random window of 60 consecutive points
(12 m). Only cars inside that window explore: each gets its own random speed
offset (±0.3 m/s), held for 2 m at a time. Only those samples teach the table,
and only those 60 speeds may change. After each update, the table is clipped to
1–15 m/s and pulled back within the braking rule. Two runs of 40 M steps (about
10 minutes each): the second continued from the first and widened the random
delay from 0–8 to 0–12 steps.

**Go deeper:**
- **Why held noise?** The simulator's speed control eases towards the command,
  so an offset lasting one 0.01 s step barely changes the speed. Held for 2 m,
  it changes the car's actual speed, so the reward can respond.
- **Why a window?** With all 380 points changing at once, it would be hard to
  tell which change caused a crash. A window focuses the learning signal. Random
  window starts mean the 2 m pieces line up differently each time, so every
  point is still tuned on its own.
- **Why mask outside the window?** Those points get no gradient, but Adam's
  momentum would still nudge them, so after each step they are reset.
- **Result:** ideal table lap 13.51 s → 12.28 s; 255 points raised, 123 lowered,
  on average +0.39 m/s. The largest rise is +1.93 m/s at point 84 (around
  (−2.5, 6.2)), the largest drop −1.86 m/s at point 166 (around (7.2, 13.3)).
  The slowest corner went 2.42 → 3.74 m/s.

**If they push:**
- *"Weakness of the method?"* After every update the table is pulled back
  within the braking rule, and that can only *lower* speeds. Random noise can
  therefore slowly drag braking and acceleration stretches down. A cleaner
  design would let the gradient see the constraint (an item for another week).

#### 6.2 Is tuning a table really reinforcement learning?

**In one line:** Yes: the table is a policy whose observation is the car's
position on the line and whose action is the target speed there, trained with
standard PPO.

**Explain:** It has all of PPO's parts: a stochastic policy (table speed plus
Gaussian noise), a learned critic that sees the car's state (speed, curvature
ahead, and position on the lap as sin/cos), GAE advantages, and the clipped
objective. What's unusual is only that the policy's "network" is a lookup
table.

#### 6.3 Why PPO and not a simpler search?

**In one line:** PPO gets a learning signal per step and per point, from
262,144 samples per update; a black-box search (random search, CMA-ES) only
gets one score per whole lap.

**Explain:** With 380 numbers, a black-box method must try many whole tables
and compare lap times, and it can't tell which corner made a lap better. PPO
with a critic knows which point each sample was at and whether that went better
than expected there. We already had the PPO pipeline for the lookahead, too.

**If they push:**
- *"Did you compare against random search?"* No. Be honest: a simpler method
  might have worked as well for a table this small. It would be a fair
  experiment.

#### 6.4 What is the reward? Could the car exploit it?

**In one line:** Metres of progress along the centreline per step, minus 10
for a wall contact (which also ends that car's episode); the car can't exploit
it, because no learned part can steer it off the racing line.

**Explain:** Cutting through the cones would score well, if it were possible.
It isn't: the steering always follows the racing line, which is built from a
map with the cone rows sealed. The lookahead only changes how far ahead pure
pursuit aims along that same line, and the table only sets the speed. Touching
anything costs −10 and ends the episode.

**Go deeper:** The lookahead reward also has three small terms from the
paper: +0.002 × speed, −0.01 × |change of L|, and −0.05 per step spent stalled.

#### 6.5 How did you train the lookahead policy? Is it useless?

**In one line:** PPO with the paper's settings, 2 M steps, on top of the tuned
table; it learned almost exactly our hand rule, so it adds little, and we say
so.

**Explain:** It follows Elgouhary & El-Wakeel (arXiv 2603.28625): a policy
that picks pure pursuit's lookahead from speed and curvature. Ours is linear
(L = b + w·s), so we can read what it learned: 0.249 × speed, with curvature
weights near zero. So it confirmed rather than improved 0.25 × speed. That is
still evidence: an optimiser searched nearby rules and found nothing clearly
better.

**Go deeper:** Why it found nothing: the racing line is already smooth; the
speed table already slows the car in corners, and L shrinks with speed, so L is
already short where it matters; and the run was short (30 updates). The
exploration noise *grew* from 0.30 to 0.37, a sign the reward gave no clear
direction.

**If they push:**
- *"Then why ship it?"* It's harmless and identical in effect. We could equally
  delete it; the honest framing is "an experiment that confirmed our rule".

#### 6.6 How much of your improvement is reinforcement learning?

**In one line:** In the simulator, RL tuning is worth about 2.3 s per lap at
zero wall contacts: a hand-tuned clean table manages ~15.9 s, the RL-tuned one
~13.6 s.

**Explain:** Three steps, measured in the training simulator:
1. Switching from the reactive speed rule to a physics speed table: 16.6 s →
   14.5 s, but it hit walls once the reaction delay grew.
2. Making it clean by hand (μ lowered to 0.8): about 15.9 s.
3. The RL-tuned table: about 13.6 s and clean.

So the *idea* of a planned table is worth a lot, and RL is what made it fast
*and* safe. In ideal-lap terms: 13.51 s → 12.28 s, about 9%.

**If they push:**
- *"Measured in ROS?"* Not as a full ablation. The commands to do it are in
  [Part C](#experiments-we-can-describe).

#### 6.7 How did you decide a model was good enough to ship?

**In one line:** Three gates: the training log (table lap down, crashes near
0), `evaluate.py` clean at every reaction delay, and the official referee
clean and faster.

**Explain:** [`test_parity.py`](../training/test_parity.py) first proves the
training maths and the driver's maths agree. Then [`evaluate.py`](../training/evaluate.py)
races one car per delay (0, 1, 2, … steps) and counts wall contacts: the
shipped table is clean through 9 steps. Then `./scripts/evaluate.sh`, the
judges' own script.

---

### 7. Simulation, testing and the real race

#### 7.1 Why trust the training simulator?

**In one line:** It is the same f1tenth_gym car model the race uses, set up
with the same parameters, and we checked it against the race.

**Explain:** Same single-track model with tyre slip, 0.01 s steps, vehicle
parameters, map, LiDAR angles and odometry point. One direct check: both put
the car into the infield hairpin at the same point, (5.09, 9.64). The known
differences are deliberate: only the LiDAR beams in the ±10° cone are
simulated (4× faster), and the reaction delay is modelled as a random 0–N steps.
In the end, the official referee is the judge, and lap times fell there too.

#### 7.2 Why did the car crash headless but not with the GUI?

**In one line:** Reaction delay: in ROS the simulator doesn't wait for our
driver, so each command lands a few steps late; headless, the simulator runs
faster, the delay in steps grew past what our table could handle, and the car
clipped walls.

**Explain:** This is our best story:
1. **Symptom.** With RViz on, clean laps of 13.6 s. With `--headless`, 11 wall
   contacts and disqualified, with the same code and the same files.
2. **Ruling things out.** We checked the driver's log: it loaded exactly the
   intended files. So it wasn't the code or the paths.
3. **Hypothesis.** The race simulator runs "as fast as possible". Headless it
   ran 1.3–1.5× real time; with RViz eating CPU, about 1.1×. Our driver answers
   in roughly fixed real time, so headless means more simulator steps pass
   before each command lands.
4. **Measurement.** In the training simulator, the table was clean up to a
   4-step delay but crashed from 5 steps on. It had only been trained on
   0–3 steps.
5. **Fixes.** Retrained with delays up to 8, then 12 steps (now clean through 9);
   the driver keeps only the newest scan (queue length 1), so it never steers
   from an old one; and it logs its measured delay every 200 scans.

**Go deeper:** At 9 m/s one 0.01 s step is 9 cm, so an 8-step delay means
steering from where the car was 0.7 m ago. At the old 6.8 m/s top speed the
same delay had been harmless, which is why it only appeared once the table
made the car faster.

**If they push:**
- *"How do you know the real delay now?"* The log line
  `reaction delay over 200 scans [steps]: age mean … max …; gap mean … max …`.
  Age is how many steps had passed since the scan when the command went out;
  gap is how many steps between answered scans (1 = keeping up).

#### 7.3 What do your parity tests check?

**In one line:** That the driver (numpy) and training (JAX) compute exactly the
same thing, so we race what we trained.

**Explain:** [`test_parity.py`](../training/test_parity.py), four checks:
1. `pure_pursuit()` still matches the original hand-written driver (1,000 random
   cases, to 1e-9).
2. numpy and JAX give the same pure pursuit command.
3. Lookahead policy: numpy and JAX agree on its inputs, L, the steering and the
   loss; JAX's gradient matches a numerical one; a saved file loads back.
4. Speed table: it obeys both rules at every point; numpy and JAX agree; the
   training loss's gradient lands only on points inside the window; a saved
   table loads back.

**Go deeper:** JAX uses 32-bit floats and numpy 64-bit, so the tolerance is
1e-3; real differences are about 1e-6.

#### 7.4 The judging machine is different. What changes?

**In one line:** Lap times are in simulated seconds, so a slower machine doesn't
slow the car directly, but it can increase the reaction delay; the table is
trained for up to 12 steps, and the log shows the real delay.

**Explain:** If the judges' delay were larger, `table_scale` = 0.95 is the
one-line safety margin. Our SUBMISSION.md says this, and the results we commit
in `results/submitted/` show what our machine did.

#### 7.5 Why not train in ROS directly?

**In one line:** ROS runs one car in real time; the JAX simulator runs 2,048
cars much faster than real time, so 40 M steps take 10 minutes instead of weeks.

---

### 8. The ROS node and the engineering

#### 8.1 What happens at startup and on every scan?

**In one line:** Startup builds everything slow once (line, table, policy);
each scan just runs the nearest-point → lookahead → pure pursuit → publish
chain, in about 0.1 ms.

**Explain:** In [`driver.py`](../race_ws/src/team_driver/team_driver/driver.py),
`__init__` does, in order: read parameters, build the racing line, compute its
curvature, build or load the speed table, load the lookahead policy, then open
the topics. ROS only starts calling the callbacks after that, so the car waits
on the grid while the line is built. The odometry callback just stores the
position, heading and speed; the scan callback plans and publishes.

#### 8.2 Why are the maths files free of ROS?

**In one line:** So training runs the *same functions* the race does:
`pure_pursuit.py`, `raceline.py` and `policies.py` take the maths library as an
argument, numpy in the driver and jax.numpy in training.

**Explain:** One copy of the maths means training and racing can't drift
apart, and the parity tests prove it. It's also why there are no `if` or `while`
statements on numbers inside those functions: JAX can't compile them.

#### 8.3 What if a model file is missing or wrong?

**In one line:** The driver always falls back to something that drives.

**Explain:**
- No `lookahead.npz`: use L = 0.25 × speed.
- No tuned table, or a table made for a different racing line: build the
  physics table from μ and a_brake.
- Racing line too close to a wall: follow the centreline.
- No odometry yet: send speed 0.
- The policy outputs something silly: L is clipped to 0.35–4 m.
- Settings changed since training: log a warning (it names the changed settings).

#### 8.4 Why a queue length of 1?

**In one line:** So the driver always answers the newest scan and never works
through a backlog of old ones, which would add exactly the reaction delay that
made the car late into corners.

---

### 9. What-ifs and limits

#### 9.1 How would you handle a track twice as fast?

**In one line:** The line and physics table rebuild themselves; we would
retrain the table, and because delay matters more at speed, measure the real
delay and train with a larger delay range.

**Explain:**
- Each step of delay covers twice the distance, so delay robustness is the first
  thing to check (`measure_delay()` logs it; train with a larger `--max-delay`).
- L = 0.25 × speed grows with speed, but the policy clips it at 4 m, so the cap
  may need raising.
- If the track has more grip, update μ and retrain.

#### 9.2 What about an obstacle mid-corner?

**In one line:** Today, the car would hit it, because in speed-table mode the
speed ignores the LiDAR; the braking part of the fix is small, but steering
around it needs a local planner.

**Explain:** The code already computes the braking limit
√(2·a_brake·(ahead − margin)) for the fallback mode, so obstacle-aware speed is
`min(table speed, that limit)`. We tested an emergency-brake version of this and
it cost 0.3 s per lap without making the car safer on the empty track, so it is
not in the race build. Steering around the obstacle needs more: shift the
pursuit target sideways into the widest LiDAR gap, or mark the obstacle in the
clearance map and re-plan the line locally.

#### 9.3 What happens on a different track?

**In one line:** Almost everything adapts automatically; only the tuned
speed table belongs to this track, and the driver detects that and falls back
to the physics table.

**Explain:** The clearance map, racing line and physics table are rebuilt from
the new map; the lookahead policy's inputs (speed and curvature) are generic.
The tuned table stores the line it was tuned on; a different line means it is
rejected. Retraining it takes about 20 minutes for the two runs.

#### 9.4 What if we double this parameter?

**In one line:** Know where it lives and which mode uses it; the full table is
in [Part C](#what-if-we-double-it).

**Explain:** The trap: with the shipped files loaded, the lookahead comes from
the policy (not `lookahead_gain`) and the speed from the tuned table (not μ or
a_brake). Those settings only matter if the learned files are off or rejected.

#### 9.5 Where does it break?

**In one line:** Obstacles, a different racing line, and reaction delays longer
than we trained for.

**Explain:**
1. **No obstacle braking** in speed-table mode ([9.2](#92-what-about-an-obstacle-mid-corner)).
2. **The tuned table is tied to one line.** Changing the map, `wall_gap` or
   `raceline_iters` makes it fall back to the slower physics table.
3. **Delay beyond 9–12 steps** is where the table starts touching walls in the
   simulator.
4. **`config/driver_params.yaml` holds stale values** (e.g. `lookahead_gain: 0.4`).
   It is only read by `ros2 launch`; the judges use `ros2 run`, which uses the
   defaults in code.
5. **The line minimises bending, not lap time.**

#### 9.6 What would you do with another week?

**In one line:** Obstacle-aware speed, ROS ablations to measure each part,
our own localisation, and a minimum-time line.

**Explain:**
1. Obstacle-aware speed (and then steering).
2. Ablations in the real referee: switch each learned part off and measure.
3. A particle filter against the map instead of ground-truth odometry (a bonus item).
4. A minimum-time racing line.
5. Let the speed-table training see the braking rule directly, instead of
   correcting after each update ([6.1](#61-how-did-you-train-the-speed-table)).

---

### 10. Things that did not work

#### 10.1 Show us a change that didn't work, and how you knew.

**In one line:** The untuned physics speed table: fast, but it hit walls once
the reaction delay grew, and we found out from a per-delay evaluation.

**Explain:** `evaluate.py` races one car per delay. The physics table lapped
14.5 s at delays 0–1 but had 12–22 wall contacts at delays 2–3. Lowering μ
to 0.8 made it clean but slow (15.9 s); tuning it with PPO made it clean and
fast (13.6 s). Other good examples are in [Part C](#things-that-did-not-work).

#### 10.2 What happened to the residual network?

**In one line:** We built a small neural network that added a capped
correction to pure pursuit's steering and speed, but never trained it to a
model good enough to race, so we moved it to `unused/`.

**Explain:** The pipeline worked (numpy and JAX agreed, weights saved and
loaded). But the planned speed table gave far more, more simply, so the
network was never worth finishing. It is kept in [unused/](../unused/README.md)
with notes on how it was wired in.

#### 10.3 Tell us about a bug you found.

**In one line:** Pick one: exploration noise that collapsed, Adam nudging
untouched speeds, or a misleading comment about the cone gaps.

**Explain:**
- **Collapsing noise.** When PPO learned the table's exploration spread, it
  shrank to 0.05 m/s mid-run and the table stopped improving. We saw it in the
  `noise_std` column of the training log; now it is fixed at 0.3.
- **Adam leaking.** Speeds outside the training window had zero gradient but
  still moved, through Adam's momentum. Fix: put them back after every step.
- **Cone-gap comment.** The comment said gaps under 0.4 m are sealed; the code
  actually seals gaps under 0.8 m (the full width of the 0.4 m-radius disc).
  The code was right; we fixed the comment.

---

## Part C: quick reference

### Numbers to remember

| What | Number |
| --- | --- |
| Lap time history (official referee) | ~27 s → < 18 s → 16.68 s → **~13.7 s** |
| Car | wheelbase 0.33 m, width 0.31 m, steering ±0.4189 rad (24°) |
| Physics step | 0.01 s; one LiDAR scan per step |
| Racing line | 380 points, 0.2 m apart, 75.9 m (centreline 392 points) |
| Wall margin | 0.155 m half-width + 0.2 m gap = 0.355 m; closest point 0.36 m |
| Line building | 20,000 steps of size 0.12, about 1.5 s at startup |
| Cone sealing | 0.4 m disc, seals gaps under 0.8 m; cone gaps are 0.25–0.5 m |
| Lookahead | hand rule max(0.5, 0.25·v); learned 0.0035 + 0.249·v, clipped 0.35–4 m |
| Grip and braking | μ = 1.0489, a_brake = 6.05 m/s² (car's maximum 9.51) |
| Speed table | physics 2.42–9.34 m/s; tuned **3.74–9.27 m/s** |
| Ideal table lap | 13.51 s physics → **12.28 s** tuned |
| RL's changes to the table | 255 raised, 123 lowered, mean +0.39 m/s, range −1.86 to +1.93 m/s |
| PPO batch | 2,048 cars × 128 steps = 262,144 samples per update |
| PPO settings | clip 0.2, γ 0.99, λ 0.95 (table) / 0.98 (lookahead), 5 epochs × 32 minibatches, target KL 0.015 |
| Speed-table training | window 60 points (12 m), noise held 10 points (2 m), spread 0.3 m/s, speeds 1–15 m/s, lr 1e-3 |
| Delays trained | table 0–8 then 0–12 steps; lookahead 0–10 |
| Training time | ~10 min per 40 M-step table run; ~2 min for the lookahead |
| Shipped runs | table `20261013-021003`; lookahead `20261013-030026` |
| Delay tolerance (simulator) | shipped table clean through 9 steps |
| Control tick | ~0.1 ms of maths per scan |

### What if we double it

The trap: with the shipped files loaded, `lookahead_gain`, `mu` and `a_brake`
don't affect the race. They matter only if the learned files are off or rejected.

| Parameter (default) | What it does | Doubled | Halved |
| --- | --- | --- | --- |
| `lookahead_gain` (0.25 s) | L = gain × speed, without the policy | Smoother but cuts corners; 0.4 already hit the hairpin, so 0.5 likely crashes | Zig-zags on straights, unstable at speed |
| `min_lookahead` (0.5 m) | Shortest L | Only matters below 2 m/s: a lazier start | Twitchy at low speed |
| `mu` (1.0489) | Grip for corner speeds (physics table) | Corner speeds +41%, beyond real grip: slides into walls | Corners 29% slower: safe, slow |
| `a_brake` (6.05 m/s²) | Braking/acceleration we plan with | Brakes later than physics allows: overshoots corners | Brakes very early: safe, slow |
| `table_scale` (1.0) | Multiplies every table speed | Up to ~18 m/s: crashes at the first corner | Half speed, about a 27 s lap. 0.9–0.95 is a sensible margin |
| `speed_lead` (0) | Lean towards the speed 1 m ahead | — | — |
| `wall_gap` (0.2 m) | Air between car and wall on the line | Safer, slower line; **and** the tuned table is rejected | Tighter line that clips walls; same rejection |
| `NUDGE` (0.12) | Step size of the line smoothing | Overshoots and blows up | Needs about twice the iterations |
| `ITERS` (20,000) | Line-smoothing steps | Smoother line, slower startup | Line not fully relaxed |
| `max_steer` (0.4189 rad) | Steering limit | The car's physical limit: changing it here doesn't change the car | — |
| `COLLISION_PENALTY` (10) | Reward lost per crash | More cautious, slower table | Riskier, faster table, more crashes |
| `--max-delay` (12) | Largest random training delay | More robust to slow machines, slower | Faster in sim, riskier in ROS |
| PPO `clip` (0.2) | Max policy change per update | Bigger, riskier steps; training can collapse | Slower, steadier learning |

### Things that did not work

| What we tried | What happened | How we knew |
| --- | --- | --- |
| Constant speed | ~27 s laps | Lap timer |
| `lookahead_gain` 0.4 | Hit the infield hairpin | Sweep: 0.2–0.35 clean, 0.4 contacts |
| Untuned physics speed table | 14.5 s but 12–22 contacts at delays 2–3 | `evaluate.py` per delay |
| Training on delays 0–3 only | Crashed headless in ROS | Referee log; sim crashes from 5 steps |
| Learned exploration noise | Shrank to 0.05 m/s; learning stopped | `noise_std` in `log.csv` |
| Noise lasting one 0.01 s step | Car's speed barely changed: nothing to learn | Hence noise held over 2 m |
| LiDAR emergency brake | 0.3 s per lap slower, no safer | `evaluate.py` at delays 0–14 |
| Learned curvature-based lookahead | Learned ≈ the hand rule | The weights in `lookahead.npz` |
| Residual neural network | Never trained to a shippable model | No shippable run; moved to `unused/` |

### Experiments we can describe

Ablations: switch one part off and measure. We have not run all of these in
ROS; say so if asked.

| Question | How |
| --- | --- |
| What is the tuned table worth over the physics table? | `ros2 run team_driver driver --ros-args -p speed_table_weights:=/nonexistent` |
| What is the learned lookahead worth? | `... -p use_lookahead:=false` |
| What is the speed table worth at all? | `... -p use_speed_table:=false` |
| Same, offline | `~/venvs/rl/bin/python training/evaluate.py --speed-table race_ws/src/team_driver/models/speed_table.npz --max-delay 12`, with and without `--lookahead race_ws/src/team_driver/models/lookahead.npz` |
| How robust is it to slow machines? | `evaluate.py ... --max-delay 15`, then look at contacts per delay |

### A 30-minute mock interview

Do this in pairs, with the asker reading questions from Part A and the
answerer not looking at Part B. Swap after 30 minutes.

1. 1.1 walk-through (2 min), then 1.2 and 1.3.
2. One deep dive the answerer did *not* build: 3.2, 4.2 or 5.2.
3. Two RL questions: 6.1 and 6.6.
4. The headless story, 7.2, start to finish.
5. Two what-ifs from section 9, then one row of the doubling table.
6. One failure from section 10, with how you knew.

Score each answer: did it start with a one-line answer, give a reason, and end
with evidence (a number or a code location)?
