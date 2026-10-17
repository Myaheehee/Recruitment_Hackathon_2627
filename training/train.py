"""Train a driving policy with PPO in the GPU training simulator; run `~/venvs/rl/bin/python training/train.py --method speed_table`.

Two kinds of policy (both in race_ws/src/team_driver/team_driver/policies.py):
    --method lookahead   (default) a linear policy, L = b + w . s, picks pure
                         pursuit's lookahead distance from 5 inputs
                         (Elgouhary & El-Wakeel 2026, arXiv 2603.28625)
    --method speed_table tunes the speed table (raceline.speed_table), one
                         speed per racing-line point; see "The speed table" below

    python training/train.py --smoke      # 2-minute check that everything runs
    python training/train.py              # lookahead, 20M steps: about 4-7 minutes
    python training/train.py --method speed_table   # 40M steps: about 8-13 minutes

"python" here means ~/venvs/rl/bin/python, the venv with JAX and
f1tenth_gym (made by training/setup_venv.sh). The times come from the runs
in training/runs/ and include the minute or two the first update spends
compiling.

Every run writes to training/runs/<date-time>/:
    lookahead.npz (or                 the policy, rewritten every 20 updates,
    speed_table.npz)                  so a stopped run still leaves usable weights
    log.csv                           one row per update, for plotting
    args.json                         the settings used, so the run can be repeated

What shipped (race_ws/src/team_driver/models/):
    speed_table.npz  run 20261013-021003: --method speed_table, 40M steps,
                     --max-delay 12, --init from run 20261013-014155
                     (itself --method speed_table, 40M steps, --max-delay 8)
    lookahead.npz    run 20261013-030026: --method lookahead, --steps 2e6,
                     --envs 512, --max-delay 10, --speed-table set to the
                     shipped speed_table.npz (about 2 minutes)

Words used in this file:
    PPO         proximal policy optimisation, the reinforcement-learning
                (RL) method used here: make actions that went better than
                expected more likely, but only by a little per update
                (--clip), so the policy cannot jump far on noisy data
    policy      what is being trained: it turns what a car sees into a
                command (a lookahead distance, or the speed table's speeds)
    episode     one car's drive from a (re)start until it crashes or reaches
                the time limit (--episode-steps, 30 s); it then restarts at a
                random point of the track
    rollout     the driving done between two updates: every car drives
                --rollout (128) steps with the current policy, and every
                step is kept as data to learn from
    critic      a second, helper network used only in training: it guesses
                how much reward is still to come from where a car is now
                (that guess is called the "value")
    advantage   how much better (positive) or worse (negative) an action
                turned out than the critic expected
    GAE         generalised advantage estimation, the recipe used for the
                advantage. It mixes the critic's guess one step later with
                the real rewards many steps ahead; --gae-lambda sets the mix
                (0 = trust the critic after one step, 1 = add up real rewards)
    return      advantage + value: the reward-to-go the critic should have
                predicted; the critic is trained towards it
    KL          Kullback-Leibler divergence: one number for how far the
                policy has moved from the one that drove the rollout. An
                update stops early once it gets too big (--target-kl)
    entropy     how spread out the policy's random choices are; a bonus for
                it (--entropy) stops exploration dying out too early
    epoch       one pass over all of a rollout's data, cut into --minibatches
                random pieces (minibatches), with one Adam step for each
    jit, vmap, lax.scan   JAX tools. jax.jit compiles a function once so
                later calls run fast on the GPU; jax.vmap turns a function
                for one car into one for every car at once; jax.lax.scan
                runs a loop inside compiled code, passing a "carry" (the
                values the loop updates) from one step to the next

The method, in the order it happens:
  1. 2048 cars drive in parallel on the GPU, each from a random point of the
     track with a random reaction delay (sim.py). Every 0.01 s the policy acts,
     plus random noise so it tries things it would not otherwise.
  2. Each step scores a reward: metres of progress along the track, minus 10
     for touching a wall (which also ends that car's episode; it restarts
     elsewhere). The lookahead method adds three terms from the paper's
     reward (eq. 13): a bonus for speed, a cost for changing L, and a cost
     for standing still (--w-speed, --w-jerk, --w-stall). The speed table
     uses the simulator's reward alone.
  3. After `rollout` steps, work out each action's advantage: how much better
     or worse it went than the critic predicted. Then gradient descent on the
     loss (PPO's policy loss from policies.py, plus the critic's error) for a
     few passes over the data. The gradient descent is Adam, written out below.
  4. Repeat until --steps simulated steps have been used.
Steps 1-3 are one "update", built by make_update() and compiled as a whole.

Everything PPO needs is written out in this file: Adam, GAE, gradient
clipping and the KL early stop. No RL or optimiser library is used; optax
is listed in training/requirements.txt but never imported.

The lookahead method uses the paper's PPO settings (section III-J): discount
0.99, GAE lambda 0.98, 5 passes, clip 0.2, target KL 0.015, entropy 0.02,
value weight 0.6, learning rate 2.3927e-4 falling linearly to 0.

The speed table (--method speed_table). The policy is the table: 380
numbers, starting from the one raceline.speed_table works out from mu and
a_brake. Each update tunes one random stretch of WINDOW (60) consecutive
points, 12 m of line:
  - Only cars inside the stretch explore. Each gets one random speed offset
    per NOISE_SEGMENT (10) points, held for the whole update: a change of
    speed command lasting one 0.01 s step barely changes the car's speed (the
    simulator eases towards it), but one held for 2 m does. Cars elsewhere
    drive the table as it is.
  - Only their samples count in the policy loss, and only the stretch's 60
    speeds may change. The critic learns from every car.
  - After the update the table is clipped to SPEED_RANGE and lowered wherever
    neighbouring points break the speed-change rule
    (raceline.limit_speed_changes), so it keeps to the same rule it started from.
Because each update's stretch starts at a random point, the 2 m pieces line
up differently every time, and over many updates every point is tuned on its
own. Steering is pure pursuit with its own lookahead rule
(DEFAULTS['lookahead_gain'] x speed, at least DEFAULTS['min_lookahead']).
This method ignores --speed-table; to keep tuning a saved table, pass it
with --init.

The advantage and loss code follows external/f1tenth_gym/examples/
jax_ppo_training.py (MIT licence), which in turn follows PureJaxRL.
"""

import argparse
import csv
import json
import math
import time
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

import sim
from team_driver.pure_pursuit import DEFAULTS
from team_driver.raceline import lap_time, limit_speed_changes, speed_table
from team_driver.policies import (
    LOOKAHEAD_INPUTS, LOOKAHEAD_RANGE, START_BIAS, START_WEIGHTS, gaussian_log_prob,
    load_speed_table, lookahead, lookahead_loss, path_geometry, speed_table_loss)

COLLISION_PENALTY = 10.0   # reward lost on touching a wall [metres of progress]
MAX_GRAD_NORM = 0.5        # gradients longer than this are shrunk (see clip_by_global_norm)
STALL_SPEED = 0.05         # [m/s] slower than this counts as standing still (the paper's value)
STALL_STEPS = 50           # stalled = standing still for longer than this many steps (0.5 s)

# Speed table only (see "The speed table" above).
WINDOW = 60                # consecutive points tuned per update (12 m of line)
NOISE_SEGMENT = 10         # a car's exploration offset is held for this many points (2 m)
SPEED_RANGE = (1.0, 15.0)  # [m/s] tuned speeds are kept inside this
SPEED_TABLE_INPUTS = 7     # what its critic sees: policies.speed_table_inputs

# PPO settings for each method; any of them can be overridden on the command line.
SETTINGS = {
    'lookahead': dict(steps=20e6, epochs=5, minibatches=32, lr=2.3927e-4, lr_decay=True,
                      gamma=0.99, gae_lambda=0.98, entropy=0.02, value_coef=0.6, target_kl=0.015),
    # Each point is only in the window about 1 update in 6, so it needs more
    # steps, and a larger learning rate than the lookahead's 5 weights. The
    # exploration spread is fixed for this method (see loss()), so entropy
    # does nothing here.
    'speed_table': dict(steps=40e6, epochs=5, minibatches=32, lr=1e-3, lr_decay=True,
                        gamma=0.99, gae_lambda=0.95, entropy=0.0, value_coef=0.5, target_kl=0.015),
}


class Sample(NamedTuple):
    """What PPO keeps from one step of one car.

    drive_one_step() in make_update() makes one Sample per step, with one
    row per car: every field has shape (cars,) or (cars, n). jax.lax.scan
    then stacks the `rollout` steps of a rollout, giving shape
    (rollout, cars) or (rollout, cars, n). The comments next to the fields
    say what each one means; the shapes and units that depend on the
    method are listed here.

    Attributes
    ----------
    inputs : jax.Array, shape (cars, n)
        What the policy and the critic saw. Lookahead: n = 5 (v, |k0|,
        |k1|, |k2|, dk); speed table: n = 7 (policies.speed_table_inputs,
        read by the critic only, as the table itself needs no inputs).
    action : jax.Array, shape (cars,)
        Lookahead: L [m], before clipping. Speed table: the table's speed
        plus any exploration offset [m/s], before clipping to SPEED_RANGE.
    log_prob : jax.Array, shape (cars,)
        Log-probability of `action`.
    reward : jax.Array, shape (cars,)
        The simulator's reward (progress [m], minus COLLISION_PENALTY on a
        crash), plus the lookahead method's own terms: see drive_lookahead
        in make_update().
    used : jax.Array, shape (cars,)
        Lookahead: L after clipping to LOOKAHEAD_RANGE [m]. Speed table:
        the speed command after clipping to SPEED_RANGE [m/s].
    """
    inputs: jax.Array          # what the policy saw
    action: jax.Array          # what it picked, noise included
                               # (lookahead: L before clipping to LOOKAHEAD_RANGE)
    log_prob: jax.Array        # how likely the policy was to pick it
    value: jax.Array           # what the critic predicted
    reward: jax.Array
    terminated: jax.Array      # episode ended by a crash
    truncated: jax.Array       # episode ended by the time limit
    end_value: jax.Array       # critic's prediction for where the step ended up
    progress: jax.Array        # metres along the track this step
    used: jax.Array            # what reached the car (lookahead: the clipped L)
    l_change: jax.Array        # lookahead: |L - previous L| [m] (speed table: 0)
    stalled: jax.Array         # lookahead: standing still too long (speed table: False)
    point: jax.Array           # speed table: the path point the car was nearest to (others: 0)
    weight: jax.Array          # 1 if the sample counts in the policy loss, else 0
                               # (speed table: inside the window; others: always 1)


class Minibatch(NamedTuple):
    """What the learning step needs from each sample.

    In update() all the rollout's samples are flattened into one Minibatch
    whose fields have a first axis of length batch = envs x rollout
    (262,144 with the defaults); one_epoch() shuffles it and cuts it into
    `minibatches` pieces, so loss() sees fields with a first axis of length
    batch / minibatches (8,192 with the lookahead and speed-table
    settings, 32 minibatches).

    Attributes
    ----------
    inputs : jax.Array, shape (batch, n)
        As Sample.inputs.
    action : jax.Array, shape (batch,)
        As Sample.action: what the policy picked, noise included.
    log_prob : jax.Array, shape (batch,)
        Log-probability of `action` under the policy that drove the
        rollout (the "old" policy, before this update changed it).
    advantage : jax.Array, shape (batch,)
        From GAE, rescaled to average 0 and spread 1.
    ret : jax.Array, shape (batch,)
        The return: advantage (before rescaling) + value, in reward units
        (mostly metres of progress); what the critic is trained towards.
    point : jax.Array, shape (batch,), int
        Speed table: nearest racing-line point (others: 0).
    weight : jax.Array, shape (batch,)
        1 if the sample counts in the policy loss, else 0.
    """
    inputs: jax.Array
    action: jax.Array
    log_prob: jax.Array
    advantage: jax.Array       # how much better than expected it went
    ret: jax.Array             # the reward-to-go the critic should have predicted
    point: jax.Array
    weight: jax.Array


# ----------------------------------------------------------------------
# The critic: a helper used only in training, a small network guessing how
# much reward is still to come from here, so that PPO can tell
# better-than-expected actions from worse ones. (The policies themselves are
# in policies.py.)
# ----------------------------------------------------------------------
def network(key, n_in, hidden, n_out, last_scale):
    """Make random starting weights for the critic network (see critic()).

    The network is n_in -> hidden -> hidden -> n_out: two hidden layers of
    tanh neurons, then a plain linear output. Weights start "orthogonal"
    (random, but with the columns of each matrix at right angles to each
    other, which keeps signals a sensible size through the layers), scaled
    by sqrt(2) for the hidden layers and by `last_scale` for the output
    layer, as in f1tenth_gym's jax_ppo_training.py example. Biases start
    at 0.

    Parameters
    ----------
    key : jax.Array
        JAX random key.
    n_in : int
        Number of inputs.
    hidden : int
        Neurons per hidden layer (--hidden, default 128).
    n_out : int
        Number of outputs.
    last_scale : float
        Scale of the output layer's starting weights (1.0 for the
        critic).

    Returns
    -------
    dict
        'W0' (n_in, hidden), 'b0' (hidden,), 'W1' (hidden, hidden),
        'b1' (hidden,), 'W2' (hidden, n_out), 'b2' (n_out,).
    """
    keys = jax.random.split(key, 3)
    sizes = [(n_in, hidden, math.sqrt(2.0)), (hidden, hidden, math.sqrt(2.0)),
             (hidden, n_out, last_scale)]
    weights = {}
    for i, (k, (a, b, scale)) in enumerate(zip(keys, sizes)):
        weights[f'W{i}'] = jax.nn.initializers.orthogonal(scale)(k, (a, b))
        weights[f'b{i}'] = jnp.zeros(b)
    return weights


def init_params(key, method, hidden, speeds=None):
    """Make the starting parameters (policy, critic and exploration noise).

    Parameters
    ----------
    key : jax.Array
        JAX random key.
    method : str
        'lookahead' or 'speed_table'.
    hidden : int
        Neurons per hidden layer of the critic.
    speeds : array of shape (points,), optional
        Speed table only: the starting table [m/s].

    Returns
    -------
    dict
        'actor': the policy. Lookahead: {'w': (5,), 'b': scalar}, starting
        at START_WEIGHTS and START_BIAS (L = 0.25 * v). Speed table:
        {'speeds': (points,)} [m/s].
        'critic': a network() with 5 (lookahead) or 7 (speed table)
        inputs and 1 output.
        'log_std': log of the exploration noise's spread (standard
        deviation), a scalar, log(0.3): 0.3 m of L or 0.3 m/s of speed.
    """
    _, k_critic = jax.random.split(key)
    if method == 'speed_table':
        return {
            'actor': {'speeds': jnp.asarray(speeds, dtype=jnp.float32)},
            'critic': network(k_critic, SPEED_TABLE_INPUTS, hidden, 1, 1.0),
            'log_std': jnp.asarray(math.log(0.3)),      # explore about +-0.3 m/s
        }
    return {
        'actor': {'w': jnp.asarray(START_WEIGHTS), 'b': jnp.asarray(START_BIAS)},
        'critic': network(k_critic, len(LOOKAHEAD_INPUTS), hidden, 1, 1.0),
        'log_std': jnp.asarray(math.log(0.3)),      # explore about +-0.3 m around L
    }


def critic(weights, x):
    """Run the critic: its guess of the reward still to come from each input.

    Two hidden layers of tanh neurons (tanh squashes any number into
    -1..1), then a plain linear output with one number.

    Parameters
    ----------
    weights : dict
        The critic's layers, from network().
    x : jax.Array, shape (..., n_in)
        Inputs, one row per sample (any number of leading axes).

    Returns
    -------
    jax.Array, shape (...)
        The value: predicted discounted reward to come, in reward units
        (mostly metres of progress).
    """
    h = jnp.tanh(x @ weights['W0'] + weights['b0'])
    h = jnp.tanh(h @ weights['W1'] + weights['b1'])
    return (h @ weights['W2'] + weights['b2'])[..., 0]


def advantages_and_returns(s, gamma, lam):
    """Work out each step's advantage and return by GAE.

    Generalised advantage estimation: for each step, how much better did
    things go than the critic expected? A crash gets no future value; a time
    limit does (the car would have kept driving). Arrays are (time, cars).

    For one step, delta = reward + gamma * value_after - value_before:
    the surprise over a single step. The advantage adds up the deltas of
    the following steps too, each further one weighted by another factor
    gamma * lam:
        advantage[t] = delta[t] + gamma * lam * advantage[t + 1]
    That is easiest to work out backwards in time, from the last step to
    the first, which is what the jax.lax.scan below does. The sum stops at
    the end of an episode (crash or time limit), because the next step of
    that car belongs to a new episode after the restart.

    Parameters
    ----------
    s : Sample
        The rollout; fields used (reward, value, end_value, terminated,
        truncated) have shape (rollout, cars).
    gamma : float
        Discount: how much a reward one step later counts (0.99).
    lam : float
        GAE lambda (0.95 or 0.98): how far ahead the sum of deltas
        reaches before it trusts the critic instead.

    Returns
    -------
    advantage : jax.Array, shape (rollout, cars)
        How much better than expected each step went.
    ret : jax.Array, shape (rollout, cars)
        advantage + value: the target the critic is trained towards.
    """
    bootstrap = jnp.where(s.terminated, 0.0, s.end_value)
    deltas = s.reward + gamma * bootstrap - s.value
    continues = ~(s.terminated | s.truncated)

    def backwards(next_advantage, inputs):
        """One step of the backwards pass (run by jax.lax.scan, last step first).

        Parameters
        ----------
        next_advantage : jax.Array, shape (cars,)
            The advantage of the step after this one (0 after the last step).
        inputs : tuple
            delta, shape (cars,), this step's one-step surprise; and cont,
            shape (cars,), bool, False where the episode ended this step.

        Returns
        -------
        carry, output : jax.Array, shape (cars,)
            This step's advantage, twice: once carried to the step before,
            once collected as the output.
        """
        delta, cont = inputs
        advantage = delta + gamma * lam * cont * next_advantage
        return advantage, advantage

    _, adv = jax.lax.scan(backwards, jnp.zeros_like(deltas[-1]),
                          (deltas[::-1], continues[::-1]))
    adv = adv[::-1]
    return adv, adv + s.value


# ----------------------------------------------------------------------
# Gradient descent: Adam, written out.
# ----------------------------------------------------------------------
class AdamState(NamedTuple):
    """What Adam remembers between steps.

    `m` and `v` are dicts with the same keys and array shapes as the
    parameters themselves (one running average per parameter).
    """
    step: jax.Array            # how many steps so far
    m: dict                    # running average of the gradient
    v: dict                    # running average of the squared gradient


def adam_init(params):
    """Make Adam's starting memory: step 0 and both running averages zero.

    Parameters
    ----------
    params : dict
        The parameters Adam will train, from init_params().

    Returns
    -------
    AdamState
        step 0; m and v zero, shaped like `params`.
    """
    return AdamState(jnp.zeros((), jnp.int32),
                     jax.tree.map(jnp.zeros_like, params), jax.tree.map(jnp.zeros_like, params))


def adam_step(params, grads, state, lr, b1=0.9, b2=0.999, eps=1e-5):
    """Take one step of Adam (Kingma & Ba 2015).

    Adam is the usual kind of gradient descent for neural networks.

    Plain gradient descent does   param <- param - lr * gradient.
    Adam keeps two running averages instead. `m` averages the gradient over
    recent steps, so minibatch noise cancels out. `v` averages its square;
    dividing by sqrt(v) makes every parameter's step about lr long, whether
    its gradient is huge (the speed weight, since v is up to 7 m/s) or tiny.

    Parameters
    ----------
    params : dict
        Current parameters.
    grads : dict
        Gradient of the loss, same keys and shapes as `params`.
    state : AdamState
        Adam's memory from the previous step.
    lr : float or scalar jax.Array
        Learning rate: roughly how far each parameter may move per step.
    b1, b2 : float, optional
        How slowly the averages m and v forget old gradients (0.9 and
        0.999: about the last 10 and 1000 steps).
    eps : float, optional
        Added to sqrt(v) so a near-zero gradient cannot cause a division
        by zero.

    Returns
    -------
    params : dict
        The parameters after the step.
    state : AdamState
        Adam's updated memory.
    """
    step = state.step + 1
    m = jax.tree.map(lambda m, g: b1 * m + (1 - b1) * g, state.m, grads)
    v = jax.tree.map(lambda v, g: b2 * v + (1 - b2) * g * g, state.v, grads)
    # Both averages start at zero, so for the first steps they are too
    # small. Dividing by (1 - b^step) undoes that.
    m_fix, v_fix = 1.0 / (1.0 - b1 ** step), 1.0 / (1.0 - b2 ** step)
    params = jax.tree.map(lambda p, m, v: p - lr * (m * m_fix) / (jnp.sqrt(v * v_fix) + eps),
                          params, m, v)
    return params, AdamState(step, m, v)


def clip_by_global_norm(grads, max_norm):
    """Shrink the gradient if it is too long (gradient clipping).

    Treat all the gradients as one long vector; if it is longer than
    max_norm, shrink it to that length. One unlucky minibatch then cannot
    throw the weights far.

    Parameters
    ----------
    grads : dict
        Gradient of the loss, one array per parameter.
    max_norm : float
        Longest length allowed (MAX_GRAD_NORM, 0.5).

    Returns
    -------
    dict
        The gradient, same direction, length at most `max_norm`.
    """
    norm = jnp.sqrt(sum(jnp.sum(g * g) for g in jax.tree.leaves(grads)))
    scale = jnp.minimum(1.0, max_norm / (norm + 1e-8))
    return jax.tree.map(lambda g: g * scale, grads)


# ----------------------------------------------------------------------
# One update: drive, then learn.
# ----------------------------------------------------------------------
def make_update(simulator, path, args, updates):
    """Build the compiled function that does one whole PPO update.

    One call of the returned update(runner) does, all on the GPU:
      1. Drive (the rollout): every one of the `args.envs` cars drives
         `args.rollout` steps of 0.01 s with the current policy plus
         exploration noise. jax.lax.scan runs drive_one_step (one of
         drive_lookahead or drive_speed_table) that many
         times and stacks each step's Sample, giving arrays of shape
         (rollout, cars) or (rollout, cars, n).
      2. Score: advantages_and_returns() (GAE) gives every step's advantage
         and return; the advantages are then rescaled to average 0 and
         spread 1.
      3. Learn: the batch of envs x rollout samples (262,144 by default) is
         flattened into one Minibatch. In each of `args.epochs` epochs it is
         shuffled and cut into `args.minibatches` minibatches, with one Adam
         step (gradient clipped to MAX_GRAD_NORM) on each. That is a
         jax.lax.scan over epochs, each a jax.lax.scan over minibatches:
         epochs x minibatches steps in all (5 x 32 = 160 with the lookahead
         and speed-table settings). Once the KL passes 1.5 x target_kl, the
         rest of this update's steps are thrown away.
      4. Measure: a dict of averages for log.csv and the printout.
    Because the whole update is one jax.jit-compiled function, Python steps
    in only once per update. The first call compiles it, which takes a
    minute or two.

    Before building it, this sets up the controllers from sim.py. The speed
    table they drive by is --speed-table if given (checked to fit this
    racing line), else the one the driver would build from DEFAULTS; either
    way scaled by DEFAULTS['table_scale'].

    The runner (what update() takes and gives back) is a 5-tuple:
        params       dict from init_params(): 'actor', 'critic', 'log_std'
        adam         AdamState for params
        carry        every car's state between steps (below)
        key          JAX random key
        index        updates done so far (int scalar), for the learning-rate decay

    The carry is a 6-tuple, one row per car:
        0 env          simulator state of every car (f1tenth_gym BatchState)
        1 observation  what each car sees now (LiDAR scan, car state);
                       a fresh one after a restart
        2 queue        (cars, MAX_DELAY + 1, 2) recent (steer, speed)
                       commands, newest first (see sim.delay_commands)
        3 delay        (cars,) int, each car's reaction delay [steps],
                       0..MAX_DELAY, drawn again when its episode restarts
        4 previous_l   (cars,) [m] lookahead only: the L chosen last step;
                       -1 = none yet (just restarted)
        5 still        (cars,) int, lookahead only: steps in a row spent
                       slower than STALL_SPEED
    step_simulator() updates items 0-3 and drive_lookahead() items 4 and 5,
    which the speed table carries along unchanged. MAX_DELAY is
    sim.MAX_DELAY, which main() sets from --max-delay before calling this.

    Parameters
    ----------
    simulator : JaxSimulator
        From sim.make_simulator(training=True).
    path : numpy.ndarray, shape (points, 2)
        The racing line [m], from sim.racing_line().
    args : argparse.Namespace
        The settings from main(), with SETTINGS[args.method] filled in.
    updates : int
        Number of updates in the whole run, for the learning-rate decay.

    Returns
    -------
    callable
        update(runner) -> (runner, metrics), compiled with jax.jit. Its
        argument is "donated": the GPU memory of the runner passed in is
        reused for the one returned, so the old runner must not be used
        again.
    """
    cars = args.envs
    batch = args.envs * args.rollout
    minibatch = batch // args.minibatches
    points = len(path)
    # The speed table to drive by: --speed-table if given (e.g. the tuned one
    # the driver races with), else whatever the driver would build from
    # DEFAULTS. Either way scaled by DEFAULTS['table_scale'], as in the race.
    speeds = None
    if args.speed_table:
        speeds, line, _ = load_speed_table(args.speed_table)
        if line.shape != path.shape or not np.allclose(line, path, atol=1e-3):
            raise SystemExit(f'{args.speed_table} was tuned on a different racing line')
    inputs, drive = sim.make_lookahead_controller(path, speeds)
    observe, steer_only = sim.make_speed_table_controller(path)

    def step_simulator(carry, steer, speed, key):
        """Send the commands (after the reaction delay) and step every car.

        Each car's new command goes into its delay queue, and the one from
        `delay` steps ago is applied. The simulator then moves every car
        0.01 s. A car that crashed or reached the time limit is put back at
        a random point of the track straight away (autoreset), and its
        queue and delay are reset.

        Parameters
        ----------
        carry : tuple
            The 6-item carry described in make_update(); items 0-3 are used.
        steer : jax.Array, shape (cars,)
            Steering command [rad].
        speed : jax.Array, shape (cars,)
            Speed command [m/s].
        key : jax.Array
            JAX random key (simulator step, restarts and new delays).

        Returns
        -------
        out : f1tenth_gym AutoResetBatchStep
            The step's results: rewards, shape (cars, 1); reset, shape
            (cars,), True where the car was just restarted;
            transition_observation, what each car saw at the end of this
            step (before any restart); metrics, with the crash and
            time-limit flags and the progress [m].
        core : tuple
            Items 0-3 of the new carry.
        """
        env, observation, queue, delay = carry[:4]
        k_step, k_reset, k_delay = jax.random.split(key, 3)
        queue, applied = sim.delay_commands(queue, jnp.stack([steer, speed], axis=1), delay)
        out = simulator.step_batch_autoreset(
            jax.random.split(k_step, cars), jax.random.split(k_reset, cars),
            env, applied[:, None, :])
        # Cars whose episode just ended were put back on the track by the
        # simulator: clear their memory and give them a new random delay.
        reset = out.reset
        queue = jnp.where(reset[:, None, None], 0.0, queue)
        delay = jnp.where(reset, jax.random.randint(k_delay, (cars,), 0, sim.MAX_DELAY + 1), delay)
        return out, (out.state, out.next_observation, queue, delay)

    def sample(params, x, x_end, action, log_prob, reward, out, used,
               l_change=None, stalled=None, point=None, weight=None):
        """Pack one step's data for every car into a Sample.

        Also runs the critic twice: on `x` (its value where the step
        started) and on `x_end` (where the step ended, before any restart).

        Parameters
        ----------
        params : dict
            Current parameters (the critic's are used).
        x, x_end : jax.Array, shape (cars, n)
            Inputs at the start and at the end of the step.
        action, log_prob, reward, used : jax.Array
            As in Sample, one row per car.
        out : f1tenth_gym AutoResetBatchStep
            The simulator step from step_simulator(); its crash and
            time-limit flags and progress are copied.
        l_change, stalled : jax.Array, shape (cars,), optional
            Lookahead only; zeros and False when not given.
        point, weight : jax.Array, shape (cars,), optional
            Speed table only; 0 and 1 when not given.

        Returns
        -------
        Sample
            Every field with one row per car.
        """
        if l_change is None:
            l_change, stalled = jnp.zeros(cars), jnp.zeros(cars, dtype=bool)
        if point is None:
            point, weight = jnp.zeros(cars, dtype=jnp.int32), jnp.ones(cars)
        return Sample(l_change=l_change, stalled=stalled, point=point, weight=weight,
            inputs=x, action=action, log_prob=log_prob,
            value=critic(params['critic'], x), reward=reward,
            terminated=out.metrics.status.terminated, truncated=out.metrics.status.truncated,
            end_value=critic(params['critic'], x_end),
            progress=out.metrics.episode.progress[:, 0], used=used)

    def drive_lookahead(params, carry, key, window):
        """Drive every car one 0.01 s step, with the lookahead policy choosing L.

        The policy's best guess for L (`mean`) gets random noise of spread
        exp(log_std) added (exploration), is clipped to LOOKAHEAD_RANGE
        (0.35-4.0 m), and pure pursuit aims that far ahead. The reward adds
        the paper's three extra terms (see the comment below).

        Parameters
        ----------
        params : dict
            Current parameters.
        carry : tuple
            The 6-item carry described in make_update().
        key : jax.Array
            JAX random key (noise and simulator).
        window : tuple
            Not used here. Both drive functions take it so that either can
            be drive_one_step; only drive_speed_table uses it.

        Returns
        -------
        carry : tuple
            The new 6-item carry.
        sample : Sample
            This step's data, one row per car.
        """
        k_noise, k_sim = jax.random.split(key)
        previous_l, still = carry[4], carry[5]                # still = steps spent standing still
        s = inputs(carry[1])                                  # (cars, 5)
        mean = lookahead(params['actor'], s, xp=jnp)
        chosen = mean + jnp.exp(params['log_std']) * jax.random.normal(k_noise, mean.shape)
        distance = jnp.clip(chosen, *LOOKAHEAD_RANGE)
        steer, speed = drive(carry[1], distance)
        out, core = step_simulator(carry, steer, speed, k_sim)

        # The reward. out.rewards is the simulator's: metres of progress
        # this step, minus COLLISION_PENALTY on touching a wall. Then three
        # terms from the paper's eq. 13:
        #   + w_speed * v              going fast is good
        #   - w_jerk * |L - L_before|  a jumpy lookahead makes jumpy steering
        #   - w_stall                  every step stuck standing still
        v = sim.odometry(out.transition_observation)[2]          # speed after the step [m/s]
        l_change = jnp.where(previous_l < 0, 0.0, jnp.abs(distance - previous_l))
        still = jnp.where(v < STALL_SPEED, still + 1, 0)
        stalled = still > STALL_STEPS
        reward = (out.rewards[:, 0] + args.w_speed * v
                  - args.w_jerk * l_change - args.w_stall * stalled)

        # A car that was just put back on the track has no previous L (-1)
        # and has not been standing still.
        previous_l = jnp.where(out.reset, -1.0, distance)
        still = jnp.where(out.reset, 0, still)
        log_prob = gaussian_log_prob(chosen, mean, params['log_std'], xp=jnp)
        return core + (previous_l, still), sample(
            params, s, inputs(out.transition_observation), chosen, log_prob,
            reward, out, distance, l_change, stalled)

    def drive_speed_table(params, carry, key, window):
        """Drive every car one step at the speed table's speeds.

        Cars inside the window get the table's speed plus their own offset
        for the 2 m piece they are in; every other car drives the table as it is.
        Steering is plain pure pursuit with its own lookahead rule. The
        reward is the simulator's alone.

        Parameters
        ----------
        params : dict
            Current parameters; params['actor']['speeds'] is the table.
        carry : tuple
            The 6-item carry described in make_update().
        key : jax.Array
            JAX random key for the simulator. (The offsets are drawn once
            per update, in update().)
        window : tuple
            (start, offsets). start: int scalar, the window's first point.
            offsets: shape (cars, WINDOW // NOISE_SEGMENT) = (cars, 6)
            [m/s], each car's offset for each 2 m piece of the window.

        Returns
        -------
        carry : tuple
            The new 6-item carry (items 4 and 5 unchanged).
        sample : Sample
            This step's data; `point` is each car's nearest point and
            `weight` is 1 inside the window, 0 outside.
        """
        start, offsets = window
        x, point = observe(carry[1])
        mean = params['actor']['speeds'][point]
        into = (point - start) % points                       # how far into the window
        inside = into < WINDOW
        offset = offsets[jnp.arange(cars), jnp.minimum(into // NOISE_SEGMENT, offsets.shape[1] - 1)]
        chosen = jnp.where(inside, mean + offset, mean)
        speed = jnp.clip(chosen, *SPEED_RANGE)
        out, core = step_simulator(carry, steer_only(carry[1]), speed, key)
        x_end, _ = observe(out.transition_observation)
        log_prob = gaussian_log_prob(chosen, mean, params['log_std'], xp=jnp)
        return core + carry[4:], sample(
            params, x, x_end, chosen, log_prob, out.rewards[:, 0], out, speed,
            point=point, weight=inside.astype(jnp.float32))

    drive_one_step = {'lookahead': drive_lookahead,
                      'speed_table': drive_speed_table}[args.method]

    def loss(params, mb):
        """Work out the loss that gradient descent makes smaller, for one minibatch.

        The loss is the policy's (policies.py: lookahead_loss or
        speed_table_loss) plus the critic's squared error, weighted by
        value_coef.

        Parameters
        ----------
        params : dict
            Current parameters.
        mb : Minibatch
            One minibatch; every field's first axis has length
            batch / minibatches.

        Returns
        -------
        total : scalar jax.Array
            policy loss + value_coef x value loss.
        aux : tuple of scalar jax.Array
            (value_loss, kl): the critic's error (half the mean squared
            difference from the return), and the KL between the policy
            that drove the rollout and the current one, on this minibatch.
            jax.value_and_grad(..., has_aux=True) passes these through
            without taking their gradient.
        """
        if args.method == 'lookahead':
            policy_loss, kl = lookahead_loss(
                params['actor'], params['log_std'], mb.inputs, mb.action, mb.log_prob,
                mb.advantage, args.clip, args.entropy, xp=jnp)
        else:
            # Speed table. The exploration spread stays fixed (stop_gradient:
            # no gradient, so Adam never moves it). Learned, PPO shrinks it
            # to nothing within a run and the table stops exploring halfway
            # through.
            policy_loss, kl = speed_table_loss(
                params['actor']['speeds'], jax.lax.stop_gradient(params['log_std']), mb.point,
                mb.action, mb.log_prob, mb.advantage, mb.weight, args.clip, args.entropy, xp=jnp)
        value_loss = 0.5 * jnp.mean((critic(params['critic'], mb.inputs) - mb.ret) ** 2)
        return policy_loss + args.value_coef * value_loss, (value_loss, kl)

    def update(runner):
        """Do one whole PPO update: drive, score, learn, measure.

        See make_update() for the steps and for the runner's layout.

        Parameters
        ----------
        runner : tuple
            (params, adam, carry, key, index).

        Returns
        -------
        runner : tuple
            The same five items after the update; index is one higher.
        metrics : dict of scalar jax.Array
            For log.csv. All methods: track_speed, average speed of
            progress along the centreline [m/s]; crashes_per_lap;
            noise_std, the exploration spread exp(log_std); value_loss and
            kl, averaged over every minibatch step; stopped_early, 1 if the
            KL stop fired. Lookahead adds L_straight, L_curve and L_mean
            [m] (L on straights, in tight bends, overall), L_change [m] and
            stalled (share of steps). Speed table adds window_start, inside
            (share of samples in the window), window_change and
            window_shift [m/s].
        """
        params, adam, carry, key, index = runner
        key, k_drive, k_shuffle = jax.random.split(key, 3)

        # Speed table only: this update's window (its first point), the
        # points in it, and each car's offset for each 2 m piece of it.
        start = offsets = tuned = speeds_before = None
        if args.method == 'speed_table':
            k_drive, k_window, k_offsets = jax.random.split(k_drive, 3)
            start = jax.random.randint(k_window, (), 0, points)
            tuned = (jnp.arange(points) - start) % points < WINDOW
            offsets = jnp.exp(params['log_std']) * jax.random.normal(
                k_offsets, (cars, WINDOW // NOISE_SEGMENT))
            speeds_before = params['actor']['speeds']

        # 1. Drive: `rollout` steps for every car.
        carry, samples = jax.lax.scan(
            lambda c, k: drive_one_step(params, c, k, (start, offsets)), carry,
            jax.random.split(k_drive, args.rollout))

        # 2. Score every step against the critic's expectation, then rescale
        #    the advantages to average 0 and spread 1 over the samples that
        #    count (for the speed table, only those inside the window).
        advantage, ret = advantages_and_returns(samples, args.gamma, args.gae_lambda)
        if args.method == 'speed_table':
            w = samples.weight
            count = jnp.maximum(jnp.sum(w), 1.0)
            average = jnp.sum(w * advantage) / count
            spread = jnp.sqrt(jnp.sum(w * (advantage - average) ** 2) / count)
            advantage = (advantage - average) / (spread + 1e-8)
        else:
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
        flat = jax.tree.map(lambda a: a.reshape((batch,) + a.shape[2:]), samples)
        data = Minibatch(flat.inputs, flat.action, flat.log_prob,
                         advantage.reshape(batch), ret.reshape(batch), flat.point, flat.weight)

        # 3. Learn: a few passes over the data in shuffled minibatches.
        #    With lr_decay the learning rate falls in a straight line to 0
        #    over the run, so late updates make only small, careful changes.
        lr = args.lr * (1.0 - index / updates) if args.lr_decay else args.lr

        def one_minibatch(state, mb):
            """Take one Adam step on one minibatch (run by jax.lax.scan).

            Once `stopped` is True the step is still worked out (a compiled
            loop cannot skip work) but thrown away: the old parameters and
            Adam memory are kept.

            Parameters
            ----------
            state : tuple
                (params, adam, stopped): current parameters, AdamState, and
                a bool scalar that is True once the KL early stop has fired
                in this update.
            mb : Minibatch
                One minibatch; first axis of length batch / minibatches.

            Returns
            -------
            state : tuple
                (params, adam, stopped) after the step.
            output : tuple of scalar jax.Array
                (value_loss, kl) on this minibatch, from before the step.
            """
            params, adam, stopped = state
            (_, (value_loss, kl)), grads = jax.value_and_grad(loss, has_aux=True)(params, mb)
            # Target KL (as Stable-Baselines3 does it): once the policy has
            # moved too far from the one that collected this data, skip the
            # rest of this update. kl is measured before the step is taken.
            if args.target_kl is not None:
                stopped = stopped | (kl > 1.5 * args.target_kl)
            new_params, new_adam = adam_step(params, clip_by_global_norm(grads, MAX_GRAD_NORM), adam, lr)
            if args.method == 'speed_table':
                # Only the window's speeds may change. (Points outside it get
                # no gradient, but Adam's running average from earlier
                # updates would still nudge them.)
                new_params = {**new_params, 'actor': {'speeds': jnp.where(
                    tuned, new_params['actor']['speeds'], params['actor']['speeds'])}}
            keep_old = lambda new, old: jax.tree.map(lambda a, b: jnp.where(stopped, b, a), new, old)
            return (keep_old(new_params, params), keep_old(new_adam, adam), stopped), (value_loss, kl)

        def one_epoch(state, key):
            """Make one pass (epoch) over all the update's data, in shuffled minibatches.

            Parameters
            ----------
            state : tuple
                (params, adam, stopped), as in one_minibatch().
            key : jax.Array
                JAX random key for the shuffle.

            Returns
            -------
            state : tuple
                (params, adam, stopped) after every minibatch.
            output : tuple of jax.Array, each shape (minibatches,)
                value_loss and kl of every minibatch.
            """
            order = jax.random.permutation(key, batch)
            mbs = jax.tree.map(lambda a: a[order].reshape((args.minibatches, minibatch) + a.shape[1:]), data)
            return jax.lax.scan(one_minibatch, state, mbs)

        (params, adam, stopped), (value_loss, kl) = jax.lax.scan(
            one_epoch, (params, adam, jnp.asarray(False)), jax.random.split(k_shuffle, args.epochs))

        metrics = {
            'track_speed': jnp.mean(samples.progress) / sim.TIMESTEP,
            # Progress is measured along the centreline, so 78.5 m of it (the
            # centreline's length, 78.49 m) is one lap. The racing line
            # itself is shorter: 75.9 m.
            'crashes_per_lap': jnp.sum(samples.terminated)
                               / jnp.maximum(jnp.sum(samples.progress) / 78.5, 1e-6),
            'noise_std': jnp.mean(jnp.exp(params['log_std'])),
            'value_loss': jnp.mean(value_loss),
            'kl': jnp.mean(kl),
            'stopped_early': stopped.astype(jnp.float32),
        }
        if args.method == 'lookahead':
            # Mean L where the line ahead is nearly straight, and where it bends hard.
            sharpest = jnp.max(samples.inputs[..., 1:4], axis=-1)
            for name, where in (('L_straight', sharpest < 0.2), ('L_curve', sharpest > 0.8)):
                metrics[name] = jnp.sum(samples.used * where) / jnp.maximum(jnp.sum(where), 1)
            metrics['L_mean'] = jnp.mean(samples.used)
            metrics['L_change'] = jnp.mean(samples.l_change)
            metrics['stalled'] = jnp.mean(samples.stalled)
        else:
            change = params['actor']['speeds'] - speeds_before
            metrics['window_start'] = start.astype(jnp.float32)
            metrics['inside'] = jnp.mean(samples.weight)              # share of samples in the window
            metrics['window_change'] = jnp.sum(jnp.abs(change)) / WINDOW   # mean |change| [m/s]
            metrics['window_shift'] = jnp.sum(change) / WINDOW             # mean change, signed
        return (params, adam, carry, key, index + 1), metrics

    # Compile update() into one GPU program. donate_argnums=0: the runner
    # passed in may be overwritten, so main() only ever uses the one returned.
    return jax.jit(update, donate_argnums=0)


# ----------------------------------------------------------------------
# Saving and loading
# ----------------------------------------------------------------------
def save(path, params, args, line=None):
    """Write the policy to an .npz file the driver can load with numpy alone.

    `line` is the racing line (speed table only: saved with it, so the driver
    can check the table still fits the line it builds).

    Every file holds log_std, the critic's layers (critic_W0 ... critic_b2,
    so --init can continue training), DEFAULTS at training time
    (base_param_names / base_param_values, so the driver and evaluate.py
    can warn when pure pursuit's settings have changed since) and the run's
    settings as JSON (args). Then, by method: lookahead w (5,), b,
    input_names, lookahead_range; speed table speeds (points,) [m/s], line
    (points, 2) [m], speed_range.

    Parameters
    ----------
    path : str or pathlib.Path
        File to write (normally training/runs/<date-time>/<method>.npz).
    params : dict
        Parameters from init_params() or training.
    args : argparse.Namespace
        The run's settings; args.method picks the layout.
    line : numpy.ndarray, shape (points, 2), optional
        Speed table only: the racing line [m] the table was tuned on.
    """
    common = dict(
        log_std=np.asarray(params['log_std']),
        **{'critic_' + k: np.asarray(v) for k, v in params['critic'].items()},
        base_param_names=np.array(list(DEFAULTS)),
        base_param_values=np.array(list(DEFAULTS.values()), dtype=float),
        args=np.array(json.dumps(vars(args), default=str)))
    if args.method == 'lookahead':
        np.savez(path, w=np.asarray(params['actor']['w']), b=np.asarray(params['actor']['b']),
                 input_names=np.array(LOOKAHEAD_INPUTS),
                 lookahead_range=np.asarray(LOOKAHEAD_RANGE), **common)
    else:
        np.savez(path, speeds=np.asarray(params['actor']['speeds'], dtype=float),
                 line=np.asarray(line), speed_range=np.asarray(SPEED_RANGE), **common)


def load_params(path, method):
    """Read a saved policy back in, to continue training from it (--init).

    The critic and log_std are read back too. Adam's memory is not saved,
    so it starts again from zero.

    Parameters
    ----------
    path : str or pathlib.Path
        An .npz written by save().
    method : str
        'lookahead' or 'speed_table'; must match the file.

    Returns
    -------
    dict
        'actor', 'critic' and 'log_std', laid out as by init_params().
    """
    data = np.load(path)
    names = ('W0', 'b0', 'W1', 'b1', 'W2', 'b2')
    if method == 'lookahead':
        policy = {'w': jnp.asarray(data['w']), 'b': jnp.asarray(data['b'])}
    else:
        policy = {'speeds': jnp.asarray(data['speeds'], dtype=jnp.float32)}
    return {'actor': policy,
            'critic': {k: jnp.asarray(data['critic_' + k]) for k in names},
            'log_std': jnp.asarray(data['log_std'])}


def keep_within_rule(params, line, device):
    """Make the tuned speed table obey the table's rules again.

    Speed table, after every update: clip the speeds to SPEED_RANGE, then
    lower them wherever neighbouring points break the speed-change rule
    (raceline.limit_speed_changes, the same code that built the starting
    table). Done here in numpy rather than inside the compiled update, so
    there is only one copy of the rule.

    Parameters
    ----------
    params : dict
        Parameters; params['actor']['speeds'] has shape (points,) [m/s].
    line : numpy.ndarray, shape (points, 2)
        The racing line [m].
    device : jax.Device
        Where to put the corrected table (the training GPU).

    Returns
    -------
    params : dict
        The same parameters with the corrected table (float32, on `device`).
    speeds : numpy.ndarray, shape (points,)
        The corrected table [m/s], for working out the table lap time.
    """
    speeds = np.clip(np.asarray(params['actor']['speeds'], dtype=float), *SPEED_RANGE)
    speeds = limit_speed_changes(line, speeds, DEFAULTS['a_brake'])
    actor = {'speeds': jax.device_put(jnp.asarray(speeds, dtype=jnp.float32), device)}
    return {**params, 'actor': actor}, speeds


def equation(policy):
    """Write the lookahead policy as readable text.

    Parameters
    ----------
    policy : dict
        'w', shape (5,), and 'b', the lookahead policy's weights and bias.

    Returns
    -------
    str
        All five terms to three decimals; for the shipped lookahead.npz,
        'L = 0.004 + 0.249*v + 0.003*|k0| + 0.001*|k1| + 0.004*|k2| - 0.001*dk'.
    """
    terms = ' '.join(f'{"-" if w < 0 else "+"} {abs(w):.3f}*{name}'
                     for w, name in zip(np.asarray(policy['w']), LOOKAHEAD_INPUTS))
    return f'L = {float(policy["b"]):.3f} {terms}'


def main():
    """Read the command line, train a policy, and save it.

    In order: fill every option not given on the command line from
    SETTINGS[--method] (--smoke shrinks the run to 10 updates of 64 cars x
    64 steps); make training/runs/<date-time>/ and write args.json; build
    the simulator, the racing line and the compiled update; start from
    --init or from init_params() (the speed table from the table worked out
    from mu and a_brake); then run the updates. After each one: speed table
    only, keep_within_rule(); add a row to log.csv; print a line every 5
    updates; save the policy every 20 updates and at the end.

    --speed-table only affects --method lookahead (it is the table it
    drives by). --method speed_table ignores it; use --init to
    continue tuning a saved table.
    """
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--method', choices=('lookahead', 'speed_table'),
                        default='lookahead')
    parser.add_argument('--steps', type=float, help='simulated steps in total')
    parser.add_argument('--envs', type=int, default=2048, help='cars driving in parallel')
    parser.add_argument('--rollout', type=int, default=128, help='steps per car between updates')
    parser.add_argument('--epochs', type=int, help='passes over each batch of data')
    parser.add_argument('--minibatches', type=int, help='pieces each pass is split into')
    parser.add_argument('--hidden', type=int, default=128, help="neurons per hidden layer of the critic")
    parser.add_argument('--lr', type=float, help='learning rate')
    parser.add_argument('--lr-decay', action=argparse.BooleanOptionalAction,
                        help='lower the learning rate linearly to 0 over the run')
    parser.add_argument('--gamma', type=float,
                        help='discount: how much future reward counts (0.99 = about 1 s ahead)')
    parser.add_argument('--gae-lambda', type=float)
    parser.add_argument('--clip', type=float, default=0.2)
    parser.add_argument('--entropy', type=float, help='bonus for keeping exploration noise')
    parser.add_argument('--value-coef', type=float, help="weight of the critic's error in the loss")
    parser.add_argument('--target-kl', type=float, help='stop an update early past this KL')
    parser.add_argument('--episode-steps', type=int, default=3000, help='time limit per episode (30 s)')
    parser.add_argument('--w-speed', type=float, default=0.002,
                        help='lookahead reward: bonus per step per m/s of speed')
    parser.add_argument('--w-jerk', type=float, default=0.01,
                        help='lookahead reward: cost per step per metre L changes by')
    parser.add_argument('--w-stall', type=float, default=0.05,
                        help='lookahead reward: cost per step spent stalled')
    parser.add_argument('--init', help='saved policy (.npz) to continue training from')
    parser.add_argument('--speed-table',
                        help='lookahead: drive by this speed_table.npz (default: the '
                             "table the driver would build from DEFAULTS)")
    parser.add_argument('--max-delay', type=int, default=sim.MAX_DELAY,
                        help='largest reaction delay [steps]; each car gets 0..this')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', choices=('gpu', 'cpu'), default='gpu')
    parser.add_argument('--smoke', action='store_true', help='tiny run to check the setup works')
    args = parser.parse_args()
    sim.MAX_DELAY = args.max_delay
    for name, value in SETTINGS[args.method].items():
        if getattr(args, name) is None:
            setattr(args, name, value)
    if args.smoke:
        args.steps, args.envs, args.rollout = 64 * 64 * 10, 64, 64
    if (args.envs * args.rollout) % args.minibatches:
        parser.error('--envs x --rollout must divide evenly into --minibatches')

    run_dir = Path(__file__).resolve().parent / 'runs' / datetime.now().strftime('%Y%m%d-%H%M%S')
    run_dir.mkdir(parents=True)
    (run_dir / 'args.json').write_text(json.dumps(vars(args), indent=2))
    out_file = run_dir / f'{args.method}.npz'

    device = jax.devices(args.device)[0]
    simulator = sim.make_simulator(device, training=True, episode_steps=args.episode_steps,
                                   collision_penalty=COLLISION_PENALTY)
    updates = max(1, int(args.steps // (args.envs * args.rollout)))
    path = sim.racing_line()
    update = make_update(simulator, path, args, updates)

    key = jax.random.key(args.seed)
    key, k_params, k_reset, k_delay = jax.random.split(key, 4)
    start_speeds = None
    if args.method == 'speed_table':
        # Start from the table the driver works out from mu and a_brake.
        start_speeds = speed_table(path, path_geometry(path)[1], DEFAULTS)
        if args.init and np.load(args.init)['line'].shape != path.shape:
            parser.error(f'{args.init} was tuned on a different racing line')
    params = (load_params(args.init, args.method) if args.init
              else init_params(k_params, args.method, args.hidden, start_speeds))
    observation, env = simulator.reset_batch(jax.random.split(k_reset, args.envs))
    carry = (env, observation,
             jnp.zeros((args.envs, sim.MAX_DELAY + 1, 2)),
             jax.random.randint(k_delay, (args.envs,), 0, sim.MAX_DELAY + 1),
             jnp.full(args.envs, -1.0), jnp.zeros(args.envs, dtype=jnp.int32))
    runner = (params, adam_init(params), carry, key, jnp.asarray(0))

    print(f'{args.method}: {updates} updates of {args.envs} cars x {args.rollout} steps on {device}. '
          f'The first one compiles, which takes a minute or two.\nSaving to {run_dir}\n')
    if args.method == 'lookahead':
        print(f'start: {equation(params["actor"])}\n')
    if args.method == 'speed_table':
        params, speeds = keep_within_rule(params, path, device)
        runner = (params,) + runner[1:]
        print(f'start: table lap {lap_time(path, speeds):.2f} s if driven exactly, '
              f'{WINDOW} of {len(path)} points tuned per update\n')
    with open(run_dir / 'log.csv', 'w', newline='') as log_file:
        log = None
        started = time.time()
        for i in range(1, updates + 1):
            runner, metrics = update(runner)
            m = {k: float(v) for k, v in jax.device_get(metrics).items()}
            if args.method == 'speed_table':
                params, speeds = keep_within_rule(runner[0], path, device)
                runner = (params,) + runner[1:]
                m['table_lap'] = lap_time(path, speeds)
            if args.method == 'lookahead':
                policy = jax.device_get(runner[0]['actor'])
                m.update({f'w_{name}': float(w) for name, w in zip(LOOKAHEAD_INPUTS, policy['w'])})
                m['b'] = float(policy['b'])
            steps = i * args.envs * args.rollout
            if i == 1:                     # the first update includes compiling
                started, compiled_steps = time.time(), steps
            rate = (steps - compiled_steps) / max(time.time() - started, 1e-9)

            if log is None:
                log = csv.writer(log_file)
                log.writerow(['update', 'steps_M', 'steps_per_s'] + list(m))
            log.writerow([i, steps / 1e6, rate] + list(m.values()))
            log_file.flush()

            if i == 1 or i % 5 == 0 or i == updates:
                line = (f'{i:5d}  {steps / 1e6:6.2f}M steps  {rate:6.0f}/s  '
                        f'speed {m["track_speed"]:.2f} m/s  crashes/lap {m["crashes_per_lap"]:.3f}  '
                        f'noise {m["noise_std"]:.2f}')
                if args.method == 'lookahead':
                    line += (f'  L straight {m["L_straight"]:.2f} m, curve {m["L_curve"]:.2f} m'
                             f'  stalled {100 * m["stalled"]:.1f}%'
                             f'\n       {equation(policy)}')
                else:
                    first = int(m['window_start'])
                    line += (f'  window {first}-{(first + WINDOW - 1) % len(path)}'
                             f' ({100 * m["inside"]:.0f}% of cars)'
                             f'  change {m["window_shift"]:+.3f} m/s (|{m["window_change"]:.3f}|)'
                             f'  table lap {m["table_lap"]:.2f} s')
                print(line)
            if i % 20 == 0 or i == updates:
                save(out_file, runner[0], args, path)

    flag = {'lookahead': '--lookahead', 'speed_table': '--speed-table'}[args.method]
    print(f'\nDone. Policy: {out_file}\nNext: python training/evaluate.py {flag} {out_file}')


if __name__ == '__main__':
    main()
