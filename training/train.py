"""Train a policy for the driver with PPO (written out by hand, in JAX) in the GPU simulator.

--method lookahead (default) learns pure pursuit's lookahead L; --method speed_table tunes the
speed table, one speed per racing-line point. Both policies are in team_driver/policies.py.

    python training/train.py --smoke                # 2-minute check that everything runs
    python training/train.py                        # --method lookahead, 20M steps: 4-7 minutes
    python training/train.py --method speed_table   # 40M steps: 8-13 minutes

"python" = ~/venvs/rl/bin/python (made by training/setup_venv.sh); training/README.md explains
the methods. Every run writes training/runs/<date-time>/:
    lookahead.npz or speed_table.npz   the policy, saved every 20 updates and at the end
    log.csv                            one row per update, for plotting
    args.json                          the settings used, so the run can be repeated
"""

import argparse
import csv
import json
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

import sim
from team_driver.pure_pursuit import DEFAULTS
from team_driver.raceline import lap_time, limit_speed_changes, speed_table
from team_driver.policies import (
    LOOKAHEAD_INPUTS, LOOKAHEAD_RANGE, START_BIAS, START_WEIGHTS, gaussian_log_prob,
    load_speed_table, lookahead, lookahead_loss, path_geometry, speed_table_loss)
from team_driver.units import Array

COLLISION_PENALTY = 10.0   # reward lost on touching a wall [metres of progress]
MAX_GRAD_NORM = 0.5        # gradients longer than this are shrunk (see clip_by_global_norm)
STALL_SPEED = 0.05         # [m/s] slower than this counts as standing still (the paper's value)
STALL_STEPS = 50           # stalled = standing still for longer than this many steps (0.5 s)

# Speed table only.
WINDOW = 60                # consecutive points tuned per update (12 m of line)
NOISE_SEGMENT = 10         # a car's exploration offset is held for this many points (2 m)
SPEED_RANGE = (1.0, 15.0)  # [m/s] tuned speeds are kept inside this
SPEED_TABLE_INPUTS = 7     # what its critic sees: policies.speed_table_inputs

# PPO settings for each method; any of them can be overridden on the command line.
SETTINGS = {
    # The paper's settings (Elgouhary & El-Wakeel 2026, section III-J).
    'lookahead': dict(steps=20e6, epochs=5, minibatches=32, lr=2.3927e-4, lr_decay=True,
                      gamma=0.99, gae_lambda=0.98, entropy=0.02, value_coef=0.6, target_kl=0.015),
    # A point is in the window only ~1 update in 6, hence more steps and a larger lr.
    # entropy does nothing here: the exploration spread is fixed (see loss()).
    'speed_table': dict(steps=40e6, epochs=5, minibatches=32, lr=1e-3, lr_decay=True,
                        gamma=0.99, gae_lambda=0.95, entropy=0.0, value_coef=0.5, target_kl=0.015),
}


class Sample(NamedTuple):
    """What PPO keeps from one step of every car.

    Each field is (cars,) or (cars, n); jax.lax.scan stacks a rollout to (rollout, cars[, n]).
    """
    inputs: jax.Array          # (cars, n) what policy and critic saw; n = 5 lookahead, 7 speed table
    action: jax.Array          # what the policy picked, noise included, unclipped: L [m] or speed [m/s]
    log_prob: jax.Array        # log-probability of `action`
    value: jax.Array           # the critic's prediction where the step started
    reward: jax.Array          # progress [m] - COLLISION_PENALTY on a crash (+ the lookahead's terms)
    terminated: jax.Array      # episode ended by a crash
    truncated: jax.Array       # episode ended by the time limit
    end_value: jax.Array       # the critic's prediction where the step ended up
    progress: jax.Array        # [m] along the track this step
    used: jax.Array            # what reached the car, after clipping: L [m] or speed [m/s]
    l_change: jax.Array        # lookahead: |L - previous L| [m] (speed table: 0)
    stalled: jax.Array         # lookahead: standing still too long (speed table: False)
    point: jax.Array           # speed table: the path point the car was nearest to (others: 0)
    weight: jax.Array          # 1 if it counts in the policy loss (speed table: inside the window)


class Minibatch(NamedTuple):
    """What the learning step needs from each sample.

    update() flattens a rollout to batch = envs x rollout rows (262,144 by default);
    one_epoch() shuffles that and cuts it into `minibatches` pieces (8,192 rows each).
    """
    inputs: jax.Array          # (batch, n) as Sample.inputs
    action: jax.Array          # as Sample.action
    log_prob: jax.Array        # of `action`, under the policy that drove the rollout
    advantage: jax.Array       # how much better than expected it went (rescaled: mean 0, spread 1)
    ret: jax.Array             # raw advantage + value: what the critic is trained towards
    point: jax.Array           # int, as Sample.point
    weight: jax.Array          # as Sample.weight


# ----------------------------------------------------------------------
# The critic: used only in training, it guesses the reward still to come.
# ----------------------------------------------------------------------
def network(key: jax.Array, n_in: int, hidden: int, n_out: int, last_scale: float) -> dict:
    """Random starting weights for n_in -> hidden -> hidden -> n_out (keys W0, b0 ... W2, b2)."""
    keys = jax.random.split(key, 3)
    # (rows, columns, scale): orthogonal start, as in f1tenth_gym's jax_ppo_training.py.
    sizes = [(n_in, hidden, math.sqrt(2.0)), (hidden, hidden, math.sqrt(2.0)),
             (hidden, n_out, last_scale)]
    weights = {}
    for i, (k, (a, b, scale)) in enumerate(zip(keys, sizes)):
        weights[f'W{i}'] = jax.nn.initializers.orthogonal(scale)(k, (a, b))
        weights[f'b{i}'] = jnp.zeros(b)
    return weights


def init_params(key: jax.Array, method: str, hidden: int, speeds: Array | None = None) -> dict:
    """Starting {'actor', 'critic', 'log_std'}; `speeds` [m/s] is the speed table's starting table."""
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


def critic(weights: dict, x: jax.Array) -> jax.Array:
    """The critic's guess of the reward still to come, one per row of `x`: (..., n_in) -> (...)."""
    h = jnp.tanh(x @ weights['W0'] + weights['b0'])
    h = jnp.tanh(h @ weights['W1'] + weights['b1'])
    return (h @ weights['W2'] + weights['b2'])[..., 0]


def advantages_and_returns(s: Sample, gamma: float, lam: float) -> tuple[jax.Array, jax.Array]:
    """Each step's advantage and return by GAE; arrays are (rollout, cars).

    A crash gets no future value; a time limit does (the car would have kept driving).
    """
    bootstrap = jnp.where(s.terminated, 0.0, s.end_value)
    deltas = s.reward + gamma * bootstrap - s.value
    continues = ~(s.terminated | s.truncated)

    # advantage[t] = delta[t] + gamma * lam * advantage[t + 1], so it is built last step first;
    # the sum stops where an episode ended (that car's next step is a new episode).
    def backwards(next_advantage: jax.Array, inputs: tuple) -> tuple[jax.Array, jax.Array]:
        """One step of the backwards pass: this step's advantage, as both carry and output."""
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
    """What Adam remembers between steps."""
    step: jax.Array            # steps taken so far
    m: dict                    # running average of the gradient, shaped like the parameters
    v: dict                    # running average of the squared gradient, likewise


def adam_init(params: dict) -> AdamState:
    """Adam's starting memory: step 0, both running averages zero."""
    return AdamState(jnp.zeros((), jnp.int32),
                     jax.tree.map(jnp.zeros_like, params), jax.tree.map(jnp.zeros_like, params))


def adam_step(params: dict, grads: dict, state: AdamState, lr: float | jax.Array,
              b1: float = 0.9, b2: float = 0.999, eps: float = 1e-5) -> tuple[dict, AdamState]:
    """One step of Adam (Kingma & Ba 2015): each parameter moves about `lr`, big gradient or small."""
    step = state.step + 1
    m = jax.tree.map(lambda m, g: b1 * m + (1 - b1) * g, state.m, grads)
    v = jax.tree.map(lambda v, g: b2 * v + (1 - b2) * g * g, state.v, grads)
    # Both averages start at zero, so early on they are too small; dividing by (1 - b^step) fixes it.
    m_fix, v_fix = 1.0 / (1.0 - b1 ** step), 1.0 / (1.0 - b2 ** step)
    params = jax.tree.map(lambda p, m, v: p - lr * (m * m_fix) / (jnp.sqrt(v * v_fix) + eps),
                          params, m, v)
    return params, AdamState(step, m, v)


def clip_by_global_norm(grads: dict, max_norm: float) -> dict:
    """Shrink the gradient (all parameters as one long vector) to length at most `max_norm`."""
    norm = jnp.sqrt(sum(jnp.sum(g * g) for g in jax.tree.leaves(grads)))
    scale = jnp.minimum(1.0, max_norm / (norm + 1e-8))
    return jax.tree.map(lambda g: g * scale, grads)


# ----------------------------------------------------------------------
# One update: drive, then learn.
# ----------------------------------------------------------------------
def make_update(simulator: sim.JaxSimulator, path: Array, args: argparse.Namespace,
                updates: int) -> Callable[[tuple], tuple[tuple, dict]]:
    """Build update(runner) -> (runner, metrics): one whole PPO update, compiled with jax.jit.

    The runner passed in is donated (its GPU memory is reused), so only use the one returned.
    `path` is the racing line (points, 2) [m]; `updates` is the run's total, for the lr decay.
    """
    # runner = (params, adam, carry, key, index):
    #   params        dict from init_params(): 'actor', 'critic', 'log_std'
    #   adam          AdamState for params
    #   carry         every car's state between steps (below)
    #   key           JAX random key
    #   index         updates done so far (int scalar), for the learning-rate decay
    # carry = 6-tuple, one row per car (step_simulator() updates 0-3, drive_lookahead() 4-5):
    #   0 env         simulator state of every car (f1tenth_gym BatchState)
    #   1 observation what each car sees now (LiDAR scan, car state); fresh after a restart
    #   2 queue       (cars, sim.MAX_DELAY + 1, 2) recent (steer, speed) commands, newest first
    #   3 delay       (cars,) int reaction delay [steps], 0..sim.MAX_DELAY (main() sets it
    #                 from --max-delay), drawn again on a restart
    #   4 previous_l  (cars,) [m] lookahead only: last step's L; -1 = none yet (just restarted)
    #   5 still       (cars,) int, lookahead only: steps in a row slower than STALL_SPEED
    cars = args.envs
    batch = args.envs * args.rollout
    minibatch = batch // args.minibatches
    points = len(path)
    # Lookahead drives by --speed-table if given, else the table built from DEFAULTS
    # (sim.py scales either by DEFAULTS['table_scale'], as in the race).
    speeds = None
    if args.speed_table:
        speeds, line, _ = load_speed_table(args.speed_table)
        if line.shape != path.shape or not np.allclose(line, path, atol=1e-3):
            raise SystemExit(f'{args.speed_table} was tuned on a different racing line')
    inputs, drive = sim.make_lookahead_controller(path, speeds)
    observe, steer_only = sim.make_speed_table_controller(path)

    def step_simulator(carry: tuple, steer: jax.Array, speed: jax.Array,
                       key: jax.Array) -> tuple[Any, tuple]:
        """Queue each car's command, apply the one from `delay` steps ago, and step every car 0.01 s.

        steer [rad] and speed [m/s] are (cars,); returns (out, carry items 0-3). Crashed or timed-out
        cars restart at once (autoreset); out.transition_observation is what they saw before that.
        """
        env, observation, queue, delay = carry[:4]
        k_step, k_reset, k_delay = jax.random.split(key, 3)
        queue, applied = sim.delay_commands(queue, jnp.stack([steer, speed], axis=1), delay)
        out = simulator.step_batch_autoreset(
            jax.random.split(k_step, cars), jax.random.split(k_reset, cars),
            env, applied[:, None, :])
        # Restarted cars: empty queue and a new random delay.
        reset = out.reset
        queue = jnp.where(reset[:, None, None], 0.0, queue)
        delay = jnp.where(reset, jax.random.randint(k_delay, (cars,), 0, sim.MAX_DELAY + 1), delay)
        return out, (out.state, out.next_observation, queue, delay)

    def sample(params: dict, x: jax.Array, x_end: jax.Array, action: jax.Array,
               log_prob: jax.Array, reward: jax.Array, out: Any, used: jax.Array,
               l_change: jax.Array | None = None, stalled: jax.Array | None = None,
               point: jax.Array | None = None, weight: jax.Array | None = None) -> Sample:
        """Pack one step of every car into a Sample; the critic is run at the step's start and end."""
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

    def drive_lookahead(params: dict, carry: tuple, key: jax.Array,
                        window: tuple) -> tuple[tuple, Sample]:
        """Drive every car one step, the policy choosing L (plus noise, clipped to LOOKAHEAD_RANGE).

        `window` is unused: it is there so both drive functions can be drive_one_step.
        """
        k_noise, k_sim = jax.random.split(key)
        previous_l, still = carry[4], carry[5]                # still = steps spent standing still
        s = inputs(carry[1])                                  # (cars, 5)
        mean = lookahead(params['actor'], s, xp=jnp)
        chosen = mean + jnp.exp(params['log_std']) * jax.random.normal(k_noise, mean.shape)
        distance = jnp.clip(chosen, *LOOKAHEAD_RANGE)
        steer, speed = drive(carry[1], distance)
        out, core = step_simulator(carry, steer, speed, k_sim)

        # Reward: the simulator's (progress, minus COLLISION_PENALTY) plus three of the paper's
        # eq. 13 terms: + w_speed * v, - w_jerk * |L - previous L| (jumpy L, jumpy steering), - w_stall.
        v = sim.odometry(out.transition_observation)[2]          # speed after the step [m/s]
        l_change = jnp.where(previous_l < 0, 0.0, jnp.abs(distance - previous_l))
        still = jnp.where(v < STALL_SPEED, still + 1, 0)
        stalled = still > STALL_STEPS
        reward = (out.rewards[:, 0] + args.w_speed * v
                  - args.w_jerk * l_change - args.w_stall * stalled)

        # Restarted cars: no previous L (-1), not standing still.
        previous_l = jnp.where(out.reset, -1.0, distance)
        still = jnp.where(out.reset, 0, still)
        log_prob = gaussian_log_prob(chosen, mean, params['log_std'], xp=jnp)
        return core + (previous_l, still), sample(
            params, s, inputs(out.transition_observation), chosen, log_prob,
            reward, out, distance, l_change, stalled)

    def drive_speed_table(params: dict, carry: tuple, key: jax.Array,
                          window: tuple) -> tuple[tuple, Sample]:
        """Drive every car one step at the table's speed, plus its offset if inside the window.

        window = (start, offsets): the window's first point, and (cars, WINDOW // NOISE_SEGMENT)
        offsets [m/s], one per car per 2 m piece, drawn once per update in update().
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

    def loss(params: dict, mb: Minibatch) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
        """Policy loss + value_coef x critic error on one minibatch -> (total, (value_loss, kl))."""
        if args.method == 'lookahead':
            policy_loss, kl = lookahead_loss(
                params['actor'], params['log_std'], mb.inputs, mb.action, mb.log_prob,
                mb.advantage, args.clip, args.entropy, xp=jnp)
        else:
            # log_std is held fixed (stop_gradient): when it was learned, PPO shrank it to
            # nothing within a run and the table stopped exploring halfway through.
            policy_loss, kl = speed_table_loss(
                params['actor']['speeds'], jax.lax.stop_gradient(params['log_std']), mb.point,
                mb.action, mb.log_prob, mb.advantage, mb.weight, args.clip, args.entropy, xp=jnp)
        value_loss = 0.5 * jnp.mean((critic(params['critic'], mb.inputs) - mb.ret) ** 2)
        return policy_loss + args.value_coef * value_loss, (value_loss, kl)

    def update(runner: tuple) -> tuple[tuple, dict]:
        """One whole PPO update: drive, score, learn, measure (metrics: one log.csv row)."""
        params, adam, carry, key, index = runner
        key, k_drive, k_shuffle = jax.random.split(key, 3)

        # Speed table only: this update's window start, its points, and each car's offsets.
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

        # 2. Score: GAE, then rescale the advantages to mean 0, spread 1 over the samples
        #    that count (speed table: only those inside the window).
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

        # 3. Learn: `epochs` shuffled passes. With lr_decay the learning rate falls linearly to 0
        #    over the run, so late updates make only small changes.
        lr = args.lr * (1.0 - index / updates) if args.lr_decay else args.lr

        def one_minibatch(state: tuple, mb: Minibatch) -> tuple[tuple, tuple]:
            """One Adam step: (params, adam, stopped) in and out, plus (value_loss, kl).

            Once `stopped`, the step is still worked out (a compiled loop cannot skip work)
            but thrown away.
            """
            params, adam, stopped = state
            (_, (value_loss, kl)), grads = jax.value_and_grad(loss, has_aux=True)(params, mb)
            # Target KL (as in Stable-Baselines3): once the policy has moved too far, skip the rest.
            if args.target_kl is not None:
                stopped = stopped | (kl > 1.5 * args.target_kl)
            new_params, new_adam = adam_step(params, clip_by_global_norm(grads, MAX_GRAD_NORM), adam, lr)
            if args.method == 'speed_table':
                # Only the window's speeds may change: Adam's running average would nudge the rest.
                new_params = {**new_params, 'actor': {'speeds': jnp.where(
                    tuned, new_params['actor']['speeds'], params['actor']['speeds'])}}
            keep_old = lambda new, old: jax.tree.map(lambda a, b: jnp.where(stopped, b, a), new, old)
            return (keep_old(new_params, params), keep_old(new_adam, adam), stopped), (value_loss, kl)

        def one_epoch(state: tuple, key: jax.Array) -> tuple[tuple, tuple]:
            """One pass over the update's data in shuffled minibatches."""
            order = jax.random.permutation(key, batch)
            mbs = jax.tree.map(lambda a: a[order].reshape((args.minibatches, minibatch) + a.shape[1:]), data)
            return jax.lax.scan(one_minibatch, state, mbs)

        (params, adam, stopped), (value_loss, kl) = jax.lax.scan(
            one_epoch, (params, adam, jnp.asarray(False)), jax.random.split(k_shuffle, args.epochs))

        metrics = {
            'track_speed': jnp.mean(samples.progress) / sim.TIMESTEP,
            # Progress is along the centreline: 78.5 m per lap (the racing line is 75.9 m).
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

    # donate_argnums=0: the runner passed in is overwritten, so main() only uses the one returned.
    return jax.jit(update, donate_argnums=0)


# ----------------------------------------------------------------------
# Saving and loading
# ----------------------------------------------------------------------
def save(path: str | Path, params: dict, args: argparse.Namespace, line: Array | None = None) -> None:
    """Write the policy to an .npz the driver can load with numpy alone.

    `line` (speed table only) is the racing line [m], saved so the driver can check the table fits.
    The critic (for --init), DEFAULTS and args are saved too, so changed settings can be noticed.
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


def load_params(path: str | Path, method: str) -> dict:
    """Read a policy written by save() back in, to continue training from it (--init).

    Adam's memory is not saved, so it starts again from zero.
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


def keep_within_rule(params: dict, line: Array, device: jax.Device) -> tuple[dict, Array]:
    """Clip the tuned table to SPEED_RANGE and re-apply raceline.limit_speed_changes.

    Done in numpy after each update, not inside the compiled update, so the rule has one copy.
    """
    speeds = np.clip(np.asarray(params['actor']['speeds'], dtype=float), *SPEED_RANGE)
    speeds = limit_speed_changes(line, speeds, DEFAULTS['a_brake'])
    actor = {'speeds': jax.device_put(jnp.asarray(speeds, dtype=jnp.float32), device)}
    return {**params, 'actor': actor}, speeds


def equation(policy: dict) -> str:
    """The lookahead policy as text, e.g. 'L = 0.004 + 0.249*v + 0.003*|k0| + ... - 0.001*dk'."""
    terms = ' '.join(f'{"-" if w < 0 else "+"} {abs(w):.3f}*{name}'
                     for w, name in zip(np.asarray(policy['w']), LOOKAHEAD_INPUTS))
    return f'L = {float(policy["b"]):.3f} {terms}'


def main() -> None:
    """Read the command line, train a policy, and save it.

    --speed-table only affects --method lookahead; to keep tuning a saved speed table, use --init.
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
