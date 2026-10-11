"""Train a policy with PPO, on the GPU, in the training simulator.

Two kinds of policy (both in race_ws/src/team_driver/team_driver/residual.py):
    --method lookahead   (default) a linear policy, L = b + w . s, picks pure
                         pursuit's lookahead distance from 5 inputs
                         (Elgouhary & El-Wakeel 2026, arXiv 2603.28625)
    --method residual    a neural network corrects pure pursuit's steering and speed

    python training/train.py --smoke      # 2-minute check that everything runs
    python training/train.py              # a real run, roughly 10 minutes
    python training/train.py --method residual

Every run writes to training/runs/<date-time>/:
    lookahead.npz (or residual.npz)   the policy, rewritten every 20 updates,
                                      so a stopped run still leaves usable weights
    log.csv                           one row per update, for plotting
    args.json                         the settings used, so the run can be repeated

The method, in the order it happens:
  1. 2048 cars drive in parallel on the GPU, each from a random point of the
     track with a random reaction delay (sim.py). Every 0.01 s the policy acts,
     plus random noise so it tries things it would not otherwise.
  2. Each step scores a reward: metres of progress along the track, minus 10
     for touching a wall (which also ends that car's episode; it restarts
     elsewhere). The lookahead method adds three terms from the paper's
     reward (eq. 13): a bonus for speed, a cost for changing L, and a cost
     for standing still (--w-speed, --w-jerk, --w-stall). The residual
     instead pays a tiny cost for large corrections.
  3. After `rollout` steps, work out each action's advantage: how much better
     or worse it went than the critic predicted. Then gradient descent on the
     loss (PPO's policy loss from residual.py, plus the critic's error) for a
     few passes over the data. The gradient descent is Adam, written out below.
  4. Repeat until --steps simulated steps have been used.

The lookahead method uses the paper's PPO settings (section III-J): discount
0.99, GAE lambda 0.98, 5 passes, clip 0.2, target KL 0.015, entropy 0.02,
value weight 0.6, learning rate 2.3927e-4 falling linearly to 0. The
residual keeps the settings it was set up with.

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
from team_driver.residual import (
    FEATURE_NAMES, LIMITS, LOOKAHEAD_INPUTS, LOOKAHEAD_RANGE, NUM_FEATURES, START_BIAS,
    START_WEIGHTS, actor, apply_residual, gaussian_log_prob, lookahead, lookahead_loss,
    ppo_policy_loss)

COLLISION_PENALTY = 10.0   # reward lost on touching a wall [metres of progress]
RESIDUAL_COST = 0.002      # residual only: per step, times the squared size of the correction
MAX_GRAD_NORM = 0.5        # gradients longer than this are shrunk (see clip_by_global_norm)
STALL_SPEED = 0.05         # [m/s] slower than this counts as standing still (the paper's value)
STALL_STEPS = 50           # stalled = standing still for longer than this many steps (0.5 s)

# PPO settings for each method; any of them can be overridden on the command line.
SETTINGS = {
    'lookahead': dict(steps=20e6, epochs=5, minibatches=32, lr=2.3927e-4, lr_decay=True,
                      gamma=0.99, gae_lambda=0.98, entropy=0.02, value_coef=0.6, target_kl=0.015),
    'residual': dict(steps=50e6, epochs=4, minibatches=8, lr=3e-4, lr_decay=False,
                     gamma=0.995, gae_lambda=0.95, entropy=0.0, value_coef=0.5, target_kl=None),
}


class Sample(NamedTuple):
    """What PPO keeps from one step of one car."""
    inputs: jax.Array          # what the policy saw
    action: jax.Array          # what it picked, noise included (residual: before tanh;
                               # lookahead: L before clipping to LOOKAHEAD_RANGE)
    log_prob: jax.Array        # how likely the policy was to pick it
    value: jax.Array           # what the critic predicted
    reward: jax.Array
    terminated: jax.Array      # episode ended by a crash
    truncated: jax.Array       # episode ended by the time limit
    end_value: jax.Array       # critic's prediction for where the step ended up
    progress: jax.Array        # metres along the track this step
    used: jax.Array            # what reached the car (residual: correction -1..1;
                               # lookahead: the clipped L)
    l_change: jax.Array        # lookahead: |L - previous L| [m] (residual: 0)
    stalled: jax.Array         # lookahead: standing still too long (residual: False)


class Minibatch(NamedTuple):
    """What the learning step needs from each sample."""
    inputs: jax.Array
    action: jax.Array
    log_prob: jax.Array
    advantage: jax.Array       # how much better than expected it went
    ret: jax.Array             # the reward-to-go the critic should have predicted


# ----------------------------------------------------------------------
# The policy and the critic. The critic is a helper used only in training:
# a small network guessing how much reward is still to come from here, so
# that PPO can tell better-than-expected actions from worse ones.
# ----------------------------------------------------------------------
def network(key, n_in, hidden, n_out, last_scale):
    """Weights for residual.actor's shape: n_in -> hidden -> hidden -> n_out."""
    keys = jax.random.split(key, 3)
    sizes = [(n_in, hidden, math.sqrt(2.0)), (hidden, hidden, math.sqrt(2.0)),
             (hidden, n_out, last_scale)]
    weights = {}
    for i, (k, (a, b, scale)) in enumerate(zip(keys, sizes)):
        weights[f'W{i}'] = jax.nn.initializers.orthogonal(scale)(k, (a, b))
        weights[f'b{i}'] = jnp.zeros(b)
    return weights


def init_params(key, method, hidden):
    k_actor, k_critic = jax.random.split(key)
    if method == 'lookahead':
        return {
            'actor': {'w': jnp.asarray(START_WEIGHTS), 'b': jnp.asarray(START_BIAS)},
            'critic': network(k_critic, len(LOOKAHEAD_INPUTS), hidden, 1, 1.0),
            'log_std': jnp.asarray(math.log(0.3)),      # explore about +-0.3 m around L
        }
    return {
        # A tiny last layer means the corrections start out almost zero, so
        # training begins from plain pure pursuit.
        'actor': network(k_actor, NUM_FEATURES, hidden, 2, 0.01),
        'critic': network(k_critic, NUM_FEATURES, hidden, 1, 1.0),
        'log_std': jnp.full(2, -1.0),                   # noise 0.37 before tanh
    }


def critic(weights, x):
    return actor(weights, x, xp=jnp)[..., 0]


def advantages_and_returns(s, gamma, lam):
    """Generalised advantage estimation: for each step, how much better did
    things go than the critic expected? A crash gets no future value; a time
    limit does (the car would have kept driving). Arrays are (time, cars)."""
    bootstrap = jnp.where(s.terminated, 0.0, s.end_value)
    deltas = s.reward + gamma * bootstrap - s.value
    continues = ~(s.terminated | s.truncated)

    def backwards(next_advantage, inputs):
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
    step: jax.Array            # how many steps so far
    m: dict                    # running average of the gradient
    v: dict                    # running average of the squared gradient


def adam_init(params):
    return AdamState(jnp.zeros((), jnp.int32),
                     jax.tree.map(jnp.zeros_like, params), jax.tree.map(jnp.zeros_like, params))


def adam_step(params, grads, state, lr, b1=0.9, b2=0.999, eps=1e-5):
    """One step of Adam (Kingma & Ba 2015), the usual kind of gradient
    descent for neural networks.

    Plain gradient descent does   param <- param - lr * gradient.
    Adam keeps two running averages instead. `m` averages the gradient over
    recent steps, so minibatch noise cancels out. `v` averages its square;
    dividing by sqrt(v) makes every parameter's step about lr long, whether
    its gradient is huge (the speed weight, since v is up to 7 m/s) or tiny.
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
    """Treat all the gradients as one long vector; if it is longer than
    max_norm, shrink it to that length. One unlucky minibatch then cannot
    throw the weights far."""
    norm = jnp.sqrt(sum(jnp.sum(g * g) for g in jax.tree.leaves(grads)))
    scale = jnp.minimum(1.0, max_norm / (norm + 1e-8))
    return jax.tree.map(lambda g: g * scale, grads)


# ----------------------------------------------------------------------
# One update: drive, then learn.
# ----------------------------------------------------------------------
def make_update(simulator, path, args, updates):
    cars = args.envs
    batch = args.envs * args.rollout
    minibatch = batch // args.minibatches
    limits = jnp.asarray(LIMITS)
    controller = sim.make_controller(path)
    inputs, drive = sim.make_lookahead_controller(path)

    def step_simulator(carry, steer, speed, key):
        """Send the commands (after the reaction delay) and step every car."""
        env, observation, last_steer, queue, delay = carry[:5]
        k_step, k_reset, k_delay = jax.random.split(key, 3)
        queue, applied = sim.delay_commands(queue, jnp.stack([steer, speed], axis=1), delay)
        out = simulator.step_batch_autoreset(
            jax.random.split(k_step, cars), jax.random.split(k_reset, cars),
            env, applied[:, None, :])
        # Cars whose episode just ended were put back on the track by the
        # simulator: clear their memory and give them a new random delay.
        reset = out.reset
        last_steer = jnp.where(reset, 0.0, steer)
        queue = jnp.where(reset[:, None, None], 0.0, queue)
        delay = jnp.where(reset, jax.random.randint(k_delay, (cars,), 0, sim.MAX_DELAY + 1), delay)
        return out, (out.state, out.next_observation, last_steer, queue, delay)

    def sample(params, x, x_end, action, log_prob, reward, out, used,
               l_change=None, stalled=None):
        if l_change is None:
            l_change, stalled = jnp.zeros(cars), jnp.zeros(cars, dtype=bool)
        return Sample(l_change=l_change, stalled=stalled,
            inputs=x, action=action, log_prob=log_prob,
            value=critic(params['critic'], x), reward=reward,
            terminated=out.metrics.status.terminated, truncated=out.metrics.status.truncated,
            end_value=critic(params['critic'], x_end),
            progress=out.metrics.episode.progress[:, 0], used=used)

    def drive_lookahead(params, carry, key):
        k_noise, k_sim = jax.random.split(key)
        previous_l, still = carry[5], carry[6]                # still = steps spent standing still
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

    def drive_residual(params, carry, key):
        k_noise, k_sim = jax.random.split(key)
        base_steer, base_speed, x = controller(carry[1], carry[2])
        mean = actor(params['actor'], x, xp=jnp)
        action = mean + jnp.exp(params['log_std']) * jax.random.normal(k_noise, mean.shape)
        squashed = jnp.tanh(action)
        steer, speed = jax.vmap(
            lambda s, v, a: apply_residual(s, v, a, limits, DEFAULTS, xp=jnp)
        )(base_steer, base_speed, squashed)
        out, core = step_simulator(carry, steer, speed, k_sim)
        reward = out.rewards[:, 0] - RESIDUAL_COST * jnp.sum(squashed ** 2, axis=1)
        _, _, x_end = controller(out.transition_observation, steer)
        log_prob = jnp.sum(gaussian_log_prob(action, mean, params['log_std'], xp=jnp), axis=-1)
        return core + carry[5:], sample(params, x, x_end, action, log_prob, reward, out, squashed)

    drive_one_step = drive_lookahead if args.method == 'lookahead' else drive_residual

    def loss(params, mb):
        """The loss gradient descent makes smaller: the policy's (residual.py)
        plus the critic's squared error, weighted by value_coef."""
        if args.method == 'lookahead':
            policy_loss, kl = lookahead_loss(
                params['actor'], params['log_std'], mb.inputs, mb.action, mb.log_prob,
                mb.advantage, args.clip, args.entropy, xp=jnp)
        else:
            mean = actor(params['actor'], mb.inputs, xp=jnp)
            log_prob = jnp.sum(gaussian_log_prob(mb.action, mean, params['log_std'], xp=jnp), axis=-1)
            policy_loss, kl = ppo_policy_loss(log_prob, mb.log_prob, mb.advantage,
                                              params['log_std'], args.clip, args.entropy, xp=jnp)
        value_loss = 0.5 * jnp.mean((critic(params['critic'], mb.inputs) - mb.ret) ** 2)
        return policy_loss + args.value_coef * value_loss, (value_loss, kl)

    def update(runner):
        params, adam, carry, key, index = runner
        key, k_drive, k_shuffle = jax.random.split(key, 3)

        # 1. Drive: `rollout` steps for every car.
        carry, samples = jax.lax.scan(
            lambda c, k: drive_one_step(params, c, k), carry,
            jax.random.split(k_drive, args.rollout))

        # 2. Score every step against the critic's expectation.
        advantage, ret = advantages_and_returns(samples, args.gamma, args.gae_lambda)
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
        flat = jax.tree.map(lambda a: a.reshape((batch,) + a.shape[2:]), samples)
        data = Minibatch(flat.inputs, flat.action, flat.log_prob,
                         advantage.reshape(batch), ret.reshape(batch))

        # 3. Learn: a few passes over the data in shuffled minibatches.
        #    With lr_decay the learning rate falls in a straight line to 0
        #    over the run, so late updates make only small, careful changes.
        lr = args.lr * (1.0 - index / updates) if args.lr_decay else args.lr

        def one_minibatch(state, mb):
            params, adam, stopped = state
            (_, (value_loss, kl)), grads = jax.value_and_grad(loss, has_aux=True)(params, mb)
            # Target KL (as Stable-Baselines3 does it): once the policy has
            # moved too far from the one that collected this data, skip the
            # rest of this update. kl is measured before the step is taken.
            if args.target_kl is not None:
                stopped = stopped | (kl > 1.5 * args.target_kl)
            new_params, new_adam = adam_step(params, clip_by_global_norm(grads, MAX_GRAD_NORM), adam, lr)
            keep_old = lambda new, old: jax.tree.map(lambda a, b: jnp.where(stopped, b, a), new, old)
            return (keep_old(new_params, params), keep_old(new_adam, adam), stopped), (value_loss, kl)

        def one_epoch(state, key):
            order = jax.random.permutation(key, batch)
            mbs = jax.tree.map(lambda a: a[order].reshape((args.minibatches, minibatch) + a.shape[1:]), data)
            return jax.lax.scan(one_minibatch, state, mbs)

        (params, adam, stopped), (value_loss, kl) = jax.lax.scan(
            one_epoch, (params, adam, jnp.asarray(False)), jax.random.split(k_shuffle, args.epochs))

        metrics = {
            'track_speed': jnp.mean(samples.progress) / sim.TIMESTEP,
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
            metrics['steer_fix'] = jnp.mean(jnp.abs(samples.used[..., 0])) * LIMITS[0]
            metrics['speed_fix'] = jnp.mean(jnp.abs(samples.used[..., 1])) * LIMITS[1]
        return (params, adam, carry, key, index + 1), metrics

    return jax.jit(update, donate_argnums=0)


# ----------------------------------------------------------------------
# Saving and loading
# ----------------------------------------------------------------------
def save(path, params, args):
    """Write the policy so the driver can load it with numpy alone."""
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
        np.savez(path, **{k: np.asarray(v) for k, v in params['actor'].items()},
                 limits=np.asarray(LIMITS), feature_names=np.array(FEATURE_NAMES), **common)


def load_params(path, method):
    """Read a saved policy back in, to continue training from it (--init)."""
    data = np.load(path)
    names = ('W0', 'b0', 'W1', 'b1', 'W2', 'b2')
    if method == 'lookahead':
        policy = {'w': jnp.asarray(data['w']), 'b': jnp.asarray(data['b'])}
    else:
        policy = {k: jnp.asarray(data[k]) for k in names}
    return {'actor': policy,
            'critic': {k: jnp.asarray(data['critic_' + k]) for k in names},
            'log_std': jnp.asarray(data['log_std'])}


def equation(policy):
    """The lookahead policy as readable text, e.g. 'L = 0.02 + 0.41*v - 0.30*|k1|'."""
    terms = ' '.join(f'{"-" if w < 0 else "+"} {abs(w):.3f}*{name}'
                     for w, name in zip(np.asarray(policy['w']), LOOKAHEAD_INPUTS))
    return f'L = {float(policy["b"]):.3f} {terms}'


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--method', choices=('lookahead', 'residual'), default='lookahead')
    parser.add_argument('--steps', type=float, help='simulated steps in total')
    parser.add_argument('--envs', type=int, default=2048, help='cars driving in parallel')
    parser.add_argument('--rollout', type=int, default=128, help='steps per car between updates')
    parser.add_argument('--epochs', type=int, help='passes over each batch of data')
    parser.add_argument('--minibatches', type=int, help='pieces each pass is split into')
    parser.add_argument('--hidden', type=int, default=128, help="neurons per hidden layer (residual, and the critic)")
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
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', choices=('gpu', 'cpu'), default='gpu')
    parser.add_argument('--smoke', action='store_true', help='tiny run to check the setup works')
    args = parser.parse_args()
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
    update = make_update(simulator, sim.racing_line(), args, updates)

    key = jax.random.key(args.seed)
    key, k_params, k_reset, k_delay = jax.random.split(key, 4)
    params = load_params(args.init, args.method) if args.init else init_params(k_params, args.method, args.hidden)
    observation, env = simulator.reset_batch(jax.random.split(k_reset, args.envs))
    carry = (env, observation, jnp.zeros(args.envs),
             jnp.zeros((args.envs, sim.MAX_DELAY + 1, 2)),
             jax.random.randint(k_delay, (args.envs,), 0, sim.MAX_DELAY + 1),
             jnp.full(args.envs, -1.0), jnp.zeros(args.envs, dtype=jnp.int32))
    runner = (params, adam_init(params), carry, key, jnp.asarray(0))

    print(f'{args.method}: {updates} updates of {args.envs} cars x {args.rollout} steps on {device}. '
          f'The first one compiles, which takes a minute or two.\nSaving to {run_dir}\n')
    if args.method == 'lookahead':
        print(f'start: {equation(params["actor"])}\n')
    with open(run_dir / 'log.csv', 'w', newline='') as log_file:
        log = None
        started = time.time()
        for i in range(1, updates + 1):
            runner, metrics = update(runner)
            m = {k: float(v) for k, v in jax.device_get(metrics).items()}
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
                    line += f'  |steer fix| {m["steer_fix"]:.3f} rad  |speed fix| {m["speed_fix"]:.2f} m/s'
                print(line)
            if i % 20 == 0 or i == updates:
                save(out_file, runner[0], args)

    flag = '--lookahead' if args.method == 'lookahead' else '--weights'
    print(f'\nDone. Policy: {out_file}\nNext: python training/evaluate.py {flag} {out_file}')


if __name__ == '__main__':
    main()
