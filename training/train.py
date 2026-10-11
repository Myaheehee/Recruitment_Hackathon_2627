"""Train the residual network with PPO, on the GPU, in the training simulator.

    python training/train.py --smoke      # 2-minute check that everything runs
    python training/train.py              # a real run, roughly 20-25 minutes

Every run writes to training/runs/<date-time>/:
    residual.npz   the network (rewritten every few updates, so a stopped
                   run still leaves usable weights)
    log.csv        one row per update, the numbers printed on screen
    args.json      the settings used, so the run can be repeated

The method, in the order it happens:
  1. 2048 cars drive in parallel on the GPU, each from a random point of the
     track with a random reaction delay (sim.py). For each car, every
     0.01 s: pure pursuit decides (steering, speed), the network adds a
     correction, plus random noise so it tries things it would not otherwise.
  2. Each step scores a reward: metres of progress along the track, minus 10
     for touching a wall (which also ends that car's episode, and it restarts
     elsewhere), minus a tiny cost for large corrections.
  3. After 128 steps, PPO (Proximal Policy Optimisation) compares what each
     action earned with what the critic network predicted, and nudges the
     policy towards actions that did better than expected. "Proximal": every
     update is clipped so the policy can only change a little at a time.
  4. Repeat until --steps simulated steps have been used.

The PPO parts (advantages, the clipped loss) follow
external/f1tenth_gym/examples/jax_ppo_training.py (MIT licence), which in
turn follows PureJaxRL.
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
import optax

import sim
from team_driver.pure_pursuit import DEFAULTS
from team_driver.residual import FEATURE_NAMES, LIMITS, NUM_FEATURES, actor, apply_residual

COLLISION_PENALTY = 10.0   # reward lost on touching a wall [metres of progress]
RESIDUAL_COST = 0.002      # per step, times the squared size of the correction
LOG_TWO_PI = math.log(2.0 * math.pi)


class Sample(NamedTuple):
    """What PPO keeps from one step of one car."""
    features: jax.Array
    latent: jax.Array          # the action before tanh
    log_prob: jax.Array        # how likely the policy was to pick it
    value: jax.Array           # what the critic predicted
    reward: jax.Array
    terminated: jax.Array      # episode ended by a crash
    truncated: jax.Array       # episode ended by the time limit
    end_value: jax.Array       # critic's prediction for where the step ended up
    progress: jax.Array        # metres along the track this step
    squashed: jax.Array        # the correction, -1..1


class Minibatch(NamedTuple):
    """What the learning step needs from each sample."""
    features: jax.Array
    latent: jax.Array
    log_prob: jax.Array
    advantage: jax.Array       # how much better than expected it went
    ret: jax.Array             # the reward-to-go the critic should have predicted


# ----------------------------------------------------------------------
# The two networks. The actor is residual.actor (the same function the
# driver runs); the critic has the same shape with one output, its guess of
# how much reward is still to come from here.
# ----------------------------------------------------------------------
def init_params(key, hidden):
    def layer(key, n_in, n_out, scale):
        return (jax.nn.initializers.orthogonal(scale)(key, (n_in, n_out)), jnp.zeros(n_out))

    def network(keys, outputs, last_scale):
        w = {}
        w['W0'], w['b0'] = layer(keys[0], NUM_FEATURES, hidden, math.sqrt(2.0))
        w['W1'], w['b1'] = layer(keys[1], hidden, hidden, math.sqrt(2.0))
        w['W2'], w['b2'] = layer(keys[2], hidden, outputs, last_scale)
        return w

    keys = jax.random.split(key, 6)
    return {
        # A tiny last layer means the corrections start out almost zero, so
        # training begins from plain pure pursuit.
        'actor': network(keys[:3], 2, 0.01),
        'critic': network(keys[3:], 1, 1.0),
        # Exploration noise: standard deviation exp(-1) = 0.37 before tanh.
        'log_std': jnp.full(2, -1.0),
    }


def critic(weights, x):
    return actor(weights, x, xp=jnp)[..., 0]


def gaussian_log_prob(latent, mean, log_std):
    z = (latent - mean) * jnp.exp(-log_std)
    return jnp.sum(-0.5 * (z * z + 2.0 * log_std + LOG_TWO_PI), axis=-1)


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
# One update: drive, then learn.
# ----------------------------------------------------------------------
def make_update(simulator, controller, args):
    cars = args.envs
    limits = jnp.asarray(LIMITS)
    optimizer = optax.chain(optax.clip_by_global_norm(0.5), optax.adam(args.lr, eps=1e-5))
    batch = args.envs * args.rollout
    minibatch = batch // args.minibatches

    def drive_one_step(params, carry, key):
        env, observation, last_steer, queue, delay = carry
        k_noise, k_step, k_reset, k_delay = jax.random.split(key, 4)

        base_steer, base_speed, x = controller(observation, last_steer)
        mean = actor(params['actor'], x, xp=jnp)
        latent = mean + jnp.exp(params['log_std']) * jax.random.normal(k_noise, mean.shape)
        squashed = jnp.tanh(latent)
        steer, speed = jax.vmap(
            lambda s, v, a: apply_residual(s, v, a, limits, DEFAULTS, xp=jnp)
        )(base_steer, base_speed, squashed)

        queue, applied = sim.delay_commands(queue, jnp.stack([steer, speed], axis=1), delay)
        out = simulator.step_batch_autoreset(
            jax.random.split(k_step, cars), jax.random.split(k_reset, cars),
            env, applied[:, None, :])

        reward = out.rewards[:, 0] - RESIDUAL_COST * jnp.sum(squashed ** 2, axis=1)
        _, _, x_end = controller(out.transition_observation, steer)
        sample = Sample(
            features=x, latent=latent,
            log_prob=gaussian_log_prob(latent, mean, params['log_std']),
            value=critic(params['critic'], x), reward=reward,
            terminated=out.metrics.status.terminated, truncated=out.metrics.status.truncated,
            end_value=critic(params['critic'], x_end),
            progress=out.metrics.episode.progress[:, 0], squashed=squashed)

        # Cars whose episode just ended were put back on the track by the
        # simulator: clear their memory and give them a new random delay.
        reset = out.reset
        last_steer = jnp.where(reset, 0.0, steer)
        queue = jnp.where(reset[:, None, None], 0.0, queue)
        delay = jnp.where(reset, jax.random.randint(k_delay, (cars,), 0, sim.MAX_DELAY + 1), delay)
        return (out.state, out.next_observation, last_steer, queue, delay), sample

    def loss(params, mb):
        mean = actor(params['actor'], mb.features, xp=jnp)
        log_prob = gaussian_log_prob(mb.latent, mean, params['log_std'])
        ratio = jnp.exp(log_prob - mb.log_prob)
        # The clipped PPO objective: improve, but never by changing the
        # probability of an action by more than clip (20%) in one update.
        clipped = jnp.clip(ratio, 1.0 - args.clip, 1.0 + args.clip)
        policy_loss = -jnp.mean(jnp.minimum(ratio * mb.advantage, clipped * mb.advantage))
        value = critic(params['critic'], mb.features)
        value_loss = 0.5 * jnp.mean((value - mb.ret) ** 2)
        entropy = jnp.sum(params['log_std'] + 0.5 * (1.0 + LOG_TWO_PI))
        total = policy_loss + 0.5 * value_loss - args.entropy * entropy
        kl = jnp.mean((ratio - 1.0) - jnp.log(ratio))
        return total, (value_loss, kl)

    def update(runner):
        params, opt_state, carry, key = runner
        key, k_drive, k_shuffle = jax.random.split(key, 3)

        # 1. Drive: `rollout` steps for every car.
        carry, samples = jax.lax.scan(
            lambda c, k: drive_one_step(params, c, k), carry,
            jax.random.split(k_drive, args.rollout))

        # 2. Score every step against the critic's expectation.
        advantage, ret = advantages_and_returns(samples, args.gamma, args.gae_lambda)
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
        flat = jax.tree.map(lambda a: a.reshape((batch,) + a.shape[2:]), samples)

        data = Minibatch(flat.features, flat.latent, flat.log_prob,
                  advantage.reshape(batch), ret.reshape(batch))

        # 3. Learn: a few passes over the data in shuffled minibatches.
        def one_minibatch(state, mb):
            params, opt_state = state
            (_, (value_loss, kl)), grads = jax.value_and_grad(loss, has_aux=True)(params, mb)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            return (optax.apply_updates(params, updates), opt_state), (value_loss, kl)

        def one_epoch(state, key):
            order = jax.random.permutation(key, batch)
            mbs = jax.tree.map(lambda a: a[order].reshape((args.minibatches, minibatch) + a.shape[1:]), data)
            return jax.lax.scan(one_minibatch, state, mbs)

        (params, opt_state), (value_loss, kl) = jax.lax.scan(
            one_epoch, (params, opt_state), jax.random.split(k_shuffle, args.epochs))

        crashes = jnp.sum(samples.terminated)
        metrics = {
            'track_speed': jnp.mean(samples.progress) / sim.TIMESTEP,
            'crashes_per_lap': crashes / jnp.maximum(jnp.sum(samples.progress) / 78.5, 1e-6),
            'steer_fix': jnp.mean(jnp.abs(samples.squashed[..., 0])) * LIMITS[0],
            'speed_fix': jnp.mean(jnp.abs(samples.squashed[..., 1])) * LIMITS[1],
            'noise_std': jnp.mean(jnp.exp(params['log_std'])),
            'value_loss': jnp.mean(value_loss),
            'kl': jnp.mean(kl),
        }
        return (params, opt_state, carry, key), metrics

    return jax.jit(update, donate_argnums=0), optimizer


def save(path, params, args):
    """Write the weights so the driver can load them with numpy alone."""
    np.savez(
        path,
        **{k: np.asarray(v) for k, v in params['actor'].items()},
        **{'critic_' + k: np.asarray(v) for k, v in params['critic'].items()},
        log_std=np.asarray(params['log_std']),
        limits=np.asarray(LIMITS),
        base_param_names=np.array(list(DEFAULTS)),
        base_param_values=np.array(list(DEFAULTS.values()), dtype=float),
        feature_names=np.array(FEATURE_NAMES),
        args=np.array(json.dumps(vars(args), default=str)),
    )


def load_params(path):
    """Read a residual.npz back in, to continue training from it (--init)."""
    data = np.load(path)
    names = ('W0', 'b0', 'W1', 'b1', 'W2', 'b2')
    return {'actor': {k: jnp.asarray(data[k]) for k in names},
            'critic': {k: jnp.asarray(data['critic_' + k]) for k in names},
            'log_std': jnp.asarray(data['log_std'])}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--steps', type=float, default=50e6, help='simulated steps in total')
    parser.add_argument('--envs', type=int, default=2048, help='cars driving in parallel')
    parser.add_argument('--rollout', type=int, default=128, help='steps per car between updates')
    parser.add_argument('--epochs', type=int, default=4, help='passes over each batch of data')
    parser.add_argument('--minibatches', type=int, default=8)
    parser.add_argument('--hidden', type=int, default=128, help='neurons per hidden layer')
    parser.add_argument('--lr', type=float, default=3e-4, help='learning rate')
    parser.add_argument('--gamma', type=float, default=0.995,
                        help='discount: how much future reward counts (0.995 = about 2 s ahead)')
    parser.add_argument('--gae-lambda', type=float, default=0.95)
    parser.add_argument('--clip', type=float, default=0.2)
    parser.add_argument('--entropy', type=float, default=0.0, help='bonus for keeping exploration noise')
    parser.add_argument('--episode-steps', type=int, default=3000, help='time limit per episode (30 s)')
    parser.add_argument('--init', help='residual.npz to continue training from')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', choices=('gpu', 'cpu'), default='gpu')
    parser.add_argument('--smoke', action='store_true', help='tiny run to check the setup works')
    args = parser.parse_args()
    if args.smoke:
        args.steps, args.envs, args.rollout = 64 * 64 * 10, 64, 64

    run_dir = Path(__file__).resolve().parent / 'runs' / datetime.now().strftime('%Y%m%d-%H%M%S')
    run_dir.mkdir(parents=True)
    (run_dir / 'args.json').write_text(json.dumps(vars(args), indent=2))

    device = jax.devices(args.device)[0]
    simulator = sim.make_simulator(device, training=True, episode_steps=args.episode_steps,
                                   collision_penalty=COLLISION_PENALTY)
    controller = sim.make_controller(sim.racing_line())
    update, optimizer = make_update(simulator, controller, args)

    key = jax.random.key(args.seed)
    key, k_params, k_reset, k_delay = jax.random.split(key, 4)
    params = load_params(args.init) if args.init else init_params(k_params, args.hidden)
    observation, env = simulator.reset_batch(jax.random.split(k_reset, args.envs))
    carry = (env, observation, jnp.zeros(args.envs),
             jnp.zeros((args.envs, sim.MAX_DELAY + 1, 2)),
             jax.random.randint(k_delay, (args.envs,), 0, sim.MAX_DELAY + 1))
    runner = (params, optimizer.init(params), carry, key)

    updates = max(1, int(args.steps // (args.envs * args.rollout)))
    print(f'{updates} updates of {args.envs} cars x {args.rollout} steps on {device}. '
          f'The first one compiles, which takes a minute or two.\n'
          f'Saving to {run_dir}\n')
    columns = ['update', 'steps_M', 'steps_per_s', 'track_speed', 'crashes_per_lap',
               'steer_fix', 'speed_fix', 'noise_std', 'value_loss', 'kl']
    print('update  steps(M)  steps/s  track speed  crashes/lap  |steer fix|  |speed fix|  noise')
    with open(run_dir / 'log.csv', 'w', newline='') as log_file:
        log = csv.writer(log_file)
        log.writerow(columns)
        started = time.time()
        for i in range(1, updates + 1):
            runner, metrics = update(runner)
            m = {k: float(v) for k, v in jax.device_get(metrics).items()}
            steps = i * args.envs * args.rollout
            if i == 1:                     # the first update includes compiling
                started, compiled_steps = time.time(), steps
            rate = (steps - compiled_steps) / max(time.time() - started, 1e-9)
            log.writerow([i, steps / 1e6, rate] + [m[c] for c in columns[3:]])
            log_file.flush()
            if i == 1 or i % 5 == 0 or i == updates:
                print(f'{i:6d}  {steps / 1e6:8.2f}  {rate:7.0f}  {m["track_speed"]:7.2f} m/s'
                      f'  {m["crashes_per_lap"]:11.3f}  {m["steer_fix"]:7.3f} rad'
                      f'  {m["speed_fix"]:7.2f} m/s  {m["noise_std"]:5.2f}')
            if i % 20 == 0 or i == updates:
                save(run_dir / 'residual.npz', runner[0], args)

    print(f'\nDone. Weights: {run_dir / "residual.npz"}\n'
          f'Next: python training/evaluate.py --weights {run_dir / "residual.npz"}')


if __name__ == '__main__':
    main()
