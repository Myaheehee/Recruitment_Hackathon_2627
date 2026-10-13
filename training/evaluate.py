"""Race the driver in the training simulator: 1 warm-up lap + 10 timed laps
from the grid, like the referee, once for every reaction delay from 0 to
sim.MAX_DELAY steps (the cars race side by side, each in its own world).

    python training/evaluate.py                       # pure pursuit only
    python training/evaluate.py --weights training/runs/<run>/residual.npz
    python training/evaluate.py --lookahead training/runs/<run>/lookahead.npz
    python training/evaluate.py --speed-table training/runs/<run>/speed_table.npz

This is a quick check, not the official result: the race runs in ROS
(scripts/evaluate.sh), and nobody knows its exact reaction delay. A driver
worth shipping should finish cleanly at every delay.
"""

import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np

import sim
from team_driver.pure_pursuit import DEFAULTS
from team_driver.residual import (
    LOOKAHEAD_RANGE, actor, apply_residual, load_lookahead, load_speed_table, load_weights,
    lookahead)

CHUNK = 1000     # physics steps per compiled chunk (10 s of racing)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--weights', help='residual.npz to add (default: pure pursuit only)')
    parser.add_argument('--lookahead', help='lookahead.npz: let it choose the lookahead distance')
    parser.add_argument('--speed-table',
                        help='speed_table.npz: drive by this tuned table (with any of the above)')
    parser.add_argument('--max-delay', type=int, default=sim.MAX_DELAY,
                        help='largest reaction delay [steps]; each car gets 0..this')
    parser.add_argument('--laps', type=int, default=11, help='laps to drive, warm-up included')
    parser.add_argument('--max-time', type=float, default=300.0, help='give up after this [s]')
    parser.add_argument('--device', choices=('cpu', 'gpu'), default='cpu',
                        help='cpu is faster for a handful of cars')
    args = parser.parse_args()
    sim.MAX_DELAY = args.max_delay

    device = jax.devices(args.device)[0]
    path = sim.racing_line()
    simulator = sim.make_simulator(device, training=False)
    speeds = None                                    # None = whatever the driver would use
    if args.speed_table:
        speeds, line, trained_with = load_speed_table(args.speed_table)
        if line.shape != path.shape or not np.allclose(line, path, atol=1e-3):
            parser.error(f'{args.speed_table} was tuned on a different racing line')
        changed = [k for k in DEFAULTS if trained_with.get(k) != DEFAULTS[k]]
        if changed:
            print(f'WARNING: pure pursuit settings changed since the table was tuned: {changed}')
    controller = sim.make_controller(path, speeds)
    inputs, drive = sim.make_lookahead_controller(path, speeds)
    delays = jnp.arange(sim.MAX_DELAY + 1)          # one car per delay
    cars = len(delays)

    weights = None
    if args.weights:
        weights, limits, trained_with = load_weights(args.weights)
        weights = {k: jnp.asarray(v) for k, v in weights.items()}
        changed = [k for k in DEFAULTS if trained_with.get(k) != DEFAULTS[k]]
        if changed:
            print(f'WARNING: pure pursuit settings changed since training: {changed}')

    policy = None
    if args.lookahead:
        if args.weights:
            parser.error('use --weights or --lookahead, not both')
        policy, trained_with = load_lookahead(args.lookahead)
        changed = [k for k in DEFAULTS if trained_with.get(k) != DEFAULTS[k]]
        if changed:
            print(f'WARNING: pure pursuit settings changed since training: {changed}')
        w, b = policy['w'], policy['b']
        print('L = ' + f'{b:.3f} ' + ' '.join(f'{"-" if x < 0 else "+"} {abs(x):.3f}*{n}'
                                            for x, n in zip(w, ('v', '|k0|', '|k1|', '|k2|', 'dk'))))
        policy = {'w': jnp.asarray(w), 'b': jnp.asarray(b)}

    def step(carry, key):
        state, observation, last_steer, queue = carry
        steer, speed, x = controller(observation, last_steer)
        if policy is not None:
            # The policy's best guess for L, with no exploration noise.
            distance = jnp.clip(lookahead(policy, inputs(observation), xp=jnp), *LOOKAHEAD_RANGE)
            steer, speed = drive(observation, distance)
        if weights is not None:
            # Racing uses the network's best guess, with no exploration noise.
            squashed = jnp.tanh(actor(weights, x, xp=jnp))
            steer, speed = jax.vmap(
                lambda s, v, a: apply_residual(s, v, a, limits, DEFAULTS, xp=jnp)
            )(steer, speed, squashed)
        queue, applied = sim.delay_commands(queue, jnp.stack([steer, speed], axis=1), delays)
        out = simulator.step_batch(jax.random.split(key, cars), state, applied[:, None, :])
        record = (out.metrics.episode.lap_counts[:, 0], out.metrics.episode.lap_times[:, 0],
                  out.observation.collisions[:, 0] > 0)
        return (out.state, out.observation, steer, queue), record

    run_chunk = jax.jit(lambda carry, keys: jax.lax.scan(step, carry, keys))

    observation, state = simulator.reset_batch_from_poses(
        jax.random.split(jax.random.key(0), cars), sim.start_poses(cars))
    carry = (state, observation, jnp.zeros(cars), jnp.zeros((cars, sim.MAX_DELAY + 1, 2)))
    key = jax.random.key(1)
    lap_times = [[] for _ in range(cars)]
    contacts = np.zeros(cars, dtype=int)
    touching = np.zeros(cars)
    previous_laps = np.zeros(cars)
    previous_hit = np.zeros(cars, dtype=bool)
    started = time.time()
    for _ in range(int(args.max_time / sim.TIMESTEP / CHUNK)):
        key, chunk_key = jax.random.split(key)
        carry, (lap_counts, lap_time, hit) = run_chunk(carry, jax.random.split(chunk_key, CHUNK))
        for counts, times, hits in zip(*map(np.asarray, (lap_counts, lap_time, hit))):
            for car in np.flatnonzero(counts > previous_laps):     # laps just completed
                lap_times[car].append(float(times[car]))
            contacts += hits & ~previous_hit                       # new touches only
            touching += hits * sim.TIMESTEP
            previous_laps, previous_hit = counts, hits
        if min(len(t) for t in lap_times) >= args.laps:
            break

    name = (f'residual {args.weights}' if args.weights else
            f'dynamic lookahead {args.lookahead}' if args.lookahead else 'pure pursuit only')
    name += (f', tuned speed table {args.speed_table}' if args.speed_table else
             ', speed table' if DEFAULTS['use_speed_table'] else ', pure pursuit speed limits')
    print(f'\n{name}'
          f'  ({time.time() - started:.0f} s of computer time)\n')
    print('delay   laps   best timed lap   10-lap total   wall contacts (time touching)')
    for car in range(cars):
        timed = lap_times[car][1:args.laps]                 # lap 1 is the warm-up
        best = f'{min(timed):8.2f} s' if timed else '       -  '
        total = f'{sum(timed):8.2f} s' if len(timed) == args.laps - 1 else '       -  '
        print(f'{int(delays[car])} step  {len(lap_times[car]):4d}   {best}       {total}'
              f'     {contacts[car]:3d} ({touching[car]:.2f} s)')
    print('\nA wall contact costs 10 s in the race, and more than 10 disqualifies.')


if __name__ == '__main__':
    main()
