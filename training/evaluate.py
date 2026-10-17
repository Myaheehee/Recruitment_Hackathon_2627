"""Quick race check in the training simulator: 1 warm-up + 10 timed laps per reaction delay.

    python training/evaluate.py                     # pure pursuit, untuned speed table
    python training/evaluate.py --lookahead training/runs/<run>/lookahead.npz
    python training/evaluate.py --speed-table training/runs/<run>/speed_table.npz
    python training/evaluate.py --lookahead race_ws/src/team_driver/models/lookahead.npz \\
        --speed-table race_ws/src/team_driver/models/speed_table.npz   # the shipped set-up

"python" = ~/venvs/rl/bin/python. The official result is the ROS race: scripts/evaluate.sh.
"""

import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np

import sim
from team_driver.pure_pursuit import DEFAULTS
from team_driver.policies import LOOKAHEAD_RANGE, load_lookahead, load_speed_table, lookahead

CHUNK = 1000     # physics steps per compiled chunk (10 s of racing)


def main() -> None:
    """Load the policy files, race one car per delay 0..--max-delay, print a lap table.

    Crashes do not stop a car (race rules). Prints only; writes no files.
    """
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--lookahead',
                        help='lookahead.npz: let it choose the lookahead distance '
                             '(default: lookahead_gain * speed)')
    parser.add_argument('--speed-table',
                        help='speed_table.npz: drive by this tuned table (with or without --lookahead)')
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

    policy = None
    if args.lookahead:
        policy, trained_with = load_lookahead(args.lookahead)
        changed = [k for k in DEFAULTS if trained_with.get(k) != DEFAULTS[k]]
        if changed:
            print(f'WARNING: pure pursuit settings changed since training: {changed}')
        w, b = policy['w'], policy['b']
        print('L = ' + f'{b:.3f} ' + ' '.join(f'{"-" if x < 0 else "+"} {abs(x):.3f}*{n}'
                                            for x, n in zip(w, ('v', '|k0|', '|k1|', '|k2|', 'dk'))))
        policy = {'w': jnp.asarray(w), 'b': jnp.asarray(b)}

    def step(carry: tuple, key: jax.Array) -> tuple[tuple, tuple]:
        """One 0.01 s step for every car: pure pursuit command -> reaction-delay queue -> simulator.

        carry is (simulator state, observation, delay queue). Also returns, per car: laps completed,
        latest lap time [s], touching a wall now. jax.lax.scan runs it CHUNK times.
        """
        state, observation, queue = carry
        if policy is not None:
            # The policy's best guess for L, with no exploration noise.
            distance = jnp.clip(lookahead(policy, inputs(observation), xp=jnp), *LOOKAHEAD_RANGE)
            steer, speed = drive(observation, distance)
        else:
            steer, speed = controller(observation)
        queue, applied = sim.delay_commands(queue, jnp.stack([steer, speed], axis=1), delays)
        out = simulator.step_batch(jax.random.split(key, cars), state, applied[:, None, :])
        record = (out.metrics.episode.lap_counts[:, 0], out.metrics.episode.lap_times[:, 0],
                  out.observation.collisions[:, 0] > 0)
        return (out.state, out.observation, queue), record

    run_chunk = jax.jit(lambda carry, keys: jax.lax.scan(step, carry, keys))

    observation, state = simulator.reset_batch_from_poses(
        jax.random.split(jax.random.key(0), cars), sim.start_poses(cars))
    carry = (state, observation, jnp.zeros((cars, sim.MAX_DELAY + 1, 2)))
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

    name = f'dynamic lookahead {args.lookahead}' if args.lookahead else 'pure pursuit only'
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
