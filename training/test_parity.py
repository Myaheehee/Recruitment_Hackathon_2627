"""Check that training and racing compute the same thing.

    python training/test_parity.py

1. pure_pursuit() agrees with the original plan() from commit 0c2a770
   (before the controller moved out of driver.py).
2. numpy (the driver) and jax.numpy (training) give the same pure pursuit
   command, the same network inputs and the same network output.
3. A weights file written by train.py reads back correctly with the
   driver's loader.

If any of these fail, a network trained here would be racing a different
controller from the one it learned to correct.
"""

import math
import subprocess
import tempfile
import textwrap
import types
from argparse import Namespace

import jax
import jax.numpy as jnp
import numpy as np

import sim
from team_driver.pure_pursuit import DEFAULTS, pure_pursuit
from team_driver.residual import actor, features, load_weights, path_geometry
import train

ORIGINAL = '0c2a770'      # last commit with plan() written out inside driver.py


def original_plan():
    """plan() exactly as it was in driver.py, loaded from git."""
    source = subprocess.run(
        ['git', '-C', str(sim.REPO), 'show',
         f'{ORIGINAL}:race_ws/src/team_driver/team_driver/driver.py'],
        capture_output=True, text=True, check=True).stdout
    start = source.index('    def plan(self, ranges, angles):')
    end = source.index('    # Output', start)
    namespace = {'math': math, 'np': np}
    exec(textwrap.dedent(source[start:end]), namespace)
    return namespace['plan']


def random_situation(rng, path, angles):
    i = rng.integers(len(path))
    return dict(car=path[i] + rng.normal(0.0, 0.3, 2), yaw=rng.uniform(-math.pi, math.pi),
                speed=rng.uniform(0.0, 8.0), yaw_rate=rng.normal(0.0, 2.0),
                last_steer=rng.uniform(-0.4, 0.4), ranges=rng.uniform(0.0, 8.0, len(angles)))


def main():
    rng = np.random.default_rng(0)
    path = sim.racing_line()
    heading, curvature = path_geometry(path)
    angles = np.deg2rad(-135.0) + np.arange(819) * np.deg2rad(270.0) / 818

    # 1. Moved code == original code.
    plan = original_plan()
    worst = 0.0
    for _ in range(1000):
        s = random_situation(rng, path, angles)
        me = types.SimpleNamespace(
            position=tuple(s['car']), yaw=s['yaw'], speed=s['speed'], path=path,
            ahead_angle=math.radians(DEFAULTS['ahead_angle']),
            **{k: v for k, v in DEFAULTS.items() if k != 'ahead_angle'})
        old = plan(me, s['ranges'], angles)
        new = pure_pursuit(path, s['car'], s['yaw'], s['speed'], s['ranges'], angles, DEFAULTS)
        worst = max(worst, abs(old[0] - new[0]), abs(old[1] - new[1]))
    print(f'1. original plan() vs pure_pursuit(): largest difference {worst:.1e}')
    assert worst < 1e-9

    # 2. numpy == jax, for pure pursuit, features and the network.
    params = train.init_params(jax.random.key(1), 128)
    # Make the network's output big enough that a mismatch would show.
    params['actor']['W2'] = params['actor']['W2'] * 100.0
    weights_np = {k: np.asarray(v, dtype=np.float64) for k, v in params['actor'].items()}
    j = {k: jnp.asarray(v) for k, v in dict(path=path, heading=heading, curvature=curvature,
                                             angles=angles).items()}
    worst = {'command': 0.0, 'features': 0.0, 'network': 0.0}
    for _ in range(300):
        s = random_situation(rng, path, angles)
        steer, speed, nearest, ahead = pure_pursuit(
            path, s['car'], s['yaw'], s['speed'], s['ranges'], angles, DEFAULTS)
        x = features(path, heading, curvature, s['car'], s['yaw'], s['speed'], s['yaw_rate'],
                     s['last_steer'], steer, speed, nearest, ahead, DEFAULTS)
        out = actor(weights_np, x)

        js = {k: jnp.asarray(v) for k, v in s.items()}
        jsteer, jspeed, jnearest, jahead = pure_pursuit(
            j['path'], js['car'], js['yaw'], js['speed'], js['ranges'], j['angles'],
            DEFAULTS, xp=jnp)
        jx = features(j['path'], j['heading'], j['curvature'], js['car'], js['yaw'], js['speed'],
                      js['yaw_rate'], js['last_steer'], jsteer, jspeed, jnearest, jahead,
                      DEFAULTS, xp=jnp)
        jout = actor(params['actor'], jx, xp=jnp)

        worst['command'] = max(worst['command'], abs(steer - float(jsteer)), abs(speed - float(jspeed)))
        worst['features'] = max(worst['features'], float(np.max(np.abs(x - np.asarray(jx)))))
        worst['network'] = max(worst['network'], float(np.max(np.abs(out - np.asarray(jout)))))
    print('2. numpy vs jax, largest differences: '
          + ', '.join(f'{k} {v:.1e}' for k, v in worst.items()))
    # JAX computes in 32-bit floats, numpy in 64-bit: expect ~1e-5, not 0.
    assert all(v < 1e-3 for v in worst.values()), worst

    # 3. A saved file loads back with the driver's loader.
    with tempfile.TemporaryDirectory() as tmp:
        file = f'{tmp}/residual.npz'
        train.save(file, params, Namespace(note='test'))
        loaded, limits, trained_with = load_weights(file)
    assert all(np.array_equal(loaded[k], np.asarray(params['actor'][k])) for k in loaded)
    assert trained_with == DEFAULTS, trained_with
    print(f'3. save -> load round trip OK, limits {limits}')
    print('\nAll parity checks passed.')


if __name__ == '__main__':
    main()
