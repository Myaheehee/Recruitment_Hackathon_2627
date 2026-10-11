"""Check that training and racing compute the same thing.

    python training/test_parity.py

1. pure_pursuit() agrees with the original plan() from commit 0c2a770
   (before the controller moved out of driver.py).
2. numpy (the driver) and jax.numpy (training) give the same pure pursuit
   command, the same network inputs and the same network output.
3. A weights file written by train.py reads back correctly with the
   driver's loader.
4. The dynamic-lookahead policy: numpy and jax.numpy agree on its 5 inputs,
   on L, on pure pursuit aiming L ahead, and on the loss; JAX's gradient of
   the loss matches a numerical one; a saved policy reads back correctly.

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
from team_driver.residual import (
    actor, features, load_lookahead, load_weights, lookahead, lookahead_inputs, lookahead_loss,
    path_geometry)
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
    params = train.init_params(jax.random.key(1), 'residual', 128)
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
        train.save(file, params, Namespace(method='residual'))
        loaded, limits, trained_with = load_weights(file)
    assert all(np.array_equal(loaded[k], np.asarray(params['actor'][k])) for k in loaded)
    assert trained_with == DEFAULTS, trained_with
    print(f'3. save -> load round trip OK, limits {limits}')

    # 4. Dynamic lookahead.
    policy = {'w': np.array([0.35, -0.4, -0.3, -0.2, 0.5]), 'b': 0.3}
    jpolicy = {'w': jnp.asarray(policy['w']), 'b': jnp.asarray(policy['b'])}
    worst = {'inputs': 0.0, 'L': 0.0, 'command': 0.0}
    for _ in range(300):
        s = random_situation(rng, path, angles)
        nearest = int(np.argmin(np.linalg.norm(path - s['car'], axis=1)))
        x = lookahead_inputs(curvature, nearest, s['speed'])
        jx = lookahead_inputs(j['curvature'], jnp.asarray(nearest), jnp.asarray(s['speed']), xp=jnp)
        distance = float(np.clip(lookahead(policy, x), 0.35, 4.0))
        jdistance = lookahead(jpolicy, jx, xp=jnp)
        command = pure_pursuit(path, s['car'], s['yaw'], s['speed'], s['ranges'], angles,
                               DEFAULTS, lookahead=distance)
        jcommand = pure_pursuit(j['path'], jnp.asarray(s['car']), s['yaw'], s['speed'],
                                jnp.asarray(s['ranges']), j['angles'], DEFAULTS, xp=jnp,
                                lookahead=jnp.asarray(distance))
        worst['inputs'] = max(worst['inputs'], float(np.max(np.abs(x - np.asarray(jx)))))
        worst['L'] = max(worst['L'], abs(float(lookahead(policy, x)) - float(jdistance)))
        worst['command'] = max(worst['command'], abs(command[0] - float(jcommand[0])),
                               abs(command[1] - float(jcommand[1])))
    # Passing the speed rule's own lookahead must change nothing.
    s = random_situation(rng, path, angles)
    rule = max(DEFAULTS['min_lookahead'], DEFAULTS['lookahead_gain'] * s['speed'])
    plain = pure_pursuit(path, s['car'], s['yaw'], s['speed'], s['ranges'], angles, DEFAULTS)
    given = pure_pursuit(path, s['car'], s['yaw'], s['speed'], s['ranges'], angles, DEFAULTS,
                         lookahead=rule)
    assert all(np.array_equal(a, b) for a, b in zip(plain, given))

    # The loss, and its gradient with respect to the 5 weights.
    batch = rng.normal(size=(256, 5)) * [3, .5, .5, .5, .3] + [4, .5, .5, .5, 0]
    chosen = lookahead(policy, batch) + rng.normal(0, 0.3, 256)
    old_log_prob = rng.normal(-0.5, 0.1, 256)
    advantage = rng.normal(size=256)
    args = (np.log(0.3), batch, chosen, old_log_prob, advantage, 0.2, 0.02)
    numpy_loss, _ = lookahead_loss(policy, *args)
    jax_args = tuple(jnp.asarray(a) for a in args)
    jax_loss, _ = lookahead_loss(jpolicy, *jax_args, xp=jnp)
    gradient = jax.grad(lambda p: lookahead_loss(p, *jax_args, xp=jnp)[0])(jpolicy)['w']
    numerical = np.zeros(5)
    for i in range(5):
        step = np.zeros(5)
        step[i] = 1e-6
        up = lookahead_loss({'w': policy['w'] + step, 'b': policy['b']}, *args)[0]
        down = lookahead_loss({'w': policy['w'] - step, 'b': policy['b']}, *args)[0]
        numerical[i] = (up - down) / 2e-6
    worst['loss'] = abs(numpy_loss - float(jax_loss))
    worst['gradient'] = float(np.max(np.abs(np.asarray(gradient) - numerical)))
    print('4. dynamic lookahead, numpy vs jax: '
          + ', '.join(f'{k} {v:.1e}' for k, v in worst.items()))
    assert all(v < 1e-3 for v in worst.values()), worst

    with tempfile.TemporaryDirectory() as tmp:
        file = f'{tmp}/lookahead.npz'
        lparams = train.init_params(jax.random.key(2), 'lookahead', 16)
        train.save(file, lparams, Namespace(method='lookahead'))
        loaded, trained_with = load_lookahead(file)
    assert np.array_equal(loaded['w'], np.asarray(lparams['actor']['w']))
    assert trained_with == DEFAULTS
    print(f'   save -> load round trip OK, starting {train.equation(lparams["actor"])}')
    print('\nAll parity checks passed.')


if __name__ == '__main__':
    main()
