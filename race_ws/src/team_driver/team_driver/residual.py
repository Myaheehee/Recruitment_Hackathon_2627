"""The learned part: a small neural network that corrects pure pursuit.

Residual RL in one sentence: pure pursuit decides (steering, speed) as usual,
and a network trained with reinforcement learning adds a small correction
on top. The corrections are capped (LIMITS), so the worst a bad network can
do is nudge a controller that already laps.

Like pure_pursuit.py, every function takes `xp` (numpy in the driver,
jax.numpy in training), so training and racing compute the network's
inputs and outputs with literally the same code.

Shapes: the functions work on ONE car. Training runs thousands of cars at
once with jax.vmap, which applies a one-car function to every car.
"""

import numpy as np

# Largest correction the network may add: (steering [rad], speed [m/s]).
# The network's raw outputs are squashed into -1..1 by tanh and then
# multiplied by these. Saved inside the weights file, so the driver always
# uses the limits the network was trained with.
LIMITS = (0.10, 1.5)

# How far ahead along the racing line the network is told the curvature,
# in path points. The line is resampled every 0.2 m (raceline.py), so these
# are 0.6, 1, 2, 3, 4 and 6 metres ahead.
CURVATURE_AHEAD = (3, 5, 10, 15, 20, 30)

FEATURE_NAMES = (
    'speed', 'yaw_rate', 'last_steer', 'offset_from_line', 'heading_error',
    'base_steer', 'base_speed', 'free_ahead',
) + tuple(f'curvature_{k}' for k in CURVATURE_AHEAD)
NUM_FEATURES = len(FEATURE_NAMES)


def path_geometry(path):
    """Heading [rad] and curvature [1/m] at every point of a closed path.
    Numpy only: computed once at startup, not every step."""
    ahead = np.roll(path, -1, axis=0)
    behind = np.roll(path, 1, axis=0)
    chord = ahead - behind                              # from the previous point to the next
    heading = np.arctan2(chord[:, 1], chord[:, 0])
    # Curvature = how fast the heading turns per metre travelled.
    turn = np.roll(heading, -1) - np.roll(heading, 1)
    turn = np.arctan2(np.sin(turn), np.cos(turn))       # wrap into -pi..pi
    curvature = turn / np.linalg.norm(chord, axis=1)
    return heading, curvature


def features(path, heading, curvature, car, yaw, speed, yaw_rate, last_steer,
             base_steer, base_speed, nearest, ahead, params, xp=np):
    """The network's inputs for one car: NUM_FEATURES numbers, each divided
    by a typical size so they all land roughly in -1..1 (networks learn
    badly from inputs of wildly different sizes)."""
    n = path.shape[0]

    # Where the car is relative to the racing line at the nearest point:
    # sideways offset (positive = left of the line) and heading error.
    direction = xp.stack([xp.cos(heading[nearest]), xp.sin(heading[nearest])])
    offset = car - path[nearest]
    offset_from_line = direction[0] * offset[1] - direction[1] * offset[0]
    heading_error = yaw - heading[nearest]
    heading_error = xp.arctan2(xp.sin(heading_error), xp.cos(heading_error))

    # How sharply the line bends at a few points ahead.
    upcoming = curvature[(nearest + xp.asarray(CURVATURE_AHEAD)) % n]

    basics = xp.stack([
        speed / 10.0,
        yaw_rate / 5.0,
        last_steer / params['max_steer'],
        offset_from_line / 0.5,
        heading_error / 0.5,
        base_steer / params['max_steer'],
        base_speed / params['v_max'],
        ahead / params['max_range'],
    ])
    return xp.concatenate([basics, upcoming])


def actor(weights, x, xp=np):
    """The policy network: features -> 2 numbers (before tanh).
    Two hidden layers of tanh neurons, then a plain linear output."""
    h = xp.tanh(x @ weights['W0'] + weights['b0'])
    h = xp.tanh(h @ weights['W1'] + weights['b1'])
    return h @ weights['W2'] + weights['b2']


def apply_residual(base_steer, base_speed, squashed, limits, params, xp=np):
    """Add the network's correction to pure pursuit's command.
    `squashed` is the network output after tanh, so each entry is in -1..1."""
    steering = xp.clip(base_steer + limits[0] * squashed[0], -params['max_steer'], params['max_steer'])
    speed = xp.maximum(base_speed + limits[1] * squashed[1], 0.0)
    return steering, speed


def load_weights(path):
    """Read a weights file written by training/train.py (numpy only)."""
    data = np.load(path)
    weights = {k: data[k] for k in ('W0', 'b0', 'W1', 'b1', 'W2', 'b2')}
    limits = tuple(float(v) for v in data['limits'])
    base_params = dict(zip(data['base_param_names'].tolist(),
                           data['base_param_values'].tolist()))
    return weights, limits, base_params
