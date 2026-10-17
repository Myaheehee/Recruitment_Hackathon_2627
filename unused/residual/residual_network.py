"""UNUSED: the residual network, kept for reference. Nothing imports this file.

A residual policy: pure pursuit decides (steering, speed) as usual, and a
small neural network adds a capped correction on top (at most LIMITS:
0.10 rad of steering and 1.5 m/s of speed), so the worst a bad network can
do is nudge a controller that already laps.

It was built and its pipeline tested (numpy and JAX agreed, weights saved
and loaded), but it was never trained to a model good enough to race, so
it was moved out of the driver. See README.md in this folder for how it was
wired into the driver and training, and how to get the fully wired version
back from git.

This is section 1 of the old team_driver/residual.py (now policies.py),
unchanged except that path_geometry(), which the other policies still use,
stays in policies.py. Like the rest of the driver's maths, every function
takes `xp` (numpy in the driver, jax.numpy in training).
"""

import numpy as np

# ======================================================================
# 1. Residual: a network adds a capped correction to pure pursuit
# ======================================================================

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


def features(path, heading, curvature, car, yaw, speed, yaw_rate, last_steer,
             base_steer, base_speed, nearest, ahead, params, xp=np):
    """The residual network's inputs for one car.

    NUM_FEATURES numbers, each divided by a typical size so they all land
    roughly in -1..1 (networks learn badly from inputs of wildly different
    sizes). The order is FEATURE_NAMES.

    Parameters
    ----------
    path : (N, 2) array
        The racing line [m].
    heading, curvature : (N,) array
        From path_geometry(): [rad] and [1/m].
    car : (2,) array
        (x, y) of the rear axle [m].
    yaw : float
        The car's heading [rad].
    speed : float
        The car's speed [m/s].
    yaw_rate : float
        How fast the car is turning [rad/s].
    last_steer : float
        The steering sent last step [rad].
    base_steer, base_speed : float
        Pure pursuit's command this step: [rad] and [m/s].
    nearest : int
        Index of the path point closest to the car.
    ahead : float
        Free distance straight ahead [m].
    params : dict
        The tuning table (uses max_steer, v_max, max_range).
    xp : module
        numpy or jax.numpy module; the same code runs in the ROS driver
        (numpy) and in training (JAX).

    Returns
    -------
    (NUM_FEATURES,) array
        The 14 scaled inputs (no units).
    """
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

    Two hidden layers of tanh neurons, then a plain linear output. (tanh
    squashes any number into -1..1.)

    Parameters
    ----------
    weights : dict
        'W0', 'b0', 'W1', 'b1', 'W2', 'b2': the layers' weight matrices
        and bias vectors, from load_weights().
    x : (NUM_FEATURES,) array
        The inputs from features().
    xp : module
        numpy or jax.numpy module; the same code runs in the ROS driver
        (numpy) and in training (JAX).

    Returns
    -------
    (2,) array
        Raw (steering, speed) corrections, before tanh and LIMITS.
    """
    h = xp.tanh(x @ weights['W0'] + weights['b0'])
    h = xp.tanh(h @ weights['W1'] + weights['b1'])
    return h @ weights['W2'] + weights['b2']


def apply_residual(base_steer, base_speed, squashed, limits, params, xp=np):
    """Add the network's correction to pure pursuit's command.

    `squashed` is the network output after tanh, so each entry is in -1..1.

    Parameters
    ----------
    base_steer : float
        Pure pursuit's steering [rad].
    base_speed : float
        Pure pursuit's speed [m/s].
    squashed : (2,) array
        tanh(actor(...)), each in -1..1.
    limits : tuple of 2 floats
        Largest corrections: (steering [rad], speed [m/s]).
    params : dict
        The tuning table (uses max_steer).
    xp : module
        numpy or jax.numpy module; the same code runs in the ROS driver
        (numpy) and in training (JAX).

    Returns
    -------
    steering : float
        Corrected steering [rad], within +-max_steer.
    speed : float
        Corrected speed [m/s], never negative.
    """
    steering = xp.clip(base_steer + limits[0] * squashed[0], -params['max_steer'], params['max_steer'])
    speed = xp.maximum(base_speed + limits[1] * squashed[1], 0.0)
    return steering, speed


def load_weights(path):
    """Read a residual.npz written by training/train.py (numpy only).

    Parameters
    ----------
    path : str
        Path to residual.npz.

    Returns
    -------
    weights : dict
        The network's weights, for actor().
    limits : tuple of 2 floats
        The correction caps it was trained with (steering [rad], speed [m/s]).
    base_params : dict
        The pure pursuit tuning it was trained with.
    """
    data = np.load(path)
    weights = {k: data[k] for k in ('W0', 'b0', 'W1', 'b1', 'W2', 'b2')}
    limits = tuple(float(v) for v in data['limits'])
    base_params = dict(zip(data['base_param_names'].tolist(),
                           data['base_param_values'].tolist()))
    return weights, limits, base_params
