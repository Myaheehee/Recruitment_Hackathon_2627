"""The learned parts. Two ways for reinforcement learning to help pure pursuit:

1. Residual (first half of this file): pure pursuit decides (steering, speed)
   as usual, and a small neural network adds a correction on top. The
   corrections are capped (LIMITS), so the worst a bad network can do is
   nudge a controller that already laps.

2. Dynamic lookahead (second half), after Elgouhary & El-Wakeel 2026,
   "Dynamic Lookahead Distance via Reinforcement Learning-Based Pure
   Pursuit for Autonomous Racing" (arXiv 2603.28625): the policy chooses
   only pure pursuit's lookahead distance L, from 5 inputs. Here the policy
   is linear, 5 weights and a bias, so what it learned can be read off.

Like pure_pursuit.py, every function takes `xp` (numpy in the driver,
jax.numpy in training), so training and racing compute the network's
inputs and outputs with literally the same code.

Shapes: the functions work on ONE car. Training runs thousands of cars at
once with jax.vmap, which applies a one-car function to every car.
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
    """Read a residual.npz written by training/train.py (numpy only)."""
    data = np.load(path)
    weights = {k: data[k] for k in ('W0', 'b0', 'W1', 'b1', 'W2', 'b2')}
    limits = tuple(float(v) for v in data['limits'])
    base_params = dict(zip(data['base_param_names'].tolist(),
                           data['base_param_values'].tolist()))
    return weights, limits, base_params


# ======================================================================
# 2. Dynamic lookahead: L = b + w . s
# ======================================================================
#
# The 5 inputs s, as in the paper (eq. 11):
#     v      speed [m/s]
#     |k0|   how sharply the racing line bends at the nearest point [1/m]
#     |k1|   ... HORIZONS[1] points (1 m) further on
#     |k2|   ... HORIZONS[2] points (3 m) further on
#     dk     |k1| - |k0|: is the bend ahead getting tighter (+) or opening (-)?
# The paper does not say how far ahead its horizons are; 1 m and 3 m are ours.
LOOKAHEAD_INPUTS = ('v', '|k0|', '|k1|', '|k2|', 'dk')
HORIZONS = (0, 5, 15)                  # path points ahead (0.2 m apart)
LOOKAHEAD_RANGE = (0.35, 4.0)          # [m] the paper's limits on L (eq. 12)

# Where training starts: L = 0.4 * v, close to the hand-made rule in
# pure_pursuit.py (max(0.5, 0.4 * v)). So training begins from today's driver.
START_WEIGHTS = (0.4, 0.0, 0.0, 0.0, 0.0)
START_BIAS = 0.0


def lookahead_inputs(curvature, nearest, speed, xp=np):
    """The 5 inputs s for one car. `curvature` comes from path_geometry(),
    `nearest` from the closest path point to the car."""
    n = curvature.shape[0]
    k0, k1, k2 = xp.abs(curvature[(nearest + xp.asarray(HORIZONS)) % n])
    return xp.stack([speed, k0, k1, k2, k1 - k0])


def lookahead(policy, s, xp=np):
    """The policy: L = b + w_v*v + w_0*|k0| + w_1*|k1| + w_2*|k2| + w_d*dk.

    `policy` is {'w': 5 weights, 'b': bias}. Works for one car (s has shape
    (5,)) or many (s has shape (cars, 5)). The caller clips the result to
    LOOKAHEAD_RANGE before using it."""
    return s @ policy['w'] + policy['b']


def gaussian_log_prob(action, mean, log_std, xp=np):
    """log of the probability density of `action` under a bell curve
    (normal distribution) centred on `mean` with spread exp(log_std).
    During training the policy picks L at random from that bell curve, so it
    tries lookaheads it would not otherwise; this says how likely each was."""
    z = (action - mean) * xp.exp(-log_std)
    return -0.5 * (z * z + 2.0 * log_std + xp.log(2.0 * xp.pi))


def ppo_policy_loss(log_prob, old_log_prob, advantage, log_std, clip, entropy_coef, xp=np):
    """PPO's loss for the policy (paper eq. 24 and 25, without the critic's
    part, which belongs to training). Gradient descent makes it smaller.

    ratio     how much more (> 1) or less (< 1) likely the policy now is to
              pick the action it picked while driving
    advantage how much better (> 0) or worse (< 0) than expected that action
              turned out (training/train.py works it out)
    The loss is lower when good actions become more likely and bad ones less
    likely, but the ratio is clipped to 1 +- clip, so one update can only
    change the policy a little: the "proximal" in PPO. The entropy term
    rewards keeping some randomness, so the policy keeps exploring.

    Returns (loss, kl). kl estimates how far the policy has moved; training
    stops an update early if it moves too far."""
    ratio = xp.exp(log_prob - old_log_prob)
    clipped = xp.clip(ratio, 1.0 - clip, 1.0 + clip)
    surrogate = xp.mean(xp.minimum(ratio * advantage, clipped * advantage))
    entropy = xp.sum(log_std + 0.5 * (1.0 + xp.log(2.0 * xp.pi)))
    kl = xp.mean((ratio - 1.0) - (log_prob - old_log_prob))
    return -surrogate - entropy_coef * entropy, kl


def lookahead_loss(policy, log_std, s, chosen, old_log_prob, advantage, clip, entropy_coef, xp=np):
    """The loss for the dynamic-lookahead policy, for a batch of samples.

    policy        {'w': the 5 weights, 'b': the bias}: what training changes
    log_std       log of the exploration spread [m]: training changes it too
    s             (samples, 5) the inputs v, |k0|, |k1|, |k2|, dk the car saw
    chosen        (samples,) the lookahead it picked, noise included, before clipping
    old_log_prob  how likely that pick was when it was made
    advantage     how much better than expected it went

    How likely would the policy, with its current weights, be to make the same
    picks? Feed that into PPO's loss. The gradient of this loss with respect
    to w and b says how to change each weight."""
    mean = lookahead(policy, s, xp)
    log_prob = gaussian_log_prob(chosen, mean, log_std, xp)
    return ppo_policy_loss(log_prob, old_log_prob, advantage, log_std, clip, entropy_coef, xp)


def load_lookahead(path):
    """Read a lookahead.npz written by training/train.py (numpy only).
    Returns (policy, base_params): the weights, and the pure pursuit tuning
    it was trained with."""
    data = np.load(path)
    policy = {'w': data['w'], 'b': float(data['b'])}
    base_params = dict(zip(data['base_param_names'].tolist(),
                           data['base_param_values'].tolist()))
    return policy, base_params
