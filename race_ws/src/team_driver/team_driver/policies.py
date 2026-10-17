"""The learned parts of the driver: a lookahead policy and a speed table.

Two ways for reinforcement learning (RL: learning by trial and error,
from a reward) to help pure pursuit:

1. Dynamic lookahead (first section), after Elgouhary & El-Wakeel 2026,
   "Dynamic Lookahead Distance via Reinforcement Learning-Based Pure
   Pursuit for Autonomous Racing" (arXiv 2603.28625): the policy chooses
   only pure pursuit's lookahead distance L, from 5 inputs. Here the policy
   is linear, 5 weights and a bias, so what it learned can be read off.
   Shipped as models/lookahead.npz.

2. Speed table (second section): the policy IS the speed table, one speed
   per racing-line point, starting from raceline.speed_table and tuned by
   PPO (Proximal Policy Optimisation, the RL method used here; see
   ppo_policy_loss). Shipped as models/speed_table.npz.

Both learn by PPO, whose loss (gaussian_log_prob, ppo_policy_loss) sits in
the first section and is shared.

Like pure_pursuit.py, every function takes `xp` (numpy in the driver,
jax.numpy in training), so training and racing compute the policies'
inputs and outputs with literally the same code.

Shapes: the functions work on ONE car. Training runs thousands of cars at
once with jax.vmap, which applies a one-car function to every car.
"""

import numpy as np

from team_driver.pure_pursuit import DEFAULTS


# ======================================================================
# The racing line's geometry, which both policies read
# ======================================================================


def path_geometry(path):
    """Heading and curvature at every point of a closed path.

    Curvature = 1 / turn radius [1/m]: 0 on a straight, larger in a tighter
    bend, positive turning left. Numpy only: computed once at startup, not
    every step.

    Parameters
    ----------
    path : (N, 2) array
        A closed loop of points [m], in driving order.

    Returns
    -------
    heading : (N,) array
        Direction of travel at each point [rad].
    curvature : (N,) array
        Curvature at each point [1/m].
    """
    ahead = np.roll(path, -1, axis=0)
    behind = np.roll(path, 1, axis=0)
    chord = ahead - behind                              # from the previous point to the next
    heading = np.arctan2(chord[:, 1], chord[:, 0])
    # Curvature = how fast the heading turns per metre travelled.
    turn = np.roll(heading, -1) - np.roll(heading, 1)
    turn = np.arctan2(np.sin(turn), np.cos(turn))       # wrap into -pi..pi
    curvature = turn / np.linalg.norm(chord, axis=1)
    return heading, curvature


# ======================================================================
# 1. Dynamic lookahead: L = b + w . s
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

# Where training starts: L = lookahead_gain * v, close to the hand-made rule
# in pure_pursuit.py (max(min_lookahead, lookahead_gain * v)). So training
# begins from today's driver.
START_WEIGHTS = (DEFAULTS['lookahead_gain'], 0.0, 0.0, 0.0, 0.0)
START_BIAS = 0.0


def lookahead_inputs(curvature, nearest, speed, xp=np):
    """The 5 lookahead-policy inputs s for one car.

    `curvature` comes from path_geometry(), `nearest` from the closest
    path point to the car.

    Parameters
    ----------
    curvature : (N,) array
        Curvature of the racing line at each point [1/m].
    nearest : int
        Index of the path point closest to the car.
    speed : float
        The car's speed [m/s].
    xp : module
        numpy or jax.numpy module; the same code runs in the ROS driver
        (numpy) and in training (JAX).

    Returns
    -------
    (5,) array
        s = (v [m/s], |k0|, |k1|, |k2| [1/m], dk [1/m]), as LOOKAHEAD_INPUTS.
    """
    n = curvature.shape[0]
    k0, k1, k2 = xp.abs(curvature[(nearest + xp.asarray(HORIZONS)) % n])
    return xp.stack([speed, k0, k1, k2, k1 - k0])


def lookahead(policy, s, xp=np):
    """The lookahead policy: L = b + w_v*v + w_0*|k0| + w_1*|k1| + w_2*|k2| + w_d*dk.

    Works for one car (s has shape (5,)) or many (s has shape (cars, 5)).
    The caller clips the result to LOOKAHEAD_RANGE before using it.

    Parameters
    ----------
    policy : dict
        {'w': (5,) array of weights, 'b': float bias}.
    s : (5,) or (cars, 5) array
        Inputs from lookahead_inputs().
    xp : module
        numpy or jax.numpy module; the same code runs in the ROS driver
        (numpy) and in training (JAX). (Unused: `@` works for both.)

    Returns
    -------
    float or (cars,) array
        Lookahead distance L [m], not yet clipped.
    """
    return s @ policy['w'] + policy['b']


def gaussian_log_prob(action, mean, log_std, xp=np):
    """Log of how likely `action` is under a bell curve centred on `mean`.

    The bell curve is a normal distribution with spread (standard
    deviation) exp(log_std). During training the policy picks L at random
    from that bell curve, so it tries lookaheads it would not otherwise;
    this says how likely each was.

    Parameters
    ----------
    action : float or array
        The value that was picked.
    mean : float or array, same shape as `action`
        Centre of the bell curve.
    log_std : float
        Log of the spread.
    xp : module
        numpy or jax.numpy module; the same code runs in the ROS driver
        (numpy) and in training (JAX).

    Returns
    -------
    float or array, same shape as `action`
        Log of the probability density.
    """
    z = (action - mean) * xp.exp(-log_std)
    return -0.5 * (z * z + 2.0 * log_std + xp.log(2.0 * xp.pi))


def ppo_policy_loss(log_prob, old_log_prob, advantage, log_std, clip, entropy_coef, xp=np,
                    weight=None):
    """PPO's loss for the policy: a number that gets smaller as the policy improves.

    PPO's loss for the policy (paper eq. 24 and 25, without the critic's
    part, which belongs to training). Gradient descent makes it smaller.

    ratio     how much more (> 1) or less (< 1) likely the policy now is to
              pick the action it picked while driving
    advantage how much better (> 0) or worse (< 0) than expected that action
              turned out (training/train.py works it out)
    The loss is lower when good actions become more likely and bad ones less
    likely, but the ratio is clipped to 1 +- clip, so one update can only
    change the policy a little: the "proximal" in PPO. The entropy term
    (how random the policy is) rewards keeping some randomness, so the
    policy keeps exploring.

    Parameters
    ----------
    log_prob : (samples,) array
        Log-probability of each action under the current policy.
    old_log_prob : (samples,) array
        Log-probability of the same actions when they were picked.
    advantage : (samples,) array
        How much better than expected each action turned out.
    log_std : float or array
        Log of the exploration spread (for the entropy term).
    clip : float
        How far the ratio may move from 1.
    entropy_coef : float
        Weight of the entropy bonus.
    xp : module
        numpy or jax.numpy module; the same code runs in the ROS driver
        (numpy) and in training (JAX).
    weight : (samples,) array, optional
        One per sample: 1 if it counts, 0 if it does not (the speed table
        only learns from cars inside the stretch being tuned). Leave it out
        and every sample counts.

    Returns
    -------
    loss : float
        The loss to minimise.
    kl : float
        KL divergence estimate: how far the policy has moved. Training
        stops an update early if it moves too far.
    """
    if weight is None:
        average = xp.mean
    else:
        average = lambda x: xp.sum(weight * x) / xp.maximum(xp.sum(weight), 1.0)
    ratio = xp.exp(log_prob - old_log_prob)
    clipped = xp.clip(ratio, 1.0 - clip, 1.0 + clip)
    surrogate = average(xp.minimum(ratio * advantage, clipped * advantage))
    entropy = xp.sum(log_std + 0.5 * (1.0 + xp.log(2.0 * xp.pi)))
    kl = average((ratio - 1.0) - (log_prob - old_log_prob))
    return -surrogate - entropy_coef * entropy, kl


def lookahead_loss(policy, log_std, s, chosen, old_log_prob, advantage, clip, entropy_coef, xp=np):
    """The loss for the dynamic-lookahead policy, for a batch of samples.

    How likely would the policy, with its current weights, be to make the same
    picks? Feed that into PPO's loss. The gradient of this loss with respect
    to w and b says how to change each weight.

    Parameters
    ----------
    policy : dict
        {'w': the 5 weights, 'b': the bias}: what training changes.
    log_std : float
        Log of the exploration spread [m]: training changes it too.
    s : (samples, 5) array
        The inputs v, |k0|, |k1|, |k2|, dk the car saw.
    chosen : (samples,) array
        The lookahead it picked [m], noise included, before clipping.
    old_log_prob : (samples,) array
        How likely that pick was when it was made.
    advantage : (samples,) array
        How much better than expected it went.
    clip, entropy_coef : float
        Passed to ppo_policy_loss().
    xp : module
        numpy or jax.numpy module; the same code runs in the ROS driver
        (numpy) and in training (JAX).

    Returns
    -------
    loss : float
    kl : float
        As ppo_policy_loss().
    """
    mean = lookahead(policy, s, xp)
    log_prob = gaussian_log_prob(chosen, mean, log_std, xp)
    return ppo_policy_loss(log_prob, old_log_prob, advantage, log_std, clip, entropy_coef, xp)


def load_lookahead(path):
    """Read a lookahead.npz written by training/train.py (numpy only).

    Parameters
    ----------
    path : str
        Path to lookahead.npz.

    Returns
    -------
    policy : dict
        {'w': (5,) array, 'b': float}: the weights, for lookahead().
    base_params : dict
        The pure pursuit tuning it was trained with.
    """
    data = np.load(path)
    policy = {'w': data['w'], 'b': float(data['b'])}
    base_params = dict(zip(data['base_param_names'].tolist(),
                           data['base_param_values'].tolist()))
    return policy, base_params


# ======================================================================
# 2. Speed table: one learned speed per racing-line point
# ======================================================================
#
# The policy is the table itself: `speeds`, one number per point, read off
# at the point nearest the car (exactly like pure_pursuit's speed_table
# argument). Training starts from raceline.speed_table and lets PPO move
# each number; training/train.py --method speed_table explains how.


def speed_table_inputs(curvature, nearest, speed, xp=np):
    """What the critic sees for one car (training only).

    The critic is the second network PPO trains: it guesses how well
    things will go from here, so the advantage can be measured against it.
    The table itself needs no inputs. The critic gets the 5 lookahead
    inputs, plus where on the lap the car is, as a point on a circle
    (sin, cos) so the end of the lap sits next to the start.

    Parameters
    ----------
    curvature : (N,) array
        Curvature of the racing line at each point [1/m].
    nearest : int
        Index of the path point closest to the car.
    speed : float
        The car's speed [m/s].
    xp : module
        numpy or jax.numpy module; the same code runs in the ROS driver
        (numpy) and in training (JAX).

    Returns
    -------
    (7,) array
        v, |k0|, |k1|, |k2|, dk, sin(lap angle), cos(lap angle).
    """
    n = curvature.shape[0]
    angle = 2.0 * xp.pi * nearest / n
    return xp.concatenate([lookahead_inputs(curvature, nearest, speed, xp),
                           xp.stack([xp.sin(angle), xp.cos(angle)])])


def speed_table_loss(speeds, log_std, point, chosen, old_log_prob, advantage, weight,
                     clip, entropy_coef, xp=np):
    """The loss for the speed table, for a batch of samples.

    The table's speed at `point` is the centre of the bell curve the speed
    was drawn from, so the gradient lands only on the points that were
    driven past: each speed learns from the cars that used it.

    Parameters
    ----------
    speeds : (points,) array
        The table [m/s]: what training changes.
    log_std : float
        Log of the exploration spread [m/s]. Fixed: training stops its
        gradient, because when learned it shrank to almost nothing.
    point : (samples,) int array
        The path point the car was nearest to.
    chosen : (samples,) array
        The speed it was sent [m/s], noise included.
    old_log_prob : (samples,) array
        How likely that speed was when it was sent.
    advantage : (samples,) array
        How much better than expected it went.
    weight : (samples,) array
        1 for cars inside the stretch being tuned, else 0.
    clip, entropy_coef : float
        Passed to ppo_policy_loss().
    xp : module
        numpy or jax.numpy module; the same code runs in the ROS driver
        (numpy) and in training (JAX).

    Returns
    -------
    loss : float
    kl : float
        As ppo_policy_loss().
    """
    log_prob = gaussian_log_prob(chosen, speeds[point], log_std, xp)
    return ppo_policy_loss(log_prob, old_log_prob, advantage, log_std, clip, entropy_coef,
                           xp, weight)


def load_speed_table(path):
    """Read a speed_table.npz written by training/train.py (numpy only).

    Parameters
    ----------
    path : str
        Path to speed_table.npz.

    Returns
    -------
    speeds : (N,) array
        One speed per racing-line point [m/s].
    line : (N, 2) array
        The racing line it was tuned on [m] (the driver checks it is still
        the same line).
    base_params : dict
        The pure pursuit tuning it was trained with.
    """
    data = np.load(path)
    base_params = dict(zip(data['base_param_names'].tolist(),
                           data['base_param_values'].tolist()))
    return data['speeds'], data['line'], base_params
