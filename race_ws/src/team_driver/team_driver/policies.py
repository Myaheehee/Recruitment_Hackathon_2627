"""The driver's two learned parts: a lookahead policy and a PPO-tuned speed table.

Race driver uses: path_geometry, lookahead_inputs, lookahead, load_lookahead, load_speed_table.
Training only: speed_table_inputs and the PPO maths (gaussian_log_prob, ppo_policy_loss,
lookahead_loss, speed_table_loss). Every function takes `xp` (numpy in the driver, jax.numpy
in training) and works on ONE car; training runs thousands at once with jax.vmap.
"""

import numpy as np

from team_driver.pure_pursuit import DEFAULTS
from team_driver.units import Array, Metres, MetresPerSecond


# ---- Racing-line geometry, read by both policies ----

def path_geometry(path: Array) -> tuple[Array, Array]:
    """Heading [rad] and curvature [1/m] at every point of a closed path.

    Curvature is positive turning left. Numpy only: computed once at startup.
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


# ---- 1. Lookahead policy: L = b + w . s (after arXiv 2603.28625) ----

# s (paper eq. 11): speed [m/s], |curvature| [1/m] at the HORIZONS, and dk = |k1| - |k0|.
LOOKAHEAD_INPUTS = ('v', '|k0|', '|k1|', '|k2|', 'dk')
HORIZONS = (0, 5, 15)                  # path points ahead: 0, 1 m, 3 m (ours; paper gives none)
LOOKAHEAD_RANGE = (0.35, 4.0)          # [m] the paper's limits on L (eq. 12)

# Training starts at L = lookahead_gain * v, i.e. from pure_pursuit.py's hand-made rule.
START_WEIGHTS = (DEFAULTS['lookahead_gain'], 0.0, 0.0, 0.0, 0.0)
START_BIAS = 0.0


def lookahead_inputs(curvature: Array, nearest: int, speed: MetresPerSecond, xp=np) -> Array:
    """The 5 lookahead-policy inputs s for one car, in LOOKAHEAD_INPUTS order."""
    n = curvature.shape[0]
    k0, k1, k2 = xp.abs(curvature[(nearest + xp.asarray(HORIZONS)) % n])
    return xp.stack([speed, k0, k1, k2, k1 - k0])


def lookahead(policy: dict, s: Array, xp=np) -> Metres:
    """The lookahead policy: L = b + w . s, with policy = {'w': (5,) weights, 'b': bias}.

    s may be (5,) for one car or (cars, 5) for many. Not clipped: the caller
    clips L to LOOKAHEAD_RANGE.
    """
    return s @ policy['w'] + policy['b']


def gaussian_log_prob(action: Array, mean: Array, log_std: float, xp=np) -> Array:
    """Log-probability of `action` under a normal distribution (mean, spread exp(log_std))."""
    z = (action - mean) * xp.exp(-log_std)
    return -0.5 * (z * z + 2.0 * log_std + xp.log(2.0 * xp.pi))


def ppo_policy_loss(log_prob: Array, old_log_prob: Array, advantage: Array, log_std: float | Array,
                    clip: float, entropy_coef: float, xp=np,
                    weight: Array | None = None) -> tuple[float, float]:
    """PPO's clipped policy loss (paper eq. 24-25, no critic term); returns (loss, kl).

    `weight` (1 or 0 per sample) leaves samples out of the averages; None counts
    them all. kl estimates how far the policy moved since the samples were taken.
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


def lookahead_loss(policy: dict, log_std: float, s: Array, chosen: Array, old_log_prob: Array,
                   advantage: Array, clip: float, entropy_coef: float,
                   xp=np) -> tuple[float, float]:
    """PPO loss of the lookahead policy over a batch of samples; returns (loss, kl).

    `chosen` is the L each car picked [m], noise included, before clipping.
    """
    mean = lookahead(policy, s, xp)
    log_prob = gaussian_log_prob(chosen, mean, log_std, xp)
    return ppo_policy_loss(log_prob, old_log_prob, advantage, log_std, clip, entropy_coef, xp)


def load_lookahead(path: str) -> tuple[dict, dict]:
    """Read lookahead.npz from training/train.py: (policy, pure pursuit params it trained with)."""
    data = np.load(path)
    policy = {'w': data['w'], 'b': float(data['b'])}
    base_params = dict(zip(data['base_param_names'].tolist(),
                           data['base_param_values'].tolist()))
    return policy, base_params


# ---- 2. Speed table: one PPO-tuned speed per racing-line point ----
# The policy is the table itself, read at the point nearest the car.
# Training starts it from raceline.speed_table.

def speed_table_inputs(curvature: Array, nearest: int, speed: MetresPerSecond, xp=np) -> Array:
    """The critic's 7 inputs for one car: the 5 lookahead inputs, then sin, cos of lap position."""
    n = curvature.shape[0]
    angle = 2.0 * xp.pi * nearest / n
    return xp.concatenate([lookahead_inputs(curvature, nearest, speed, xp),
                           xp.stack([xp.sin(angle), xp.cos(angle)])])


def speed_table_loss(speeds: Array, log_std: float, point: Array, chosen: Array,
                     old_log_prob: Array, advantage: Array, weight: Array,
                     clip: float, entropy_coef: float, xp=np) -> tuple[float, float]:
    """PPO loss of the speed table over a batch of samples; returns (loss, kl).

    Only speeds[point] get a gradient, so each speed learns from the cars that
    drove past it. `weight` is 1 for cars inside the stretch being tuned, else 0.
    """
    log_prob = gaussian_log_prob(chosen, speeds[point], log_std, xp)
    return ppo_policy_loss(log_prob, old_log_prob, advantage, log_std, clip, entropy_coef,
                           xp, weight)


def load_speed_table(path: str) -> tuple[Array, Array, dict]:
    """Read speed_table.npz from training/train.py: (speeds [m/s], line, params it trained with).

    `line` is the racing line it was tuned on: the driver rejects the table if
    its own line differs.
    """
    data = np.load(path)
    base_params = dict(zip(data['base_param_names'].tolist(),
                           data['base_param_values'].tolist()))
    return data['speeds'], data['line'], base_params
