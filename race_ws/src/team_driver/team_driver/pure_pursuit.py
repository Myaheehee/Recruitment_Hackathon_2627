"""Pure pursuit: steering and speed for one car.

No ROS code, so training/sim.py can run the exact same maths. Every function
takes `xp`: numpy in the driver, jax.numpy in training.
"""

import math
from typing import NamedTuple

import numpy as np

from team_driver.units import Array, Metres, MetresPerSecond, Radians

# The race settings (the judges pass no parameter file). driver.py turns each
# one into a ROS parameter; training reads them too.
DEFAULTS = {
    'v_max': 6.8,            # [m/s]   top speed (only without a speed table)
    'a_brake': 6.05,         # [m/s^2] braking we plan with
    'margin': 0.55,          # [m]     stop this far short of a wall (only without a speed table)
    'mu': 1.0489,            #         cornering grip we plan with
    'ahead_angle': 10.0,     # [deg]   half-width of the "straight ahead" LiDAR cone
    'max_range': 8.0,        # [m]     LiDAR ranges are clipped to this
    'lookahead_gain': 0.25,  # [s]     lookahead = gain * speed (when no policy is loaded)
    'min_lookahead': 0.5,    # [m]     ... but never shorter than this
    'wheelbase': 0.33,       # [m]
    'max_steer': 0.4189,     # [rad]
    'use_speed_table': True, #         False: speed from the three limits instead
    'table_scale': 1.0,      #         every table speed is multiplied by this
    'speed_lead': 0.0,       #         k: lean towards the table's speed 1 m ahead (0 = off)
}

G = 9.81                     # [m/s^2]
MIN_CURVATURE = 1e-6         # [1/m]   avoids dividing by zero on a straight
LEAD_POINTS = 5              # 1 m along the racing line (points are 0.2 m apart)


class Command(NamedTuple):
    """What pure_pursuit() decides."""
    steering: Radians        # positive = left, within +-max_steer
    speed: MetresPerSecond
    nearest: int             # index of the racing-line point closest to the car
    ahead: Metres            # free distance straight ahead, from the LiDAR


def preprocess(ranges: Array, max_range: Metres, xp=np) -> Array:
    """Clean a LiDAR scan: NaN -> 0, infinity -> max_range, clip to 0..max_range."""
    ranges = xp.nan_to_num(ranges, nan=0.0, posinf=max_range, neginf=0.0)
    return xp.clip(ranges, 0.0, max_range)


def pure_pursuit(path: Array, car: Array, yaw: Radians, speed: MetresPerSecond,
                 ranges: Array, angles: Array, params: dict, xp=np,
                 lookahead: Metres | None = None, speed_table: Array | None = None) -> Command:
    """Steer along the circle through the car and the racing-line point
    `lookahead` ahead; take the speed from the speed table.

    `car` is the rear axle (what odometry reports). Without `lookahead`, aim
    lookahead_gain * speed ahead. Without `speed_table`, the speed is the
    smallest of a braking, a cornering and a top-speed limit.
    """
    # 1. Nearest point of the line.
    distances = xp.linalg.norm(path - car, axis=1)
    nearest = xp.argmin(distances)

    # 2. How far ahead to aim. (`is None` is decided before JAX compiles.)
    if lookahead is None:
        lookahead = xp.maximum(params['min_lookahead'], params['lookahead_gain'] * speed)

    # 3. Target: the first point, going round from `nearest`, at least `lookahead` away.
    order = (nearest + xp.arange(len(path))) % len(path)
    first_far_enough = xp.argmax(distances[order] >= lookahead)
    target = path[order[first_far_enough]]

    # 4. How far the target is to the car's left.
    dx = target[0] - car[0]
    dy = target[1] - car[1]
    target_left = -xp.sin(yaw) * dx + xp.cos(yaw) * dy

    # 5. Steering for the circle through car and target (bicycle model).
    curvature = 2.0 * target_left / lookahead ** 2
    steering = xp.arctan(params['wheelbase'] * curvature)
    steering = xp.clip(steering, -params['max_steer'], params['max_steer'])

    # 6. Free distance ahead: the shortest beam in the cone.
    in_cone = xp.abs(angles) < math.radians(params['ahead_angle'])
    ahead = xp.min(xp.where(in_cone, ranges, params['max_range']))

    # 7. Speed from the table, plus a lead that fades out as the steering grows.
    if speed_table is not None:
        here = speed_table[nearest]
        next_metre = speed_table[(nearest + LEAD_POINTS) % path.shape[0]]
        straightness = 1.0 - xp.abs(steering) / params['max_steer']
        lead = params['speed_lead'] * straightness * (next_metre - here)
        return Command(steering, xp.maximum(here + lead, 0.0), nearest, ahead)

    # Without a table: the smallest of three limits.
    v_brake = xp.sqrt(2.0 * params['a_brake'] * xp.maximum(ahead - params['margin'], 0.0))
    v_corner = xp.sqrt(params['mu'] * G / xp.maximum(xp.abs(curvature), MIN_CURVATURE))
    speed_cmd = xp.minimum(params['v_max'], xp.minimum(v_brake, v_corner))

    return Command(steering, speed_cmd, nearest, ahead)
