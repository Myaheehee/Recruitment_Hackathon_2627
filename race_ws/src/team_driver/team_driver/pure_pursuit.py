"""Pure pursuit steering and speed for one car, shared by the driver and training.

Pure pursuit: steer towards a point a little way ahead on the racing line,
and drive as fast as the braking and cornering limits allow.

Used in two places, which is why it has no ROS code:
    driver.py        calls it on every LiDAR scan, with numpy
    training/sim.py  calls it inside the GPU simulator, with jax.numpy

About `xp`: numpy and jax.numpy have the same function names (xp.sqrt,
xp.argmin, ...), so every function here takes the library as an argument
called `xp`. It is numpy unless you say otherwise. That is also why there
is no `while` loop or `if` on numbers below: JAX cannot compile those.
"""

import math

import numpy as np

# Every number you can tune, with the value used in the race. The judges
# pass no parameter file, so these defaults ARE the race settings.
# driver.py turns each one into a ROS parameter; training reads them too.
#
# Tuned 2026-10-12 with a sweep in the training simulator (training/evaluate.py
# reproduces it): these lap in about 17.9 s with no wall contact at every
# reaction delay from 0 to 3 steps. The critical one is lookahead_gain: every
# value from 0.2 to 0.35 runs clean, 0.4 cuts the infield hairpin and hits it.
# That sweep predates the speed table (speed came from the three limits in
# step 7). With the PPO-tuned speed table and the lookahead policy (models/)
# the team measured laps of about 13.7 s (commit f9e6c43).
DEFAULTS = {
    'v_max': 6.8,            # [m/s]   top speed
    'a_brake': 6.05,          # [m/s^2] braking we plan with
    'margin': 0.55,          # [m]     stop this far short of a wall
    'mu': 1.0489,               #         cornering grip we plan with; the simulator's
                             #         tyres have 1.0489, so this leaves a safety margin
    'ahead_angle': 10.0,     # [deg]   half-width of the "straight ahead" cone
    'max_range': 8.0,        # [m]     LiDAR ranges are clipped to this
    'lookahead_gain': 0.25,  # [s]     lookahead distance = gain * speed
    'min_lookahead': 0.5,    # [m]     ... but never shorter than this
    'wheelbase': 0.33,       # [m]
    'max_steer': 0.4189,     # [rad]
    'use_speed_table': True, # True: speed from raceline.speed_table (built from mu
                             # and a_brake); False: the three limits in step 7
    'table_scale': 1.0,      #         speed table only: every speed in the table
                             #         (tuned or not) is multiplied by this. Below 1
                             #         leaves room for a slow ROS reaction (headless)
    'speed_lead': 0.0,       #         speed table only: k in step 7. 0 = off
}

G = 9.81                     # [m/s^2] gravity
MIN_CURVATURE = 1e-6         # [1/m]   avoids dividing by zero on a straight
LEAD_POINTS = 5              # 1 m along the racing line (its points are 0.2 m apart)


def preprocess(ranges, max_range, xp=np):
    """Clean a raw LiDAR scan so every range is a usable number.

    NaN becomes 0, infinity becomes max_range, and everything is clipped
    to 0..max_range.

    Parameters
    ----------
    ranges : (n_beams,) array
        Raw LiDAR ranges [m], one per beam.
    max_range : float
        Longest range kept [m].
    xp : module
        numpy or jax.numpy module; the same code runs in the ROS driver
        (numpy) and in training (JAX).

    Returns
    -------
    (n_beams,) array
        The cleaned ranges [m], each in 0..max_range.
    """
    ranges = xp.nan_to_num(ranges, nan=0.0, posinf=max_range, neginf=0.0)
    return xp.clip(ranges, 0.0, max_range)


def pure_pursuit(path, car, yaw, speed, ranges, angles, params, xp=np, lookahead=None,
                 speed_table=None):
    """Decide the steering angle and speed for one car.

    The "lookahead" is how far ahead along the racing line the car aims.
    The steering is the angle that drives the car on a circle through the
    target point. Curvature = 1 / turn radius [1/m]: 0 on a straight,
    larger in a tighter bend.

    Speed comes from one of two places (step 7). With a speed table, the
    table's speed at the nearest point, plus a small lead (speed_lead) so
    the car starts braking or speeding up about 1 m (LEAD_POINTS) early.
    This branch ignores the LiDAR's free distance `ahead` and v_max. Without
    a table, the smallest of a braking limit (from `ahead`), a cornering
    limit and v_max.

    Parameters
    ----------
    path : (N, 2) array
        The racing line [m], points 0.2 m apart, in driving order.
    car : (2,) array
        (x, y) of the rear axle [m], which is what odometry reports.
    yaw : float
        The car's heading [rad].
    speed : float
        The car's speed [m/s].
    ranges : (n_beams,) array
        Cleaned LiDAR ranges [m] (see preprocess()).
    angles : (n_beams,) array
        The angle of each beam [rad], 0 = straight ahead.
    params : dict
        The tuning table: DEFAULTS, or the driver's ROS parameters.
    xp : module
        numpy or jax.numpy module; the same code runs in the ROS driver
        (numpy) and in training (JAX).
    lookahead : float, optional
        How far ahead to aim [m]. Leave it out to use the speed rule in
        step 2 (a learned policy can choose it: policies.lookahead).
    speed_table : (N,) array, optional
        One speed [m/s] per path point (raceline.speed_table). Leave it out
        to use the three limits in step 7 instead.

    Returns
    -------
    steering : float
        Steering angle [rad], within +-max_steer (positive = left).
    speed : float
        Target speed [m/s], never negative.
    nearest : int
        Index of the path point closest to the car.
    ahead : float
        Free distance straight ahead [m], from the LiDAR.
    """
    # 1. Which path point is the car closest to?
    distances = xp.linalg.norm(path - car, axis=1)
    nearest = xp.argmin(distances)

    # 2. How far ahead to aim: further when faster, but at least min_lookahead.
    #    (Unless the caller chose it. `is None` is checked by Python before
    #    JAX compiles anything, so this `if` is allowed.)
    if lookahead is None:
        lookahead = xp.maximum(params['min_lookahead'], params['lookahead_gain'] * speed)

    # 3. The target: the first point after `nearest` that is at least
    #    `lookahead` away from the car. `order` lists the path indices
    #    starting at `nearest` and wrapping round the lap; argmax returns
    #    the position of the first True.
    order = (nearest + xp.arange(len(path))) % len(path)
    first_far_enough = xp.argmax(distances[order] >= lookahead)
    target = path[order[first_far_enough]]

    # 4. How far the target is to the car's left (negative = right).
    dx = target[0] - car[0]
    dy = target[1] - car[1]
    target_left = -xp.sin(yaw) * dx + xp.cos(yaw) * dy

    # 5. Steering. The circle through the car and the target has curvature
    #    2 * left / lookahead^2, and a car with this wheelbase drives that
    #    curvature with steering angle atan(wheelbase * curvature).
    curvature = 2.0 * target_left / lookahead ** 2
    steering = xp.arctan(params['wheelbase'] * curvature)
    steering = xp.clip(steering, -params['max_steer'], params['max_steer'])

    # 6. The free distance straight ahead: the shortest LiDAR beam in a
    #    narrow cone in front.
    in_cone = xp.abs(angles) < math.radians(params['ahead_angle'])
    ahead = xp.min(xp.where(in_cone, ranges, params['max_range']))

    # 7. Speed. With a speed table, worked out once at startup, read off the
    #    speed for this point of the line, plus a lead: the car's speed always
    #    lags its command, so add k (speed_lead) times how much the table
    #    changes over the next metre. Before a corner the table falls, so the
    #    car brakes a little earlier; after one it rises, so it speeds up a
    #    little earlier. The lead fades out with the steering angle (all of it
    #    going straight, none at full lock), so it does nothing mid-corner.
    if speed_table is not None:
        here = speed_table[nearest]
        next_metre = speed_table[(nearest + LEAD_POINTS) % path.shape[0]]
        straightness = 1.0 - xp.abs(steering) / params['max_steer']
        lead = params['speed_lead'] * straightness * (next_metre - here)
        return steering, xp.maximum(here + lead, 0.0), nearest, ahead

    #    Without one, the smallest of three limits.
    #    a) Braking. Stopping from speed v takes v^2 / (2 * a_brake) metres, so
    #       the fastest speed that can still stop `margin` short of the wall is:
    v_brake = xp.sqrt(2.0 * params['a_brake'] * xp.maximum(ahead - params['margin'], 0.0))

    #    b) Cornering. On a curve the tyres grip at most mu * g sideways, and
    #       the sideways acceleration is v^2 * curvature, so:
    v_corner = xp.sqrt(params['mu'] * G / xp.maximum(xp.abs(curvature), MIN_CURVATURE))

    #    c) Top speed.
    speed_cmd = xp.minimum(params['v_max'], xp.minimum(v_brake, v_corner))

    return steering, speed_cmd, nearest, ahead
