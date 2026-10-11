"""Pure pursuit: steer towards a point a little way ahead on the racing line,
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
DEFAULTS = {
    'v_max': 7.0,            # [m/s]   top speed
    'a_brake': 6.07,         # [m/s^2] braking we plan with
    'margin': 0.55,          # [m]     stop this far short of a wall
    'mu': 1.0489,            #         tyre friction (the simulator's value)
    'ahead_angle': 10.0,     # [deg]   half-width of the "straight ahead" cone
    'max_range': 8.0,        # [m]     LiDAR ranges are clipped to this
    'lookahead_gain': 0.4,   # [s]     lookahead distance = gain * speed
    'min_lookahead': 0.5,    # [m]     ... but never shorter than this
    'wheelbase': 0.33,       # [m]
    'max_steer': 0.4189,     # [rad]
}

G = 9.81                     # [m/s^2] gravity
MIN_CURVATURE = 1e-6         # [1/m]   avoids dividing by zero on a straight


def preprocess(ranges, max_range, xp=np):
    """Clean a raw LiDAR scan: NaN becomes 0, infinity becomes max_range,
    and everything is clipped to 0..max_range."""
    ranges = xp.nan_to_num(ranges, nan=0.0, posinf=max_range, neginf=0.0)
    return xp.clip(ranges, 0.0, max_range)


def pure_pursuit(path, car, yaw, speed, ranges, angles, params, xp=np):
    """Decide the steering angle and speed for one car.

    Inputs
        path     (N, 2) array: the racing line, points 0.2 m apart, in driving order
        car      (x, y) of the rear axle, which is what odometry reports
        yaw      the car's heading [rad]
        speed    the car's speed [m/s]
        ranges   cleaned LiDAR ranges [m], with `angles` the angle of each beam [rad]
        params   the tuning table: DEFAULTS, or the driver's ROS parameters

    Returns (steering [rad], speed [m/s], nearest, ahead). The last two are
    the index of the path point closest to the car and the free distance
    straight ahead [m]; the residual network uses them as inputs.
    """
    # 1. Which path point is the car closest to?
    distances = xp.linalg.norm(path - car, axis=1)
    nearest = xp.argmin(distances)

    # 2. How far ahead to aim: further when faster, but at least min_lookahead.
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

    # 6. Speed: the smallest of three limits.
    #    a) Braking. `ahead` is the shortest LiDAR beam in a narrow cone in
    #       front. Stopping from speed v takes v^2 / (2 * a_brake) metres, so
    #       the fastest speed that can still stop `margin` short of the wall is:
    in_cone = xp.abs(angles) < math.radians(params['ahead_angle'])
    ahead = xp.min(xp.where(in_cone, ranges, params['max_range']))
    v_brake = xp.sqrt(2.0 * params['a_brake'] * xp.maximum(ahead - params['margin'], 0.0))

    #    b) Cornering. On a curve the tyres grip at most mu * g sideways, and
    #       the sideways acceleration is v^2 * curvature, so:
    v_corner = xp.sqrt(params['mu'] * G / xp.maximum(xp.abs(curvature), MIN_CURVATURE))

    #    c) Top speed.
    speed_cmd = xp.minimum(params['v_max'], xp.minimum(v_brake, v_corner))

    return steering, speed_cmd, nearest, ahead
