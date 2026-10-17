"""Builds the racing line and its physics speed table from the map, at startup.

    build_raceline()       1 left_directions  2 slide_limits  3 minimise_bending  4 resample
    speed_table()          cornering limit, then limit_speed_changes(); lap_time() times it
    load_clearance_map()   distance to the nearest wall, with the cone rows sealed
No ROS code, so training/ builds the same line.
"""

import os
from typing import Callable

import numpy as np
import yaml
from PIL import Image
from scipy import ndimage

from team_driver.pure_pursuit import G, MIN_CURVATURE
from team_driver.units import Array, Metres, Seconds

CAR_HALF_WIDTH = 0.155   # [m] the car is 0.31 m wide (docs/02-simulator.md, 2.8)
WALL_GAP = 0.2           # [m] air to keep between the car's side and a wall
ITERS = 20000            # how many nudges in step 3
SEAL_RADIUS = 0.40       # [m] sealing disc radius: wall gaps under 2x this count as wall
SAMPLE_STEP = 0.025      # [m] step 2 tests sideways positions this far apart ...
MAX_SLIDE = 3.0          # [m] ... up to this far each way
NUDGE = 0.12             # step size in step 3 (bigger makes it blow up)
SPACING = 0.2            # [m] distance between points of the finished line


def build_raceline(centre: Array, clearance: Callable[[Array], Array], margin: Metres, iters: int,
                   log: Callable[[str], None] = print) -> Array:
    """Slide the centreline sideways to bend as little as possible; points SPACING apart.

    `margin` is the closest the car's middle may come to a wall. If the finished
    line breaks it (by over 0.05 m), the centreline is returned instead.
    """
    left = left_directions(centre)
    max_left, max_right = slide_limits(centre, left, clearance, margin)
    shift = minimise_bending(centre, left, max_left, max_right, iters)
    raceline = resample(centre + shift[:, None] * left, SPACING)

    # Check against the map; the 0.05 m allows for map pixels and resampling.
    closest = clearance(raceline).min()
    length = np.linalg.norm(np.roll(raceline, -1, axis=0) - raceline, axis=1).sum()
    log(f'racing line: {len(raceline)} points, {length:.1f} m per lap '
        f'(centreline {len(centre)} points), closest approach to a wall {closest:.2f} m')
    if closest < margin - 0.05:
        log('racing line comes too close to a wall; following the centreline instead')
        return centre
    return raceline


def left_directions(points: Array) -> Array:
    """Step 1. A unit vector pointing left of the direction of travel, at every point of a loop."""
    forward = np.roll(points, -1, axis=0) - np.roll(points, 1, axis=0)
    forward /= np.linalg.norm(forward, axis=1, keepdims=True)
    return np.stack([-forward[:, 1], forward[:, 0]], axis=1)    # turn 90 degrees left


def slide_limits(centre: Array, left: Array, clearance: Callable[[Array], Array],
                 margin: Metres) -> tuple[Array, Array]:
    """Step 2. How far [m] each point may slide left, and right, staying `margin` from walls."""
    steps = int(MAX_SLIDE / SAMPLE_STEP)
    offsets = np.arange(-steps, steps + 1) * SAMPLE_STEP     # -3.0 ... +3.0 m, 0 in the middle
    samples = centre[:, None, :] + offsets[None, :, None] * left[:, None, :]
    fits = clearance(samples) >= margin       # fits[i, j]: point i slid by offsets[j] is OK

    going_left = fits[:, steps:]              # offsets 0, +0.025, +0.05, ...
    going_right = fits[:, steps::-1]          # offsets 0, -0.025, -0.05, ...

    def furthest(ok: Array) -> Array:
        """Furthest slide [m] before the first position that does not fit, per point."""
        # cumprod zeroes everything after the first misfit; - 1 as position 0 is the centreline.
        in_a_row = np.cumprod(ok, axis=1).sum(axis=1)
        return np.maximum(in_a_row - 1, 0) * SAMPLE_STEP

    return furthest(going_left), furthest(going_right)


def minimise_bending(centre: Array, left: Array, max_left: Array, max_right: Array,
                     iters: int) -> Array:
    """Step 3. Sideways shift [m] of each point (+ = left) that bends the line least.

    Projected gradient descent: nudge every point downhill, clamp to the slide limits, repeat.
    """
    shift = np.zeros(len(centre))
    for _ in range(iters):
        points = centre + shift[:, None] * left
        # How far each point sticks out from its neighbours; total bending = sum of bend**2.
        bend = np.roll(points, 1, axis=0) - 2.0 * points + np.roll(points, -1, axis=0)
        # The gradient: moving one point changes three bends, its own and its neighbours' ...
        push = np.roll(bend, 1, axis=0) - 2.0 * bend + np.roll(bend, -1, axis=0)
        # ... so slide against it, sideways only.
        shift -= NUDGE * np.sum(push * left, axis=1)
        shift = np.clip(shift, -max_right, max_left)
    return shift


def resample(points: Array, spacing: Metres) -> Array:
    """Step 4. Redraw the closed loop as points `spacing` apart (sliding bunched them in bends)."""
    closed = np.vstack([points, points[:1]])                   # repeat point 0 to close the loop
    step_lengths = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    distance = np.concatenate([[0.0], np.cumsum(step_lengths)])  # along the loop, to each point
    count = int(round(distance[-1] / spacing))
    wanted = np.linspace(0.0, distance[-1], count, endpoint=False)
    return np.stack([np.interp(wanted, distance, closed[:, 0]),
                     np.interp(wanted, distance, closed[:, 1])], axis=1)


def speed_table(path: Array, curvature: Array, params: dict) -> Array:
    """Fastest speed [m/s] at each point: the cornering limit, then limit_speed_changes().

    No top speed: on a straight only the speed-change rule limits it. Uses
    params['mu'] and params['a_brake'].
    """
    corner = np.sqrt(params['mu'] * G / np.maximum(np.abs(curvature), MIN_CURVATURE))
    return limit_speed_changes(path, corner, params['a_brake'])


def limit_speed_changes(path: Array, speeds: Array, a_brake: float) -> Array:
    """Lower speeds so v_next**2 <= v**2 + 2 * a_brake * d between neighbours, both ways round.

    Only ever lowers; returns a new array. a_brake is in m/s^2.
    """
    gap = np.linalg.norm(np.roll(path, -1, axis=0) - path, axis=1)   # gap[i]: point i to point i+1
    v = np.array(speeds, dtype=float)
    n = len(v)
    # A slow corner near the end also limits the start: repeat until a round changes nothing.
    while True:
        before = v.copy()
        for i in range(n):                    # forwards: speeding up
            v[i] = min(v[i], np.sqrt(v[i - 1] ** 2 + 2.0 * a_brake * gap[i - 1]))
        for i in reversed(range(n)):          # backwards: slowing down
            after = (i + 1) % n
            v[i] = min(v[i], np.sqrt(v[after] ** 2 + 2.0 * a_brake * gap[i]))
        if np.array_equal(v, before):
            return v


def lap_time(path: Array, speeds: Array) -> Seconds:
    """Lap time of the closed `path` at these speeds (each gap at the mean of its end speeds)."""
    gap = np.linalg.norm(np.roll(path, -1, axis=0) - path, axis=1)
    return float(np.sum(gap / (0.5 * (speeds + np.roll(speeds, -1)))))


def load_clearance_map(map_yaml: str) -> Callable[[Array], Array]:
    """Read the map; return clearance(points), the distance [m] to the nearest wall.

    Cone rows are sealed into solid walls first: driving between cones is
    against the rules (rule 11). Off the map counts as 0.
    """
    with open(map_yaml) as f:
        meta = yaml.safe_load(f)
    image = np.array(Image.open(os.path.join(os.path.dirname(map_yaml), meta['image'])))
    resolution = meta['resolution']                            # metres per pixel
    origin_x, origin_y = meta['origin'][0], meta['origin'][1]  # map x, y of the bottom-left pixel
    height = image.shape[0]

    # A pixel is wall when darker than the yaml's occupied_thresh.
    wall = (255.0 - image) / 255.0 > meta['occupied_thresh']

    # Seal the cone rows: grow every wall by a disc and shrink it back ("closing"),
    # which fills every gap narrower than the disc (cone gaps, but not corridors).
    radius = int(round(SEAL_RADIUS / resolution))            # in pixels
    y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    wall = ndimage.binary_closing(wall, structure=x * x + y * y <= radius * radius)

    distance_to_wall = ndimage.distance_transform_edt(~wall) * resolution

    def clearance(points: Array) -> Array:
        """Distance [m] from each (..., 2) point to the nearest wall."""
        # (x, y) -> pixel (row, column). Rows count down while y counts up; the 0.5
        # is because pixel centres are half a pixel in from their corners.
        cols = (points[..., 0] - origin_x) / resolution - 0.5
        rows = height - 1 - ((points[..., 1] - origin_y) / resolution - 0.5)
        # order=1 blends the 4 nearest pixels; off the map counts as wall.
        return ndimage.map_coordinates(distance_to_wall, [rows, cols],
                                       order=1, mode='constant', cval=0.0)

    return clearance
