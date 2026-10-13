"""The racing line, built from the map when the driver starts.

The idea: start from the track's centreline and slide each point sideways
so that the lap bends as little as possible, while keeping the car clear of
the walls. A gently bending line can be driven faster (see the cornering
limit in pure_pursuit.py), and "wide in, clip the inside, wide out" is
exactly what a bend-minimising line looks like.

    build_raceline()        the whole recipe, in four steps
      left_directions()     step 1: which way is sideways at each point
      slide_limits()        step 2: how far each point may slide
      minimise_bending()    step 3: slide the points
      resample()            step 4: space the points evenly again
    speed_table()           a target speed for every point of the line
      limit_speed_changes() its rule for neighbouring points
    lap_time()              how long a lap takes at those speeds
    load_clearance_map()    how far any point is from the nearest wall

No ROS in here, so the training code (training/) can build the same line.
"""

import os

import numpy as np
import yaml
from PIL import Image
from scipy import ndimage

from team_driver.pure_pursuit import G, MIN_CURVATURE

CAR_HALF_WIDTH = 0.155   # [m] the car is 0.31 m wide (docs/02-simulator.md, 2.8)
WALL_GAP = 0.2           # [m] air to keep between the car's side and a wall
ITERS = 20000            # how many times the points get nudged in step 3
SEAL_RADIUS = 0.40       # [m] gaps in the walls narrower than this count as wall
SAMPLE_STEP = 0.025      # [m] step 2 tests sideways positions this far apart ...
MAX_SLIDE = 3.0          # [m] ... up to this far each way
NUDGE = 0.12             # size of each nudge in step 3 (bigger makes the loop blow up)
SPACING = 0.2            # [m] distance between points of the finished line


def build_raceline(centre, clearance, margin, iters, log=print):
    """Return the racing line as an (N, 2) array of x, y points.

    centre     (N, 2) centreline points, in driving order
    clearance  the function from load_clearance_map()
    margin     [m] closest the middle of the car may come to a wall
    iters      how many nudges in step 3
    log        how to report: the driver passes its ROS logger
    """
    left = left_directions(centre)
    max_left, max_right = slide_limits(centre, left, clearance, margin)
    shift = minimise_bending(centre, left, max_left, max_right, iters)
    raceline = resample(centre + shift[:, None] * left, SPACING)

    # Check the finished line against the map. The 0.05 m allows for the
    # map's pixels and the resampling.
    closest = clearance(raceline).min()
    length = np.linalg.norm(np.roll(raceline, -1, axis=0) - raceline, axis=1).sum()
    log(f'racing line: {len(raceline)} points, {length:.1f} m per lap '
        f'(centreline {len(centre)} points), closest approach to a wall {closest:.2f} m')
    if closest < margin - 0.05:
        log('racing line comes too close to a wall; following the centreline instead')
        return centre
    return raceline


def left_directions(points):
    """Step 1. For each point of a closed loop, a length-1 vector pointing
    to the left of the direction of travel.

    np.roll shifts an array round by one place, so np.roll(points, -1)[i] is
    the next point and np.roll(points, 1)[i] the previous one. It wraps
    round at the ends, which is right because the lap is a loop.
    """
    forward = np.roll(points, -1, axis=0) - np.roll(points, 1, axis=0)
    forward /= np.linalg.norm(forward, axis=1, keepdims=True)
    return np.stack([-forward[:, 1], forward[:, 0]], axis=1)    # turn 90 degrees left


def slide_limits(centre, left, clearance, margin):
    """Step 2. How far [m] each point may slide left, and how far right,
    before the car would come closer than `margin` to a wall.

    For every point, test positions sideways from it, SAMPLE_STEP apart, and
    keep going until one is too close to a wall.
    """
    steps = int(MAX_SLIDE / SAMPLE_STEP)
    offsets = np.arange(-steps, steps + 1) * SAMPLE_STEP           # -3.0 ... +3.0 m, 0 in the middle
    samples = centre[:, None, :] + offsets[None, :, None] * left[:, None, :]
    fits = clearance(samples) >= margin       # fits[i, j]: point i slid by offsets[j] is OK

    going_left = fits[:, steps:]              # offsets 0, +0.025, +0.05, ...
    going_right = fits[:, steps::-1]          # offsets 0, -0.025, -0.05, ...

    def furthest(ok):
        # cumprod turns each row into 1s up to the first failure and 0s
        # after it, so the sum is how many positions in a row fit. The first
        # of those is the centreline itself, hence the - 1. A point that is
        # already too close to a wall gets 0 and stays where it is.
        in_a_row = np.cumprod(ok, axis=1).sum(axis=1)
        return np.maximum(in_a_row - 1, 0) * SAMPLE_STEP

    return furthest(going_left), furthest(going_right)


def minimise_bending(centre, left, max_left, max_right, iters):
    """Step 3. How far each point slides to the left (negative = right) so
    that the line bends as little as possible.

    The loop repeats one small move: work out which way each point should
    slide to reduce the total bending, slide it a little that way, and clamp
    it to the limits from step 2.
    """
    shift = np.zeros(len(centre))
    for _ in range(iters):
        points = centre + shift[:, None] * left
        # How far each point sticks out from the straight line through its
        # two neighbours: zero on a straight, bigger in a tighter bend. The
        # total bending is the sum of bend**2.
        bend = np.roll(points, 1, axis=0) - 2.0 * points + np.roll(points, -1, axis=0)
        # Moving one point changes three bends: its own and both its
        # neighbours'. This combination of the three is the direction in
        # which moving the point makes the total bending grow fastest
        # (the gradient) ...
        push = np.roll(bend, 1, axis=0) - 2.0 * bend + np.roll(bend, -1, axis=0)
        # ... so slide the opposite way. Only the sideways part counts.
        shift -= NUDGE * np.sum(push * left, axis=1)
        shift = np.clip(shift, -max_right, max_left)
    return shift


def resample(points, spacing):
    """Step 4. The same closed loop, redrawn as points `spacing` apart.
    (Sliding bunched the points up on the inside of bends.)"""
    closed = np.vstack([points, points[:1]])                   # repeat point 0 to close the loop
    step_lengths = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    distance = np.concatenate([[0.0], np.cumsum(step_lengths)])  # distance along the loop to each point
    count = int(round(distance[-1] / spacing))
    wanted = np.linspace(0.0, distance[-1], count, endpoint=False)
    # np.interp reads off x, then y, at each wanted distance.
    return np.stack([np.interp(wanted, distance, closed[:, 0]),
                     np.interp(wanted, distance, closed[:, 1])], axis=1)


def speed_table(path, curvature, params):
    """The fastest speed [m/s] at each point of the closed `path`, one per point.

    curvature  how sharply the line bends at each point [1/m]
               (residual.path_geometry works it out)
    params     uses mu and a_brake from the tuning table

    Two rules, and no top speed: on a straight only rule 2 limits the speed.
    1. Corners. At each point on its own: no faster than the cornering limit
       sqrt(mu * g / |curvature|) from pure_pursuit.py.
    2. Neighbours. Between two points d metres apart, the speed may change by
       at most 2 * a_brake * d / (v + v_next), speeding up or slowing down.
       Multiplied out, (v_next - v) * (v_next + v) = v_next^2 - v^2, so this
       is v_next^2 <= v^2 + 2 * a_brake * d: the distance needed to change
       speed at a_brake.
    Rule 2 is limit_speed_changes(), below.
    """
    corner = np.sqrt(params['mu'] * G / np.maximum(np.abs(curvature), MIN_CURVATURE))
    return limit_speed_changes(path, corner, params['a_brake'])


def limit_speed_changes(path, speeds, a_brake):
    """Lower the speeds of the closed `path` where needed, so that between
    any two neighbouring points the speed changes by at most
    2 * a_brake * d / (v + v_next). Speeds are only ever lowered.

    Done by walking round the lap forwards (how fast can the car be here,
    having sped up from the point before?) and then backwards (how fast can
    it be here and still slow down in time for the point after?). Training
    uses it too, to keep a tuned speed table within the rule.
    """
    gap = np.linalg.norm(np.roll(path, -1, axis=0) - path, axis=1)   # gap[i]: point i to point i+1
    v = np.array(speeds, dtype=float)
    n = len(v)
    # The lap is a loop, so a slow corner near the end also limits the start.
    # Keep walking round until a full round changes nothing (2-3 rounds).
    while True:
        before = v.copy()
        for i in range(n):                    # forwards: speeding up
            v[i] = min(v[i], np.sqrt(v[i - 1] ** 2 + 2.0 * a_brake * gap[i - 1]))
        for i in reversed(range(n)):          # backwards: slowing down
            after = (i + 1) % n
            v[i] = min(v[i], np.sqrt(v[after] ** 2 + 2.0 * a_brake * gap[i]))
        if np.array_equal(v, before):
            return v


def lap_time(path, speeds):
    """Seconds for one lap of the closed `path` at these speeds: each gap
    between points takes its length divided by the average of its two ends."""
    gap = np.linalg.norm(np.roll(path, -1, axis=0) - path, axis=1)
    return float(np.sum(gap / (0.5 * (speeds + np.roll(speeds, -1)))))


def load_clearance_map(map_yaml):
    """Read the map and return a function clearance(points) giving, for any
    (x, y) points, the distance in metres to the nearest wall.

    Rows of cones are sealed into solid walls first: the gaps between cones
    are wide enough for the car, but driving through them is against the
    rules (rule 11).
    """
    with open(map_yaml) as f:
        meta = yaml.safe_load(f)
    image = np.array(Image.open(os.path.join(os.path.dirname(map_yaml), meta['image'])))
    resolution = meta['resolution']                            # metres per pixel
    origin_x, origin_y = meta['origin'][0], meta['origin'][1]  # map position of the bottom-left pixel
    height = image.shape[0]

    # A pixel is wall when it is dark enough (the rule from the map's yaml).
    wall = (255.0 - image) / 255.0 > meta['occupied_thresh']

    # Seal the cone rows by "closing" the image: grow every wall by a disc,
    # then shrink it back. Walls that touched while grown stay joined, so
    # every gap narrower than the disc gets filled. Cones are 0.25-0.5 m
    # apart and the corridors over 1 m wide, so a 0.4 m disc seals exactly
    # the cone gaps.
    radius = int(round(SEAL_RADIUS / resolution))            # in pixels
    y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    wall = ndimage.binary_closing(wall, structure=x * x + y * y <= radius * radius)

    # For every pixel, the distance to the nearest wall pixel, in metres.
    distance_to_wall = ndimage.distance_transform_edt(~wall) * resolution

    def clearance(points):
        # Map (x, y) -> pixel (row, column). Image rows count down from the
        # top while y counts up, hence the flip; the 0.5 is because pixel
        # centres are half a pixel in from their corners.
        cols = (points[..., 0] - origin_x) / resolution - 0.5
        rows = height - 1 - ((points[..., 1] - origin_y) / resolution - 0.5)
        # Blend the 4 nearest pixels (order=1). Off the map counts as wall.
        return ndimage.map_coordinates(distance_to_wall, [rows, cols],
                                       order=1, mode='constant', cval=0.0)

    return clearance
