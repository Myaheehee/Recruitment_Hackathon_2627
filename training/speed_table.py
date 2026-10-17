"""Compute, print and plot the physics-based speed table; run `~/venvs/rl/bin/python training/speed_table.py`.

    python training/speed_table.py                      # with pure_pursuit.DEFAULTS
    python training/speed_table.py --mu 0.8 --a-brake 5  # try other values

("python" here means ~/venvs/rl/bin/python.)

The table (raceline.speed_table) gives one target speed per racing-line
point: as fast as the corner allows (mu), with neighbouring points never
further apart in speed than braking at a_brake allows. There is no top speed.

The driver does use a speed table, by default (use_speed_table=True in
pure_pursuit.DEFAULTS). It races a PPO-tuned table,
race_ws/src/team_driver/models/speed_table.npz, made by
`train.py --method speed_table`, which starts from the table worked out
here. (If the tuned table's stored racing line does not match the one the
driver builds at startup, the driver falls back to the table worked out
here.) This script only works out and draws that untuned starting table;
it writes no .npz and changes nothing the driver uses. (DEFAULTS['v_max']
only matters when use_speed_table is False.)

Prints a summary and every 10th point, and writes
    training/speed_table.csv   every point: index, distance along the lap, x, y, curvature, speed
    training/speed_table.png   the line on the map, blue = slow, red = fast
"""

import argparse
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw

import sim
from team_driver.pure_pursuit import DEFAULTS
from team_driver.raceline import lap_time, speed_table
from team_driver.policies import path_geometry

HERE = Path(__file__).resolve().parent
SCALE = 3          # the png is drawn this many times larger than the map


def main():
    """Build the table from the command-line mu and a_brake, print it and save it.

    Reads --mu and --a-brake (defaults from pure_pursuit.DEFAULTS), builds the
    racing line and its table, prints a summary and every 10th point, and
    writes training/speed_table.csv and training/speed_table.png.
    """
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--mu', type=float, default=DEFAULTS['mu'], help='cornering grip')
    parser.add_argument('--a-brake', type=float, default=DEFAULTS['a_brake'],
                        help='largest change of speed [m/s^2], both ways')
    args = parser.parse_args()
    params = dict(DEFAULTS, mu=args.mu, a_brake=args.a_brake)

    path = sim.racing_line()
    _, curvature = path_geometry(path)
    speeds = speed_table(path, curvature, params)

    gap = np.linalg.norm(np.roll(path, -1, axis=0) - path, axis=1)
    along = np.concatenate([[0.0], np.cumsum(gap)[:-1]])      # metres from point 0
    slow, fast = np.argmin(speeds), np.argmax(speeds)
    print(f'\nmu {args.mu}, a_brake {args.a_brake} m/s^2')
    print(f'{len(path)} points, lap {gap.sum():.1f} m')
    print(f'slowest {speeds[slow]:.2f} m/s at point {slow} ({path[slow, 0]:.1f}, {path[slow, 1]:.1f})')
    print(f'fastest {speeds[fast]:.2f} m/s at point {fast} ({path[fast, 0]:.1f}, {path[fast, 1]:.1f})')
    print(f'lap time at exactly these speeds: {lap_time(path, speeds):.2f} s '
          f'(the car will be slower: it reacts late and does not hit every speed exactly)\n')

    print(' point      s [m]      x      y   curvature  speed [m/s]')
    for i in range(0, len(path), 10):
        print(f'{i:6d}   {along[i]:7.1f}  {path[i, 0]:5.1f}  {path[i, 1]:5.1f}'
              f'   {curvature[i]:8.3f}   {speeds[i]:6.2f}')

    table = np.column_stack([np.arange(len(path)), along, path, curvature, speeds])
    np.savetxt(HERE / 'speed_table.csv', table, delimiter=',', fmt=['%d'] + ['%.4f'] * 5,
               header='point,s,x,y,curvature,speed', comments='')
    draw(path, speeds, HERE / 'speed_table.png')
    print(f'\nwrote {HERE / "speed_table.csv"} and {HERE / "speed_table.png"}')


def draw(path, speeds, file):
    """Draw the racing line on the map, each piece coloured by its speed.

    Parameters
    ----------
    path : numpy.ndarray, shape (points, 2)
        Racing-line points (x, y) in map coordinates [m].
    speeds : numpy.ndarray, shape (points,)
        Target speed at each point [m/s].
    file : pathlib.Path or str
        Where to save the PNG (blue = slowest, red = fastest).
    """
    meta = yaml.safe_load(open(sim.MAP_YAML))
    image = Image.open(sim.MAP_YAML.parent / meta['image']).convert('RGB')
    height, resolution = image.height, meta['resolution']
    origin_x, origin_y = meta['origin'][:2]
    image = image.resize((image.width * SCALE, height * SCALE), Image.NEAREST)
    pen = ImageDraw.Draw(image)

    def pixel(point):                       # map (x, y) [m] -> image (column, row)
        """Convert a map point (x, y) [m] to a pixel (column, row) in the scaled image."""
        return ((point[0] - origin_x) / resolution * SCALE,
                (height - (point[1] - origin_y) / resolution) * SCALE)

    low, high = speeds.min(), speeds.max()
    for i in range(len(path)):
        share = (speeds[i] - low) / max(high - low, 1e-9)     # 0 = slowest, 1 = fastest
        colour = (int(255 * share), 0, int(255 * (1 - share)))
        pen.line([pixel(path[i]), pixel(path[(i + 1) % len(path)])], fill=colour, width=5)
    pen.text((10, 10), f'blue {low:.1f} m/s ... red {high:.1f} m/s', fill=(0, 0, 0))
    image.save(file)


if __name__ == '__main__':
    main()
