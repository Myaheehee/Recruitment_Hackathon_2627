#!/usr/bin/env python3
"""YOUR DRIVER GOES HERE.

This is the node the judges run. Keep `driver` as the executable name and
`/drive` as the output topic and everything else is yours to change - rewrite
this file completely if you want to.

--------------------------------------------------------------------------
THIS TEMPLATE DOES NOT DRIVE
--------------------------------------------------------------------------
It is wiring, not a driver. It connects to the simulator, subscribes to the
sensors, and then asks for a slow constant speed with the wheels straight. It
will set off from the grid and into the first thing in front of it. That is
deliberate and it is the whole point: **there is no algorithm here and no
algorithm is shipped anywhere else in this repository.** Writing one is the
hackathon.

What the template is good for is proving your setup works. If the car moves
when you run it, then the image, the bridge, the workspace, the topics and
your commands are all correct, and every problem left is yours.

    ros2 run team_driver driver
    ros2 launch team_driver driver.launch.py

`docs/04-algorithms.md` lists the approaches worth starting from - reactive
ones that need nothing but the LiDAR, planners that follow a line, model-based
control, and learned policies - with what each needs and where each breaks.
Pick one and replace `plan()` below.

What you are allowed to read (see docs/06-rules.md):
    /scan               LiDAR, 819 beams over 270 degrees
    /ego_racecar/odom   ground-truth pose and velocity - ALLOWED and RECOMMENDED
    TF, /map            the static map
What you publish:
    /drive              AckermannDriveStamped - you ask for a SPEED, and the
                        simulator closes that loop for you
    /driver/...         anything of your own, for visualisation
"""

import math
import os

import numpy as np
import rclpy
import yaml
from PIL import Image
from scipy import ndimage
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker, MarkerArray

CAR_HALF_WIDTH = 0.155   # [m] the car is 0.31 m wide (docs/02-simulator.md, 2.8)
SEAL_RADIUS = 0.40       # [m] gaps in the walls narrower than this are treated as wall


class Driver(Node):

    def __init__(self):
        super().__init__('driver')

        # Declared parameters can be retuned without editing code:
        #   ros2 run team_driver driver --ros-args -p v_max:=4.0
        # Add your own as you go; config/driver_params.yaml loads them.
        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('odom_topic', '/ego_racecar/odom')
        self.declare_parameter('drive_topic', '/drive')
        self.declare_parameter('v_max', 7.0)              # [m/s] top speed
        self.declare_parameter('a_brake', 6.05)           # [m/s^2] braking we plan with
        self.declare_parameter('margin', 0.5)             # [m] stop this far short of a wall
        self.declare_parameter('mu', 1.0489)              # tyre friction (the sim's value)
        self.declare_parameter('ahead_angle', 10.0)       # [deg] half-width of the "ahead" cone
        self.declare_parameter('max_range', 8.0)          # [m] clip the scan here
        self.declare_parameter('lookahead_gain', 0.4)      # [s] lookahead = gain * speed
        self.declare_parameter('min_lookahead', 0.5)       # [m] lookahead never shorter than this
        self.declare_parameter('wheelbase', 0.33)          # [m]
        self.declare_parameter('max_steer', 0.4189)        # [rad]
        self.declare_parameter('centerline_csv', '/hackathon/maps/icra26_centerline.csv')
        self.declare_parameter('map_yaml', '/hackathon/maps/icra26.yaml')
        self.declare_parameter('wall_gap', 0.2)            # [m] air between the car's side and the wall
        self.declare_parameter('raceline_iters', 20000)    # smoothing passes; more = smoother, slower start
        self.declare_parameter('raceline_csv', '/tmp/raceline.csv')
        self.declare_parameter('use_raceline', True)       # False = follow the plain centreline

        self.v_max = self.get_parameter('v_max').value
        self.a_brake = self.get_parameter('a_brake').value
        self.margin = self.get_parameter('margin').value
        self.mu = self.get_parameter('mu').value
        self.ahead_angle = math.radians(self.get_parameter('ahead_angle').value)
        self.max_range = self.get_parameter('max_range').value
        self.lookahead_gain = self.get_parameter('lookahead_gain').value
        self.min_lookahead = self.get_parameter('min_lookahead').value
        self.wheelbase = self.get_parameter('wheelbase').value
        self.max_steer = self.get_parameter('max_steer').value

        # The shipped centreline: an (N, 2) array of x, y points in the map
        # frame, in driving order. Lines starting with '#' are comments.
        centreline = np.loadtxt(
            self.get_parameter('centerline_csv').value, delimiter=',', comments='#')

        # Build our own racing line from the map: a function that tells us
        # how far any point is from the nearest wall, then a smoother that
        # slides the centreline sideways to take the bends as gently as the
        # walls allow. The line must keep the car's half width plus `wall_gap`
        # of clearance everywhere.
        clearance = self.load_clearance_map(self.get_parameter('map_yaml').value)
        margin = CAR_HALF_WIDTH + self.get_parameter('wall_gap').value
        self.raceline = self.build_raceline(centreline, clearance, margin)

        # Keep a copy on disk so you can plot it or load it elsewhere.
        csv_path = self.get_parameter('raceline_csv').value
        try:
            np.savetxt(csv_path, self.raceline, delimiter=',', fmt='%.4f',
                       header='x_m, y_m\nRacing line built by team_driver at startup '
                              'from the map and the centreline.')
        except OSError as err:
            self.get_logger().warn(f'could not save the racing line to {csv_path}: {err}')

        # The path pure pursuit follows.
        self.path = self.raceline if self.get_parameter('use_raceline').value else centreline

        # Both lines drawn for RViz: centreline in yellow, racing line in red.
        # They never change, so build them once; publish_marker() sends them.
        self.centre_marker = self.line_marker(centreline, marker_id=1, colour=(1.0, 0.8, 0.0))
        self.raceline_marker = self.line_marker(self.raceline, marker_id=2, colour=(1.0, 0.0, 0.0))

        # Latest known pose and speed. Ground truth from the simulator, which
        # the rules allow you to use - so use it.
        self.position = None      # (x, y) in the map frame
        self.yaw = 0.0            # [rad]
        self.speed = 0.0          # [m/s]

        self.drive_pub = self.create_publisher(
            AckermannDriveStamped, self.get_parameter('drive_topic').value, 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/driver/markers', 1)

        self.create_subscription(
            LaserScan, self.get_parameter('scan_topic').value, self.scan_callback, 10)
        self.create_subscription(
            Odometry, self.get_parameter('odom_topic').value, self.odom_callback, 10)

        self._marker_divisor = 0
        self.get_logger().warn(
            'team_driver is up, but this template has no driving logic: it will '
            'crawl straight ahead until it hits something. Implement plan().')

    # ------------------------------------------------------------------
    # The map, read once: how far is any point from the nearest wall?
    # ------------------------------------------------------------------
    def load_clearance_map(self, map_yaml):
        """Read the map image and return a function clearance(points) that
        gives, for any (x, y) in the map frame, the distance in metres to the
        nearest wall.

        Cone rows are sealed first. The gaps between cones are wide enough to
        fit the car, but driving through one is against the rules (rule 11),
        so for us a row of cones has to count as one solid wall.
        """
        with open(map_yaml) as f:
            meta = yaml.safe_load(f)
        image_path = os.path.join(os.path.dirname(map_yaml), meta['image'])
        image = np.array(Image.open(image_path))   # rows x columns, 0 = black
        resolution = meta['resolution']            # metres per pixel
        origin_x, origin_y = meta['origin'][0], meta['origin'][1]
        height = image.shape[0]

        # True where the map is a wall. The map's yaml says a pixel is
        # occupied when (255 - value) / 255 > occupied_thresh.
        wall = (255.0 - image) / 255.0 > meta['occupied_thresh']

        # Seal the cone rows. "Closing" an image means: grow every wall by a
        # disc, then shrink it back by the same disc. Two walls that touched
        # while grown stay joined after shrinking, so any gap the disc cannot
        # pass through gets filled in. Cones are 0.25-0.5 m apart, so a
        # 0.4 m disc bridges every cone gap and leaves the corridors, which
        # are all over 1 m wide, alone.
        radius = int(round(SEAL_RADIUS / resolution))            # in pixels
        yy, xx = np.ogrid[-radius:radius + 1, -radius:radius + 1]
        disc = xx * xx + yy * yy <= radius * radius
        wall = ndimage.binary_closing(wall, structure=disc)

        # For every pixel, the distance to the nearest wall pixel, in metres.
        clearance_grid = ndimage.distance_transform_edt(~wall) * resolution

        def clearance(points):
            """points: an (..., 2) array of x, y -> metres to the nearest wall."""
            # World (x, y) -> pixel (row, col). Image rows count down from the
            # top while map y counts up, hence the flip. Pixel centres sit on
            # whole numbers, so the -0.5 puts a point in the right place
            # between them.
            cols = (points[..., 0] - origin_x) / resolution - 0.5
            rows = height - 1 - ((points[..., 1] - origin_y) / resolution - 0.5)
            # Look every point up in one call, blending the 4 nearest pixels
            # (order=1). Off the map (cval) counts as touching a wall.
            return ndimage.map_coordinates(
                clearance_grid, [rows, cols], order=1, mode='constant', cval=0.0)

        return clearance

    # ------------------------------------------------------------------
    # The racing line, built once from the centreline and the map.
    # ------------------------------------------------------------------
    def build_raceline(self, centre, clearance, margin):
        """Slide the centreline's points sideways so the lap bends as little
        as possible, without ever coming closer than `margin` to a wall.

        Why: the speed the car can carry through a bend depends on how tight
        the bend is (plan() step 6b), so the line that bends least is the
        line that can be driven fastest. Wide in, clip the inside, wide out
        is exactly what a bend-minimising line looks like.

        How: each point may only move sideways (at right angles to the
        track), so the whole line is described by one number per point, how
        far it has slid. A simple loop then nudges every point a little in
        whichever sideways direction reduces the total bending, clamps it to
        where the car still fits, and repeats thousands of times.
        """
        n = len(centre)

        # 1. The direction of travel at every point, and the direction that
        #    is "left" of it. np.roll shifts the array by one place, so
        #    np.roll(centre, -1)[i] is the next point and np.roll(centre, 1)[i]
        #    the previous one; it wraps round because the lap is a loop.
        tangent = np.roll(centre, -1, axis=0) - np.roll(centre, 1, axis=0)
        tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)    # make length 1
        left = np.stack([-tangent[:, 1], tangent[:, 0]], axis=1)       # turn 90 degrees anticlockwise

        # 2. How far may each point slide? Lay out sample points sideways
        #    from every centreline point, 2.5 cm apart, up to 3 m each way,
        #    and ask the map how far each sample is from a wall. A point may
        #    slide up to the last sample at which the car still has `margin`.
        step = 0.025
        reach = int(3.0 / step)
        shifts = np.arange(-reach, reach + 1) * step                    # -3.0, -2.975, ..., +3.0 (left)
        zero = reach                                                     # index of shift 0, the centreline itself
        samples = centre[:, None, :] + shifts[None, :, None] * left[:, None, :]   # shape (n, shifts, 2)
        fits = clearance(samples) >= margin                              # shape (n, shifts), True = car fits

        def steps_until_blocked(mask):
            # How many True in a row at the start of each row of `mask`.
            blocked = ~mask
            return np.where(blocked.any(axis=1), blocked.argmax(axis=1), mask.shape[1])

        # Going left from the centreline is fits[:, zero:]; going right is the
        # same columns read backwards. The "- 1" is because the count includes
        # the centreline sample itself, and the maximum with 0 means a point
        # that is already too close to a wall simply stays where it is.
        max_left = np.maximum((steps_until_blocked(fits[:, zero:]) - 1) * step, 0.0)
        max_right = np.maximum((steps_until_blocked(fits[:, :zero + 1][:, ::-1]) - 1) * step, 0.0)

        # 3. The smoothing loop. `shift` is how far each point has slid to
        #    the left (negative = to the right); everyone starts at 0.
        #    `rate` is how big each nudge is: 0.12 is just under the largest
        #    value that does not make the loop blow up.
        rate = 0.12
        shift = np.zeros(n)
        for _ in range(self.get_parameter('raceline_iters').value):
            points = centre + shift[:, None] * left
            # How far each point sticks out from the straight line through its
            # two neighbours. Zero on a straight, bigger the tighter the bend.
            # The total bending we want to minimise is the sum of bend**2.
            bend = np.roll(points, 1, axis=0) - 2.0 * points + np.roll(points, -1, axis=0)
            # Moving point i changes three bends: its own (where it counts
            # twice, with a minus sign) and those of both its neighbours.
            # Adding them up with those weights gives the direction in which
            # moving point i makes the total bending grow fastest ...
            push = np.roll(bend, 1, axis=0) - 2.0 * bend + np.roll(bend, -1, axis=0)
            # ... so nudge it the opposite way. Only the sideways part of the
            # push counts, because sideways is the only way a point may move.
            shift -= rate * np.sum(push * left, axis=1)
            # And never further than the walls allow.
            shift = np.clip(shift, -max_right, max_left)

        raceline = centre + shift[:, None] * left

        # 4. The points have bunched up on the inside of bends and spread out
        #    on the outside; space them out evenly again.
        raceline = self.resample(raceline, spacing=0.2)

        # 5. Belt and braces: check the finished line against the map. The
        #    0.05 m allows for the pixel grid and the resampling.
        worst = clearance(raceline).min()
        length = np.linalg.norm(np.roll(raceline, -1, axis=0) - raceline, axis=1).sum()
        self.get_logger().info(
            f'racing line: {len(raceline)} points, {length:.1f} m per lap '
            f'(centreline {len(centre)} points), closest approach to a wall {worst:.2f} m')
        if worst < margin - 0.05:
            self.get_logger().error(
                'racing line comes too close to a wall; following the centreline instead')
            return centre
        return raceline

    @staticmethod
    def resample(points, spacing):
        """The same closed loop, redrawn as evenly spaced points."""
        closed = np.vstack([points, points[:1]])                   # repeat point 0 to close the loop
        segment = np.linalg.norm(np.diff(closed, axis=0), axis=1)  # length of each step along the loop
        along = np.concatenate([[0.0], np.cumsum(segment)])        # distance travelled to reach each point
        count = int(round(along[-1] / spacing))
        wanted = np.linspace(0.0, along[-1], count, endpoint=False)  # where the new points go
        # np.interp reads off x (and then y) at each wanted distance.
        return np.stack([np.interp(wanted, along, closed[:, 0]),
                         np.interp(wanted, along, closed[:, 1])], axis=1)

    def line_marker(self, points, marker_id, colour):
        """A closed loop drawn as a thin line in RViz. `colour` is (r, g, b), 0..1."""
        marker = Marker()
        marker.header.frame_id = 'map'
        marker.ns = 'team_driver'
        marker.id = marker_id
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.05                                      # line width [m]
        marker.color.r, marker.color.g, marker.color.b = colour
        marker.color.a = 1.0
        marker.pose.orientation.w = 1.0
        for x, y in np.vstack([points, points[:1]]):              # repeat point 0 to close the loop
            marker.points.append(Point(x=float(x), y=float(y)))
        return marker

    # ------------------------------------------------------------------
    # Odometry: where the car is. Free, accurate, and worth building on.
    # ------------------------------------------------------------------
    def odom_callback(self, msg):
        self.position = (msg.pose.pose.position.x, msg.pose.pose.position.y)
        q = msg.pose.pose.orientation
        self.yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                              1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        self.speed = math.hypot(msg.twist.twist.linear.x, msg.twist.twist.linear.y)

    # ------------------------------------------------------------------
    # The control loop, once per LiDAR scan (about 40 Hz).
    # ------------------------------------------------------------------
    def scan_callback(self, scan):
        ranges, angles = self.preprocess(scan)
        steering, speed = self.plan(ranges, angles)
        self.publish(steering, speed)

        self._marker_divisor = (self._marker_divisor + 1) % 10
        if self._marker_divisor == 0:
            self.publish_marker(steering)

    def preprocess(self, scan):
        """Turn a raw scan into clean ranges plus the angle of each beam.

        Kept because every approach needs some version of it and the details
        are fiddly rather than interesting: the LiDAR reports NaN and inf, and
        arithmetic on those propagates silently through everything downstream.
        """
        ranges = np.asarray(scan.ranges, dtype=np.float64)
        ranges = np.nan_to_num(ranges, nan=0.0, posinf=self.max_range, neginf=0.0)
        ranges = np.clip(ranges, 0.0, self.max_range)

        angles = scan.angle_min + np.arange(len(ranges)) * scan.angle_increment
        return ranges, angles

    # ==================================================================
    # THIS IS THE PART YOU WRITE.
    # ==================================================================
    def plan(self, ranges, angles):
        """Decide what the car should do, given the latest scan.

        Returns `(steering, speed)`: a steering angle in radians, and a speed
        in m/s. Unlike Track 2, the speed is a genuine request - the simulator
        runs the controller that achieves it - so you can think in the units
        your algorithm naturally produces.

        Right now it returns "straight ahead, slowly", which is not driving:
        it ignores `ranges` entirely, so the car will hold its heading off the
        grid and put itself into the first wall it meets. Replace the whole
        method.

        You have more to work with than the scan. `self.position`, `self.yaw`
        and `self.speed` are ground truth from /ego_racecar/odom and are yours
        to use, the occupancy grid is published on /map, and nothing stops you
        subscribing to more topics, loading a line you computed offline, or
        running a policy you trained. See docs/04-algorithms.md for the
        approaches and what each one needs
        """
        # No odometry yet, so we don't know where we are: stay still.
        if self.position is None:
            return 0.0, 0.0

        car = np.array(self.position)

        # 1. Find the path point closest to the car.
        distances = np.linalg.norm(self.path - car, axis=1) #distance of car from each point in self.path
        i = int(np.argmin(distances))

        # 2. Look further ahead the faster we go, but never less than
        #    min_lookahead (at standstill 0.4 * 0 would be 0 m).
        lookahead = max(self.min_lookahead, self.lookahead_gain * self.speed)

        ''' 3. Walk forward along the path until a point is at least one
            lookahead distance away. The % wraps round at the end of the lap. '''
        while distances[i] < lookahead:
            i = (i + 1) % len(self.path)
        target = self.path[i]

        # 4. Express the target in the car's frame (x forward, y left).
        dx, dy = target - car
        local_y = -math.sin(self.yaw) * dx + math.cos(self.yaw) * dy

        # 5. Pure pursuit: the arc from the rear axle through the target has
        #    curvature 2*y / L^2, and a bicycle with wheelbase W follows
        #    curvature k with steering angle atan(W * k).
        curvature = 2.0 * local_y / lookahead ** 2
        steering = math.atan(self.wheelbase * curvature)
        steering = max(-self.max_steer, min(self.max_steer, steering))

        # 6. Speed: the smallest of three limits.
        #    a) Braking: the free distance straight ahead is the shortest beam
        #       in a narrow cone in front. A car braking at a_brake needs
        #       v^2 / (2 * a_brake) metres to stop, so the fastest speed that
        #       can still stop `margin` short of that wall is sqrt(2*a*d).
        ahead = ranges[np.abs(angles) < self.ahead_angle].min()
        v_brake = math.sqrt(2.0 * self.a_brake * max(ahead - self.margin, 0.0))

        #    b) Cornering: on a circle of radius R the tyres can hold at most
        #       mu*g sideways, and sideways acceleration is v^2 / R, so
        #       v = sqrt(R * mu * g). R is the radius of the arc pure pursuit
        #       just chose (R = 1 / curvature); a straight line has no limit.
        if curvature == 0.0:
            v_corner = float('inf')
        else:
            v_corner = math.sqrt(self.mu * 9.81 / abs(curvature))

        speed = min(self.v_max, v_brake, v_corner)

        return steering, speed

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    def publish(self, steering, speed):
        msg = AckermannDriveStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.drive.steering_angle = float(steering)
        msg.drive.speed = float(speed)
        self.drive_pub.publish(msg)

    def publish_marker(self, target_angle):
        """Draw where the car thinks it is going. Add /driver/markers in RViz."""
        marker = Marker()
        marker.header.frame_id = 'ego_racecar/base_link'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'team_driver'
        marker.id = 0
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        marker.scale.x, marker.scale.y, marker.scale.z = 1.5, 0.15, 0.15
        marker.color.g, marker.color.b, marker.color.a = 0.8, 1.0, 0.9
        marker.pose.orientation.z = math.sin(target_angle / 2.0)
        marker.pose.orientation.w = math.cos(target_angle / 2.0)
        array = MarkerArray()
        array.markers.append(marker)
        array.markers.append(self.centre_marker)
        array.markers.append(self.raceline_marker)
        self.marker_pub.publish(array)


def main(args=None):
    rclpy.init(args=args)
    node = Driver()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

#stopping the program and resetting car position

#   Press Ctrl+C in Terminal 2. 
#If you've lost track of where it's running, use pkill -f team_driver.

#   ros2 topic pub --once /drive ackermann_msgs/msg/AckermannDriveStamped "{drive: {speed: 0.0, steering_angle: 0.0}}"

#ros2 topic pub --once /initialpose geometry_msgs/msg/PoseWithCovarianceStamped \
#"{header: {frame_id: map}, pose: {pose: {position: {x: -1.68, y: -0.01}, orientation: {w: 1.0}}}}"

#ros2 topic echo /ego_racecar/odom --no-arr