#!/usr/bin/env python3
"""The ROS 2 node that drives the car around the track with pure pursuit.

ROS 2 (Robot Operating System 2) lets separate programs, called nodes,
talk by sending messages on named channels, called topics: a node
"publishes" on a topic and any node that "subscribes" to it receives the
messages. This file is the team's only node. The real work lives in
modules with no ROS code (raceline.py, pure_pursuit.py, policies.py), so
that training/ can run exactly the same maths; this file connects them to
the simulator's topics.

Pure pursuit means: steer towards a point a little way ahead on the
racing line. The car senses the walls with a LiDAR (a laser scanner that
measures the distance to the nearest obstacle in many directions) and
gets its position from odometry (the simulator's report of where the car
is and how it is moving).

What happens, and where to read about it:
    at startup     build the racing line from the map             raceline.py
                   and its speed table (if use_speed_table): the
                   tuned one if trained, else from mu and a_brake raceline.py
                   load the lookahead policy, if there is one     policies.py
    on odometry    remember where the car is and how fast         odom_callback()
    on each scan   the lookahead policy (if loaded) picks how
                   far ahead to aim                               policies.py
                   pure pursuit picks steering and speed          pure_pursuit.py
                   send the command on /drive                     scan_callback()

Which learned parts run. Two (a policy is the learned rule that makes a
decision): the lookahead policy (models/lookahead.npz) and the tuned speed
table (models/speed_table.npz). Each is used when its file is there; the
speed table only if it was tuned on the same racing line as the one built
at startup.

Topics
    in    /scan               LiDAR, 819 beams over 270 degrees
          /ego_racecar/odom   true position and speed (allowed by the rules)
    out   /drive              steering angle + target speed; the simulator holds the speed
          /driver/markers     the lines and a steering arrow, for RViz

Settings. Every number in pure_pursuit.DEFAULTS becomes a ROS parameter
(a named setting that can be changed when the node starts) with that
default. The judges start the car with `ros2 run team_driver driver` and
no parameter file, so pure_pursuit.DEFAULTS are the race settings.
config/driver_params.yaml is only read when it is passed in, which
`ros2 launch team_driver driver.launch.py` does automatically.

Run it:              ros2 run team_driver driver
Change a setting:    ros2 run team_driver driver --ros-args -p v_max:=6.0
                     (write 6.0, not 6: if the default has a decimal point,
                     a whole number stops the node at startup with an
                     InvalidParameterTypeException)

Handy while testing:
    stop the car      ros2 topic pub --once /drive ackermann_msgs/msg/AckermannDriveStamped \
                          "{drive: {speed: 0.0, steering_angle: 0.0}}"
    back to the grid  ros2 topic pub --once /initialpose geometry_msgs/msg/PoseWithCovarianceStamped \
                          "{header: {frame_id: map}, pose: {pose: {position: {x: -1.68, y: -0.01}, orientation: {w: 1.0}}}}"
    watch odometry    ros2 topic echo /ego_racecar/odom --no-arr
    lost the driver?  pkill -f team_driver
"""

import math
import os

import numpy as np
import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker, MarkerArray

from team_driver.pure_pursuit import DEFAULTS, preprocess, pure_pursuit
from team_driver.raceline import (
    CAR_HALF_WIDTH, ITERS, WALL_GAP, build_raceline, lap_time, load_clearance_map, speed_table)
from team_driver.policies import (
    LOOKAHEAD_INPUTS, LOOKAHEAD_RANGE, load_lookahead, load_speed_table, lookahead,
    lookahead_inputs, path_geometry)

SIM_STEP = 0.01           # [s] one physics step of the race simulator
DELAY_REPORT_EVERY = 200  # scans between reaction-delay reports in the log


class Driver(Node):
    """The ROS 2 node "driver": answers each LiDAR scan it receives with a drive command.

    The work is split so that each scan costs as little as possible:

    startup (``__init__``, once)
        Read the settings, build the racing line from the map, work out
        the line's curvature, build the speed table, load the
        learned policy, then create the topics. All the slow work (20000
        line-smoothing steps by default) happens here, before the car moves.
    tick (``scan_callback``, once per LiDAR scan)
        Clean the scan, ``plan()`` a (steering, speed) command,
        ``publish()`` it on /drive, log the reaction delay, and on every
        10th scan send markers to RViz (the ROS 3-D viewer). "Tick" means
        one pass of this loop.
    odometry (``odom_callback``, whenever odometry arrives)
        Only stores the car's position, heading and speed for the next
        tick. It never sends a command.

    A callback is a function that ROS calls for you when a message
    arrives. ``rclpy.spin()`` in ``main()`` runs the callbacks one at a
    time, so the odometry can never change in the middle of ``plan()``.

    Topics (the first three names are parameters; these are the defaults)
        subscribes  /scan              sensor_msgs/LaserScan, queue depth 1
                    /ego_racecar/odom  nav_msgs/Odometry, queue depth 1
        publishes   /drive             ackermann_msgs/AckermannDriveStamped, queue depth 10
                    /driver/markers    visualization_msgs/MarkerArray, queue depth 1
    Queue depth 1 means only the newest unread message is kept.

    Attributes
    ----------
    params : dict
        The pure pursuit settings (the keys of pure_pursuit.DEFAULTS),
        read once at startup.
    path : (N, 2) array
        The line the car follows [m]: the racing line (380 points, 0.2 m
        apart, on icra26), or the centreline if use_raceline is false.
    curvature : (N,) array
        Curvature of the line at each point [1/m]: 1 / turn radius,
        positive when the line turns left.
    speeds : (N,) array or None
        Target speed at each point [m/s], or None if use_speed_table is false.
    lookahead_policy : dict or None
        {'w': (5,) array, 'b': float} from models/lookahead.npz, or None.
    position : tuple of 2 floats or None
        (x, y) of the rear axle [m]; None until the first odometry.
    yaw : float
        The car's heading [rad].
    speed : float
        The car's speed [m/s].
    line_markers : list of visualization_msgs.msg.Marker
        The centreline and the racing line, for RViz.
    """

    def __init__(self):
        """Start the node: build everything once, then open the topics.

        The order matters, because each step uses what the one before made:

        1. setup_parameters(): every later step reads a setting.
        2. build_path(): the racing line, from the map and the centreline.
        3. path_geometry(): the line's curvature. The speed table and the
           lookahead policy's inputs both need it.
        4. build_speed_table(): needs the line and its curvature, and
           checks that a tuned table was trained on this same line.
        5. load_policy(): the lookahead policy.
        6. The car's state, starting as "no odometry yet" (position None).
        7. Publishers and subscriptions, last.

        ROS only runs the callbacks once main() calls rclpy.spin(), which
        is after __init__ has returned, so no message can reach a
        half-built node. Nothing is published until then, so the car waits
        on the grid while the line is built.
        """
        super().__init__('driver')
        self.params = self.setup_parameters()
        self.path = self.build_path()
        _, self.curvature = path_geometry(self.path)
        self.speeds = self.build_speed_table()
        self.load_policy()

        # The car's latest state, from odometry.
        self.position = None      # (x, y) of the rear axle; None until odometry arrives
        self.yaw = 0.0            # [rad] heading
        self.speed = 0.0          # [m/s]

        # Reaction-delay measurement (see measure_delay()).
        self.odom_stamp = None    # [s] simulation time of the newest odometry
        self.last_scan_stamp = None
        self.ages, self.gaps = [], []

        self.drive_pub = self.create_publisher(AckermannDriveStamped, self.param('drive_topic'), 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/driver/markers', 1)
        # Queue length 1: if messages pile up (the simulator runs as fast as
        # it can, faster than real time when headless), keep only the newest.
        # Steering from an old scan is exactly the reaction delay that makes
        # the car late into corners.
        self.create_subscription(LaserScan, self.param('scan_topic'), self.scan_callback, 1)
        self.create_subscription(Odometry, self.param('odom_topic'), self.odom_callback, 1)
        self.scan_count = 0
        self.get_logger().info('driver ready')

    def param(self, name):
        """Return the current value of one ROS parameter.

        The driver only reads parameters during startup, so changing one
        with `ros2 param set` while it runs has no effect: restart it.

        Parameters
        ----------
        name : str
            A parameter declared in setup_parameters().

        Returns
        -------
        bool, int, float or str
            Its value: the default, unless it was set with
            `--ros-args -p name:=value` or a parameter file.
        """
        return self.get_parameter(name).value

    # ------------------------------------------------------------------
    # Startup
    # ------------------------------------------------------------------
    def setup_parameters(self):
        """Declare every ROS parameter, and return the pure pursuit ones.

        Their defaults live in pure_pursuit.DEFAULTS because the training
        code needs the same numbers. The other parameters (topic names,
        file paths and on/off switches) are read with param() when needed
        and are not in the returned dict, so warn_if_retuned() ignores them.

        Declaring a parameter with a default also fixes its type: v_max
        (default 6.8) only accepts a value written with a decimal point
        (7.0, not 7).

        Returns
        -------
        dict
            {name: value} for every key of pure_pursuit.DEFAULTS: the
            default, or the value given on the command line or in a
            parameter file.
        """
        for name, value in DEFAULTS.items():
            self.declare_parameter(name, value)
        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('odom_topic', '/ego_racecar/odom')
        self.declare_parameter('drive_topic', '/drive')
        self.declare_parameter('centerline_csv', '/hackathon/maps/icra26_centerline.csv')
        self.declare_parameter('map_yaml', '/hackathon/maps/icra26.yaml')
        self.declare_parameter('wall_gap', WALL_GAP)       # [m] air between the car's side and a wall
        self.declare_parameter('raceline_iters', ITERS)    # more = smoother line, slower startup
        self.declare_parameter('raceline_csv', '/tmp/raceline.csv')   # a copy of the line, for plotting
        self.declare_parameter('use_raceline', True)       # False = follow the plain centreline
        self.declare_parameter('use_lookahead', True)      # False = ignore models/lookahead.npz
        self.declare_parameter('lookahead_weights', '')    # blank = models/lookahead.npz in this package
        self.declare_parameter('speed_table_weights', '')  # blank = models/speed_table.npz in this package
        return {name: self.param(name) for name in DEFAULTS}

    def build_path(self):
        """Build the racing line (raceline.py) and return the line to follow.

        Reads the centreline (centerline_csv) and the map (map_yaml); the
        defaults are the paths inside the judging container, under
        /hackathon/maps. The racing line aims to keep the middle of the car
        at least CAR_HALF_WIDTH + wall_gap (0.155 + 0.2 = 0.355 m by
        default) from every wall; if the finished line comes too close,
        build_raceline() gives back the centreline instead.

        Two side effects: a copy of the racing line is saved to
        raceline_csv (only a warning if that fails), and self.line_markers
        is set for RViz. The markers always show the racing line, even
        when use_raceline is false.

        Returns
        -------
        (N, 2) array
            Points to follow [m], in driving order: the racing line, or
            the centreline if use_raceline is false.
        """
        # The shipped centreline: x, y points in driving order.
        centreline = np.loadtxt(self.param('centerline_csv'), delimiter=',', comments='#')
        clearance = load_clearance_map(self.param('map_yaml'))
        margin = CAR_HALF_WIDTH + self.param('wall_gap')
        raceline = build_raceline(centreline, clearance, margin, self.param('raceline_iters'),
                                  log=self.get_logger().info)

        try:
            np.savetxt(self.param('raceline_csv'), raceline, delimiter=',', fmt='%.4f',
                       header='x_m, y_m\nRacing line built by team_driver at startup.')
        except OSError as err:
            self.get_logger().warn(f'could not save the racing line: {err}')

        # For RViz: centreline in yellow, racing line in red.
        self.line_markers = [line_marker(centreline, 1, (1.0, 0.8, 0.0)),
                             line_marker(raceline, 2, (1.0, 0.0, 0.0))]

        return raceline if self.param('use_raceline') else centreline

    def build_speed_table(self):
        """Build the speed table: one target speed per point of the line.

        Returns None instead to use pure pursuit's three speed limits. The
        tuned table from training (models/speed_table.npz) if there is one
        and it fits this racing line; otherwise worked out from mu and
        a_brake (raceline.speed_table).

        "Fits" means the line stored in the .npz has the same number of
        points as self.path and every coordinate agrees to about 1 mm
        (np.allclose with atol 1e-3). A different wall_gap, raceline_iters
        or map gives a different line, so the tuned table is then refused
        with a warning. Only an accepted tuned table is checked by
        warn_if_retuned(). Either table is then multiplied by table_scale.

        Returns
        -------
        (N,) array or None
            Target speed at each point of self.path [m/s], or None if
            use_speed_table is false.
        """
        if not self.params['use_speed_table']:
            self.get_logger().info('speed from pure pursuit\'s limits (use_speed_table is off)')
            return None
        speeds, source = None, 'worked out from mu and a_brake'
        path = self.policy_file('speed_table')
        if path:
            tuned, line, trained_with = load_speed_table(path)
            if line.shape == self.path.shape and np.allclose(line, self.path, atol=1e-3):
                speeds, source = tuned, f'tuned, from {path}'
                self.warn_if_retuned(trained_with)
            else:
                self.get_logger().warn(f'{path} was tuned on a different racing line '
                                       '(wall_gap or the map changed?): not using it')
        if speeds is None:
            speeds = speed_table(self.path, self.curvature, self.params)
        # Slow the whole table down by table_scale. Scaling every speed down
        # never breaks the neighbour rule, so the table stays within it.
        speeds = speeds * self.params['table_scale']
        self.get_logger().info(
            f'speed table {source}, scaled by {self.params["table_scale"]}: '
            f'{speeds.min():.2f} to {speeds.max():.2f} m/s, '
            f'{lap_time(self.path, speeds):.2f} s per lap if driven exactly')
        return speeds

    def load_policy(self):
        """Load the trained lookahead policy (see training/), if there is one.

            models/lookahead.npz   chooses pure pursuit's lookahead distance

        Without it, pure pursuit uses its own rule
        (lookahead = lookahead_gain * speed).

        Sets self.lookahead_policy (None when not loaded).

        Returns
        -------
        None
        """
        self.lookahead_policy = None

        path = self.policy_file('lookahead')
        if path:
            self.lookahead_policy, trained_with = load_lookahead(path)
            w, b = self.lookahead_policy['w'], self.lookahead_policy['b']
            equation = ' '.join(f'{x:+.3f}*{name}' for x, name in zip(w, LOOKAHEAD_INPUTS))
            self.get_logger().info(f'lookahead policy loaded from {path}: L = {b:.3f} {equation}')
            self.warn_if_retuned(trained_with)
            return

        self.get_logger().info('no trained lookahead policy: lookahead_gain * speed')

    def policy_file(self, kind):
        """Find a trained policy's .npz file.

        Where the `kind` ('lookahead' or 'speed_table') policy
        is, or None if it is switched off (use_<kind>:=false) or has not been
        trained yet.

        The default place is models/<kind>.npz in the package's installed
        share/ folder, not the source folder: a newly trained file has to
        be copied into race_ws/src/team_driver/models/ and installed with
        `colcon build`. A path given with <kind>_weights that does not exist
        is not an error: it is treated like a missing file.

        Parameters
        ----------
        kind : str
            'lookahead' or 'speed_table'.

        Returns
        -------
        str or None
            Path to the .npz file, or None.
        """
        if not self.param(f'use_{kind}'):
            return None
        path = self.param(f'{kind}_weights') or os.path.join(
            get_package_share_directory('team_driver'), 'models', f'{kind}.npz')
        return path if os.path.isfile(path) else None

    def warn_if_retuned(self, trained_with):
        """Log a warning if pure pursuit's settings differ from training.

        A policy learned to work with pure pursuit tuned the way it was
        during training. If the tuning has changed since, retrain it.

        Every key of self.params is compared; a key the file does not have
        counts as changed, and keys only in the file are ignored. It is
        only a warning: the policy is still used. Both shipped .npz files
        were saved before table_scale and speed_lead existed, so every
        start warns about those two (harmless while they keep their
        neutral defaults 1.0 and 0.0). Starting with
        config/driver_params.yaml adds v_max, a_brake, margin and
        lookahead_gain to the list.

        Parameters
        ----------
        trained_with : dict
            {name: value} of the pure pursuit settings stored in the .npz.

        Returns
        -------
        None
        """
        changed = [k for k in self.params if trained_with.get(k) != self.params[k]]
        if changed:
            self.get_logger().warn(f'pure pursuit settings changed since the policy was '
                                   f'trained: {changed}. Retrain it (training/README.md).')

    # ------------------------------------------------------------------
    # Every message
    # ------------------------------------------------------------------
    def odom_callback(self, msg):
        """Store the car's latest position, heading and speed.

        Called for every message on /ego_racecar/odom: the simulator's exact
        ("ground truth") state, which the rules allow. Nothing is sent from
        here; the next scan_callback() uses these values.

        Fields read
            pose.pose.position.x, .y   position of the rear axle [m], in the map frame
            pose.pose.orientation      heading, as a quaternion (x, y, z, w)
            twist.twist.linear.x, .y   velocity along and across the car [m/s]
            header.stamp               simulation time, for measure_delay()

        The position is the rear axle (the simulator moves its centre-of-
        mass position back to it before publishing), which is the point
        pure pursuit's geometry assumes.

        A quaternion stores a rotation as four numbers. For a car that only
        turns about the vertical axis, x = y = 0, z = sin(yaw/2) and
        w = cos(yaw/2), so the atan2 formula below reduces to
        atan2(sin(yaw), cos(yaw)) = yaw.

        The speed is the length of the velocity vector, so it stays right
        when the car slides sideways.

        Parameters
        ----------
        msg : nav_msgs.msg.Odometry
            The newest odometry message.

        Returns
        -------
        None
        """
        self.position = (msg.pose.pose.position.x, msg.pose.pose.position.y)
        q = msg.pose.pose.orientation             # heading as a quaternion -> angle
        self.yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        self.speed = math.hypot(msg.twist.twist.linear.x, msg.twist.twist.linear.y)
        self.odom_stamp = stamp_seconds(msg.header.stamp)

    def scan_callback(self, scan):
        """Answer one LiDAR scan with a drive command: the control loop.

        Runs once per LiDAR scan (one "tick"). The team measured about one
        scan per 0.01 s physics step, so about 100 per simulated second,
        although docs/06-rules.md rule 35 says "roughly 40 Hz"; the `gap`
        that measure_delay() logs shows the real rate.

        Fields read from the sensor_msgs/LaserScan message
            ranges           distance seen by each of the 819 beams [m]
            angle_min        angle of the first beam [rad], 0 = straight ahead
            angle_increment  angle between neighbouring beams [rad]
            header.stamp     simulation time of the scan, for measure_delay()
        Nothing else (angle_max, range_min, range_max, intensities) is used.

        Parameters
        ----------
        scan : sensor_msgs.msg.LaserScan
            The newest scan.

        Returns
        -------
        None
        """
        ranges = preprocess(np.asarray(scan.ranges, dtype=np.float64), self.params['max_range'])
        angles = scan.angle_min + np.arange(len(ranges)) * scan.angle_increment
        steering, speed = self.plan(ranges, angles)
        self.publish(steering, speed)
        self.measure_delay(stamp_seconds(scan.header.stamp))

        self.scan_count += 1
        if self.scan_count % 10 == 0:             # drawing every scan would be wasteful
            self.publish_markers(steering)

    def measure_delay(self, scan_stamp):
        """Log how late the driver reacts, in simulator steps (0.01 s each).

        The simulator stamps every scan and odometry message with its own
        clock, so when a command goes out:
            age  how far the simulation had got (newest odometry seen) past
                 the scan just answered. The command still has to travel to
                 the simulator, so the real delay is at least this.
            gap  steps between the scans the driver answered: 1 = it kept up
                 with every step; more = it skipped scans and each command
                 was held that many steps.
        training/evaluate.py --max-delay N tells you which delays the speed
        table survives: compare with the max age + gap reported here.

        Collects one age and one gap per scan and logs their mean and
        maximum every DELAY_REPORT_EVERY (200) scans. It only measures; it
        changes nothing about the driving.

        Parameters
        ----------
        scan_stamp : float
            Simulation time of the scan just answered [s].

        Returns
        -------
        None
        """
        if self.odom_stamp is not None:
            self.ages.append((self.odom_stamp - scan_stamp) / SIM_STEP)
        if self.last_scan_stamp is not None:
            self.gaps.append((scan_stamp - self.last_scan_stamp) / SIM_STEP)
        self.last_scan_stamp = scan_stamp
        if len(self.gaps) >= DELAY_REPORT_EVERY:
            ages, gaps = np.array(self.ages), np.array(self.gaps)
            self.get_logger().info(
                f'reaction delay over {len(gaps)} scans [steps]: age mean {ages.mean():.1f} '
                f'max {ages.max():.0f}; gap mean {gaps.mean():.1f} max {gaps.max():.0f}')
            self.ages, self.gaps = [], []

    def plan(self, ranges, angles):
        """Decide (steering [rad], speed [m/s]) from the latest scan and odometry.

        Two steps, explained in the comments below: (1) the lookahead
        policy picks how far ahead to aim, clipped to LOOKAHEAD_RANGE
        (0.35 to 4.0 m); (2) pure_pursuit() turns that into steering and
        speed.

        Before the first odometry message it returns (0, 0): stand still.
        After that it always uses the newest stored odometry; there is no
        check that it is recent. With the lookahead policy loaded the
        nearest path point is found twice, here and inside pure_pursuit()
        (same answer, a little wasted work).

        Parameters
        ----------
        ranges : (n_beams,) array
            Cleaned LiDAR ranges [m] (pure_pursuit.preprocess()); 819 beams.
        angles : (n_beams,) array
            Angle of each beam [rad], 0 = straight ahead, positive = left.

        Returns
        -------
        steering : float
            Steering angle [rad], within +-max_steer (0.4189), positive = left.
        speed : float
            Target speed [m/s], never negative.
        """
        if self.position is None:                 # no odometry yet: stay still
            return 0.0, 0.0
        car = np.array(self.position)

        # 1. How far ahead to aim. The lookahead policy, if loaded, decides
        #    from the speed and how sharply the line bends ahead. Otherwise
        #    (None) pure pursuit uses its own rule, lookahead_gain * speed.
        distance = None
        if self.lookahead_policy is not None:
            nearest = np.argmin(np.linalg.norm(self.path - car, axis=1))
            s = lookahead_inputs(self.curvature, nearest, self.speed)
            distance = float(np.clip(lookahead(self.lookahead_policy, s), *LOOKAHEAD_RANGE))

        # 2. Pure pursuit along the racing line. Speed from the speed table,
        #    or from pure pursuit's own limits if there is none.
        steering, speed, _, _ = pure_pursuit(
            self.path, car, self.yaw, self.speed, ranges, angles, self.params,
            lookahead=distance, speed_table=self.speeds)
        return float(steering), float(speed)

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    def publish(self, steering, speed):
        """Send one drive command on /drive.

        Fills three fields of an ackermann_msgs/AckermannDriveStamped
        message: header.stamp (the node's clock now: wall-clock time unless
        use_sim_time is set, so not comparable with the scan's simulation
        time), drive.steering_angle and drive.speed. header.frame_id is left
        empty, and acceleration, jerk and steering_angle_velocity are left
        at 0. The simulator reads only drive.speed and drive.steering_angle
        (clipping the steering to the car's limits) and keeps using the
        latest command until the next one arrives.

        Parameters
        ----------
        steering : float
            Steering angle [rad], positive = left.
        speed : float
            Target speed [m/s]; the simulator's own speed control tries to
            reach and hold it.

        Returns
        -------
        None
        """
        msg = AckermannDriveStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.drive.steering_angle = float(steering)
        msg.drive.speed = float(speed)
        self.drive_pub.publish(msg)

    def publish_markers(self, steering):
        """Draw both lines and an arrow showing the steering, for RViz.

        The arrow is 1.5 m long, drawn in the car's own frame
        (ego_racecar/base_link, at the rear axle) and turned by `steering`;
        the lines are in the map frame. A frame is the coordinate system a
        position is given in. Called on every 10th scan only.

        Parameters
        ----------
        steering : float
            The steering angle just sent [rad], positive = left.

        Returns
        -------
        None
        """
        arrow = Marker()
        arrow.header.frame_id = 'ego_racecar/base_link'           # drawn relative to the car
        arrow.header.stamp = self.get_clock().now().to_msg()
        arrow.ns = 'team_driver'
        arrow.id = 0
        arrow.type = Marker.ARROW
        arrow.action = Marker.ADD
        arrow.scale.x, arrow.scale.y, arrow.scale.z = 1.5, 0.15, 0.15
        arrow.color.g, arrow.color.b, arrow.color.a = 0.8, 1.0, 0.9
        arrow.pose.orientation.z = math.sin(steering / 2.0)        # rotate the arrow by `steering`
        arrow.pose.orientation.w = math.cos(steering / 2.0)
        self.marker_pub.publish(MarkerArray(markers=[arrow] + self.line_markers))


def stamp_seconds(stamp):
    """Convert a ROS time stamp to seconds.

    Parameters
    ----------
    stamp : builtin_interfaces.msg.Time
        A message header's stamp: whole seconds (`sec`) plus nanoseconds
        (`nanosec`).

    Returns
    -------
    float
        The same time in seconds [s].
    """
    return stamp.sec + stamp.nanosec * 1e-9


def line_marker(points, marker_id, colour):
    """Make an RViz marker that draws a closed loop of points as a thin line.

    colour = (r, g, b), each 0..1. The line is 0.05 m wide, in the map
    frame, and the first point is repeated at the end to close the loop.

    Parameters
    ----------
    points : (N, 2) array
        The loop's points [m], in order.
    marker_id : int
        Number that tells this marker apart from the others in the
        'team_driver' namespace (0 is the steering arrow).
    colour : tuple of 3 floats
        (red, green, blue), each 0..1.

    Returns
    -------
    visualization_msgs.msg.Marker
        A LINE_STRIP marker.
    """
    marker = Marker()
    marker.header.frame_id = 'map'
    marker.ns = 'team_driver'
    marker.id = marker_id
    marker.type = Marker.LINE_STRIP
    marker.action = Marker.ADD
    marker.scale.x = 0.05                                       # line width [m]
    marker.color.r, marker.color.g, marker.color.b = colour
    marker.color.a = 1.0
    marker.pose.orientation.w = 1.0
    for x, y in np.vstack([points, points[:1]]):               # repeat point 0 to close the loop
        marker.points.append(Point(x=float(x), y=float(y)))
    return marker


def main(args=None):
    """Run the driver node until it is stopped.

    This is what `ros2 run team_driver driver` calls: setup.py names it as
    the `driver` program. In order:

    1. rclpy.init() starts ROS for this process.
    2. Driver() does all the startup work (see Driver.__init__).
    3. rclpy.spin() waits for messages and runs the callbacks, one at a
       time, until the program is stopped.
    4. Ctrl+C (KeyboardInterrupt) is the normal way to stop, so it is not
       treated as an error. The node is then destroyed and ROS shut down,
       unless it has already been shut down (rclpy.ok() is False).

    Parameters
    ----------
    args : list of str, optional
        Command-line arguments for ROS, such as
        ['--ros-args', '-p', 'v_max:=6.0']. None means the program's own
        command line.

    Returns
    -------
    None
    """
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
