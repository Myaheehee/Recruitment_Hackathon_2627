#!/usr/bin/env python3
"""The team's ROS 2 node: answers each LiDAR scan with a drive command (pure pursuit).

The maths lives in ROS-free modules so training/ runs the same code. Topics: /scan and
/ego_racecar/odom in; /drive and /driver/markers (RViz) out.

    at startup    build the racing line and its speed table     raceline.py
                  load the lookahead policy, if trained         policies.py
    on odometry   remember where the car is and how fast        odom_callback()
    on each scan  the lookahead policy picks how far to aim     policies.py
                  pure pursuit picks steering and speed         pure_pursuit.py
                  send the command on /drive                    scan_callback()
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
from team_driver.units import Array, Metres, MetresPerSecond, Radians, Seconds

SIM_STEP = 0.01           # [s] one physics step of the race simulator
DELAY_REPORT_EVERY = 200  # scans between reaction-delay reports in the log


class Driver(Node):
    """The "driver" node. All slow work happens in __init__, before the car moves."""

    def __init__(self) -> None:
        """Build everything once, then open the topics (callbacks only start in rclpy.spin())."""
        super().__init__('driver')
        # In this order: each step uses what the one before made.
        self.params: dict = self.setup_parameters()            # 1. pure pursuit settings
        self.path: Array = self.build_path()                   # 2. [m] (N, 2) line to follow
        _, self.curvature = path_geometry(self.path)           # 3. [1/m] per point, + = left
        self.speeds: Array | None = self.build_speed_table()   # 4. [m/s] per point, or None
        self.load_policy()                                     # 5. sets self.lookahead_policy

        # 6. The car's latest state, from odometry.
        self.position: tuple[Metres, Metres] | None = None    # rear axle; None until odometry
        self.yaw: Radians = 0.0
        self.speed: MetresPerSecond = 0.0

        # Reaction-delay measurement (see measure_delay()).
        self.odom_stamp: Seconds | None = None    # simulation time of the newest odometry
        self.last_scan_stamp: Seconds | None = None
        self.ages, self.gaps = [], []             # [sim steps]

        # 7. Topics, last.
        self.drive_pub = self.create_publisher(AckermannDriveStamped, self.param('drive_topic'), 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/driver/markers', 1)
        # Queue depth 1: keep only the newest message; steering from an old scan is reaction delay.
        self.create_subscription(LaserScan, self.param('scan_topic'), self.scan_callback, 1)
        self.create_subscription(Odometry, self.param('odom_topic'), self.odom_callback, 1)
        self.scan_count = 0
        self.get_logger().info('driver ready')

    def param(self, name: str) -> bool | int | float | str:
        """Return the value of a ROS parameter declared in setup_parameters()."""
        return self.get_parameter(name).value

    # ---- Startup ----
    def setup_parameters(self) -> dict:
        """Declare every ROS parameter; return the pure pursuit ones (the keys of DEFAULTS).

        A default fixes the type: write `-p v_max:=7.0`, not `7`, or startup fails.
        Parameters are only read at startup, so `ros2 param set` on a running node does nothing.
        """
        for name, value in DEFAULTS.items():
            self.declare_parameter(name, value)
        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('odom_topic', '/ego_racecar/odom')
        self.declare_parameter('drive_topic', '/drive')
        self.declare_parameter('centerline_csv', '/hackathon/maps/icra26_centerline.csv')
        self.declare_parameter('map_yaml', '/hackathon/maps/icra26.yaml')
        self.declare_parameter('wall_gap', WALL_GAP)       # [m] air between the car's side and wall
        self.declare_parameter('raceline_iters', ITERS)    # more = smoother line, slower startup
        self.declare_parameter('raceline_csv', '/tmp/raceline.csv')   # copy of the line, for plots
        self.declare_parameter('use_raceline', True)       # False = follow the plain centreline
        self.declare_parameter('use_lookahead', True)      # False = ignore models/lookahead.npz
        self.declare_parameter('lookahead_weights', '')    # blank = default models/lookahead.npz
        self.declare_parameter('speed_table_weights', '')  # blank = default models/speed_table.npz
        return {name: self.param(name) for name in DEFAULTS}

    def build_path(self) -> Array:
        """Build the racing line from the map; return the line to follow, (N, 2) [m].

        Side effects: saves a copy to raceline_csv, and sets self.line_markers (which always
        show the racing line, even when use_raceline is false).
        """
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
        self.line_markers: list[Marker] = [line_marker(centreline, 1, (1.0, 0.8, 0.0)),
                                           line_marker(raceline, 2, (1.0, 0.0, 0.0))]

        return raceline if self.param('use_raceline') else centreline

    def build_speed_table(self) -> Array | None:
        """Target speed [m/s] at each point of self.path, or None to use pure pursuit's limits.

        The tuned models/speed_table.npz is used only if it was tuned on this same line;
        otherwise the table is worked out from mu and a_brake. Either is scaled by table_scale.
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
        # Scaling every speed down never breaks the neighbouring-points braking rule.
        speeds = speeds * self.params['table_scale']
        self.get_logger().info(
            f'speed table {source}, scaled by {self.params["table_scale"]}: '
            f'{speeds.min():.2f} to {speeds.max():.2f} m/s, '
            f'{lap_time(self.path, speeds):.2f} s per lap if driven exactly')
        return speeds

    def load_policy(self) -> None:
        """Set self.lookahead_policy from models/lookahead.npz, or None if there is none."""
        self.lookahead_policy: dict | None = None    # {'w': (5,) array, 'b': float}

        path = self.policy_file('lookahead')
        if path:
            self.lookahead_policy, trained_with = load_lookahead(path)
            w, b = self.lookahead_policy['w'], self.lookahead_policy['b']
            equation = ' '.join(f'{x:+.3f}*{name}' for x, name in zip(w, LOOKAHEAD_INPUTS))
            self.get_logger().info(f'lookahead policy loaded from {path}: L = {b:.3f} {equation}')
            self.warn_if_retuned(trained_with)
            return

        self.get_logger().info('no trained lookahead policy: lookahead_gain * speed')

    def policy_file(self, kind: str) -> str | None:
        """Path to the 'lookahead' or 'speed_table' .npz, or None if switched off or missing.

        The default is the installed share/ copy, not the source folder: copy a newly trained
        file into race_ws/src/team_driver/models/ and run `colcon build`.
        """
        if not self.param(f'use_{kind}'):
            return None
        path = self.param(f'{kind}_weights') or os.path.join(
            get_package_share_directory('team_driver'), 'models', f'{kind}.npz')
        return path if os.path.isfile(path) else None

    def warn_if_retuned(self, trained_with: dict) -> None:
        """Warn if pure pursuit's settings differ from those a policy was trained with.

        Only a warning: the policy is still used. The shipped .npz files predate table_scale
        and speed_lead, so every start lists those two (harmless at their defaults 1.0 and 0.0).
        """
        changed = [k for k in self.params if trained_with.get(k) != self.params[k]]
        if changed:
            self.get_logger().warn(f'pure pursuit settings changed since the policy was '
                                   f'trained: {changed}. Retrain it (training/README.md).')

    # ---- Every message ----
    def odom_callback(self, msg: Odometry) -> None:
        """Store the car's position (rear axle), heading and speed for the next scan."""
        self.position = (msg.pose.pose.position.x, msg.pose.pose.position.y)
        q = msg.pose.pose.orientation             # heading as a quaternion -> angle
        self.yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        self.speed = math.hypot(msg.twist.twist.linear.x, msg.twist.twist.linear.y)
        self.odom_stamp = stamp_seconds(msg.header.stamp)

    def scan_callback(self, scan: LaserScan) -> None:
        """The control loop: answer one LiDAR scan with a drive command."""
        ranges = preprocess(np.asarray(scan.ranges, dtype=np.float64), self.params['max_range'])
        angles = scan.angle_min + np.arange(len(ranges)) * scan.angle_increment
        steering, speed = self.plan(ranges, angles)
        self.publish(steering, speed)
        self.measure_delay(stamp_seconds(scan.header.stamp))

        self.scan_count += 1
        if self.scan_count % 10 == 0:             # drawing every scan would be wasteful
            self.publish_markers(steering)

    def measure_delay(self, scan_stamp: Seconds) -> None:
        """Log the reaction delay in sim steps (SIM_STEP), every DELAY_REPORT_EVERY scans.

        age: how far the simulation had got (newest odometry) past the scan just answered.
        gap: steps between answered scans; 1 = kept up, more = scans skipped, command held.
        Compare max age + gap with the delays `training/evaluate.py --max-delay N` survives.
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

    def plan(self, ranges: Array, angles: Array) -> tuple[Radians, MetresPerSecond]:
        """Decide (steering, speed) from the cleaned scan and the latest odometry.

        Uses the newest odometry however old it is: there is no freshness check.
        """
        if self.position is None:                 # no odometry yet: stay still
            return 0.0, 0.0
        car = np.array(self.position)

        # 1. How far ahead to aim: the lookahead policy's choice, or None = lookahead_gain * speed.
        distance = None
        if self.lookahead_policy is not None:
            nearest = np.argmin(np.linalg.norm(self.path - car, axis=1))
            s = lookahead_inputs(self.curvature, nearest, self.speed)
            distance = float(np.clip(lookahead(self.lookahead_policy, s), *LOOKAHEAD_RANGE))

        # 2. Pure pursuit along the line; speed from the speed table, or its own limits if None.
        steering, speed, _, _ = pure_pursuit(
            self.path, car, self.yaw, self.speed, ranges, angles, self.params,
            lookahead=distance, speed_table=self.speeds)
        return float(steering), float(speed)

    # ---- Output ----
    def publish(self, steering: Radians, speed: MetresPerSecond) -> None:
        """Send one drive command on /drive; the simulator holds it until the next one.

        The stamp is the node's clock (wall time unless use_sim_time), not simulation time.
        """
        msg = AckermannDriveStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.drive.steering_angle = float(steering)
        msg.drive.speed = float(speed)
        self.drive_pub.publish(msg)

    def publish_markers(self, steering: Radians) -> None:
        """Draw both lines and a 1.5 m arrow showing the steering, for RViz."""
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


def stamp_seconds(stamp) -> Seconds:
    """Convert a ROS time stamp (sec + nanosec) to seconds."""
    return stamp.sec + stamp.nanosec * 1e-9


def line_marker(points: Array, marker_id: int, colour: tuple[float, float, float]) -> Marker:
    """An RViz marker drawing `points` (N, 2) [m] as a closed loop; colour is (r, g, b), 0..1.

    marker_id must be unique in the 'team_driver' namespace (0 is the steering arrow).
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


def main(args: list[str] | None = None) -> None:
    """Run the driver node until it is stopped; `ros2 run team_driver driver` calls this."""
    rclpy.init(args=args)
    node = Driver()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:                     # Ctrl+C is the normal way to stop
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
