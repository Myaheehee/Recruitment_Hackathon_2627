#!/usr/bin/env python3
"""The race driver: a ROS 2 node that follows a racing line with pure pursuit.

What happens, and where to read about it:
    at startup     build the racing line from the map             raceline.py
                   load the trained correction network, if any    residual.py
    on odometry    remember where the car is and how fast         odom_callback()
    on each scan   pure pursuit picks steering and speed          pure_pursuit.py
                   the network (if loaded) adds a small correction
                   send the command on /drive                     scan_callback()

Topics
    in    /scan               LiDAR, 819 beams over 270 degrees
          /ego_racecar/odom   true position and speed (allowed by the rules)
    out   /drive              steering angle + target speed; the simulator holds the speed
          /driver/markers     the lines and a steering arrow, for RViz

Run it:              ros2 run team_driver driver
Change a setting:    ros2 run team_driver driver --ros-args -p v_max:=6.0

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
from team_driver.raceline import CAR_HALF_WIDTH, ITERS, WALL_GAP, build_raceline, load_clearance_map
from team_driver.residual import actor, apply_residual, features, load_weights, path_geometry


class Driver(Node):

    def __init__(self):
        super().__init__('driver')
        self.params = self.setup_parameters()
        self.path = self.build_path()
        self.load_network()

        # The car's latest state, from odometry.
        self.position = None      # (x, y) of the rear axle; None until odometry arrives
        self.yaw = 0.0            # [rad] heading
        self.speed = 0.0          # [m/s]
        self.yaw_rate = 0.0       # [rad/s] how fast the car is turning
        self.last_steer = 0.0     # [rad] the steering we sent last time

        self.drive_pub = self.create_publisher(AckermannDriveStamped, self.param('drive_topic'), 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/driver/markers', 1)
        self.create_subscription(LaserScan, self.param('scan_topic'), self.scan_callback, 10)
        self.create_subscription(Odometry, self.param('odom_topic'), self.odom_callback, 10)
        self.scan_count = 0
        self.get_logger().info('driver ready')

    def param(self, name):
        return self.get_parameter(name).value

    # ------------------------------------------------------------------
    # Startup
    # ------------------------------------------------------------------
    def setup_parameters(self):
        """Declare every ROS parameter, and return the pure pursuit ones as a
        dict. Their defaults live in pure_pursuit.DEFAULTS because the
        training code needs the same numbers."""
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
        self.declare_parameter('use_residual', True)       # False = ignore the trained network
        self.declare_parameter('residual_weights', '')     # blank = models/residual.npz in this package
        return {name: self.param(name) for name in DEFAULTS}

    def build_path(self):
        """Build the racing line (raceline.py) and return the line to follow."""
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

    def load_network(self):
        """Load the trained correction network (see training/), if there is
        one. Without it the car drives on pure pursuit alone."""
        self.weights = None
        self.heading, self.curvature = path_geometry(self.path)
        if not self.param('use_residual'):
            return
        path = self.param('residual_weights') or os.path.join(
            get_package_share_directory('team_driver'), 'models', 'residual.npz')
        if not os.path.isfile(path):
            self.get_logger().info(f'no residual network at {path}: pure pursuit only')
            return

        self.weights, self.limits, trained_with = load_weights(path)
        self.get_logger().info(f'residual network loaded from {path}, limits {self.limits}')
        # The network learned to correct pure pursuit tuned the way it was
        # during training. If the tuning has changed since, retrain it.
        changed = [k for k, v in trained_with.items() if self.params.get(k) != v]
        if changed:
            self.get_logger().warn(f'pure pursuit settings changed since the network was '
                                   f'trained: {changed}. Retrain it (training/README.md).')

    # ------------------------------------------------------------------
    # Every message
    # ------------------------------------------------------------------
    def odom_callback(self, msg):
        self.position = (msg.pose.pose.position.x, msg.pose.pose.position.y)
        q = msg.pose.pose.orientation             # heading as a quaternion -> angle
        self.yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        self.speed = math.hypot(msg.twist.twist.linear.x, msg.twist.twist.linear.y)
        self.yaw_rate = msg.twist.twist.angular.z

    def scan_callback(self, scan):
        """The control loop: runs once per LiDAR scan."""
        ranges = preprocess(np.asarray(scan.ranges, dtype=np.float64), self.params['max_range'])
        angles = scan.angle_min + np.arange(len(ranges)) * scan.angle_increment
        steering, speed = self.plan(ranges, angles)
        self.publish(steering, speed)

        self.scan_count += 1
        if self.scan_count % 10 == 0:             # drawing every scan would be wasteful
            self.publish_markers(steering)

    def plan(self, ranges, angles):
        """Decide (steering [rad], speed [m/s]) from the latest scan and odometry."""
        if self.position is None:                 # no odometry yet: stay still
            return 0.0, 0.0
        car = np.array(self.position)

        # 1. Pure pursuit along the racing line, with the speed limits.
        steering, speed, nearest, ahead = pure_pursuit(
            self.path, car, self.yaw, self.speed, ranges, angles, self.params)

        # 2. The learned correction, if a network is loaded: describe the
        #    situation as numbers, ask the network, add its capped answer.
        if self.weights is not None:
            x = features(self.path, self.heading, self.curvature, car, self.yaw,
                         self.speed, self.yaw_rate, self.last_steer,
                         steering, speed, nearest, ahead, self.params)
            squashed = np.tanh(actor(self.weights, x))         # each output in -1..1
            steering, speed = apply_residual(steering, speed, squashed, self.limits, self.params)

        self.last_steer = float(steering)
        return float(steering), float(speed)

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    def publish(self, steering, speed):
        msg = AckermannDriveStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.drive.steering_angle = float(steering)
        msg.drive.speed = float(speed)
        self.drive_pub.publish(msg)

    def publish_markers(self, steering):
        """Draw both lines and an arrow showing the steering, for RViz."""
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


def line_marker(points, marker_id, colour):
    """A closed loop of points drawn as a thin line in RViz. colour = (r, g, b), each 0..1."""
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
