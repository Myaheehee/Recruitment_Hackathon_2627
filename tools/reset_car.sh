#!/usr/bin/env bash
# Stop the driver, stop the car, and put it back on the start line.
# Run inside the container (needs ROS sourced):  /hackathon/tools/reset_car.sh

# 1. Kill the driver node, or it keeps publishing and drives off again.
#    (The [d] trick stops pkill matching its own command line.)
pkill -f '[t]eam_driver driver|lib/team_driver/[d]river' && echo "driver stopped"

# 2. Ask for zero speed and straight wheels.
ros2 topic pub --once /drive ackermann_msgs/msg/AckermannDriveStamped \
  "{drive: {speed: 0.0, steering_angle: 0.0}}" > /dev/null

# 3. Teleport to the start pose from maps/tracks.yaml: [-1.68, -0.01, 0.0].
ros2 topic pub --once /initialpose geometry_msgs/msg/PoseWithCovarianceStamped \
  "{header: {frame_id: map}, pose: {pose: {position: {x: -1.68, y: -0.01}, orientation: {w: 1.0}}}}" > /dev/null

echo "car reset to start line"
