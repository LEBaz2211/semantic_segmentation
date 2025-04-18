#!/bin/bash
set -e

# Setup ROS environment
source "/opt/ros/$ROS_DISTRO/setup.bash"
source "/ros2_ws/install/setup.bash"

# Execute the command passed to docker run
exec "$@"