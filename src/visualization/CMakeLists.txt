"""
This CMakeLists.txt file is responsible for configuring the build process of the ROS 1 package named "drone_viz". 
It specifies the package's dependencies, defines how to build and install the package, and ensures that the necessary assets  are properly installed. Additionally, it registers Python scripts as executable nodes that can be easily run using ROS commands.

- CMake manages the build process. In ROS, used to define how packages are built and how they depend on each other
- ROS1 is a framework for writing robot software
- catkin is the build system used in ROS1, which is built on top of CMake and provides more functionality to ROS packages
"""
cmake_minimum_required(VERSION 3.0.2)
project(drone_viz)

# Find catkin macros and libraries. Note: catkin_package() is required before any other catkin_* commands
find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  joint_state_publisher
  mavros
  mavros_msgs
  nav_msgs
  robot_state_publishers
  roscpp
  rospy
  rviz
  sensor_msgs
  tf
  tf2_ros
  urdf
  xacro
)

# Declare the catkin package and its dependencies
catkin_package(
  CATKIN_DEPENDS
    geometry_msgs
    joint_state_publisher
    mavros
    mavros_msgs
    nav_msgs
    robot_state_publisher
    roscpp
    rospy
    rviz
    sensor_msgs
    tf
    tf2_ros
    urdf
    xacro
)

# Install Python scripts as executable nodes
catkin_install_python(PROGRAMS
  scripts/mavros_tf_bridge.py
  scripts/path_builder.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

# Install launch files, URDF models, RViz configs, and other config files to the package share dirr
install(DIRECTORY launch urdf rviz config
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
