# This README exists to explain why the ROS 1 workspace was added inside the repo
# and to give the minimum build/run steps for the real-drone RViz pipeline.

This workspace contains the `drone_viz` ROS 1 package for Goal A:

- connect a real Pixhawk 4 through MAVROS
- publish a clean TF tree for RViz
- show the drone model moving live
- leave placeholders for future environmental sensors

Build from `catkin_ws` with:

```bash
```text
Implement this system for me end-to-end. Do not just explain it. Produce the actual files, package structure, launch files, URDF, TF setup, and ROS nodes needed for a working first version.

Project goal:
I have a real drone with a Pixhawk 4. I want a live RViz visualisation pipeline for the real vehicle. This is not a fake simulator. The drone already exists physically. I want RViz to reflect the motion of the real drone, and later I may want to visualise the surrounding environment too.

Available stack:
- Pixhawk 4
- MAVLink
- ROS 1 as the main target
- MAVProxy available if needed, but do not rely on it unless necessary
- ROS 2 may exist, but do not make ROS 2 the main implementation
- Use MAVROS for the ROS bridge unless there is a very strong reason not to

What you must build:
Create a ROS 1 package workspace design for a first working version that achieves Goal A first.

Goal A:
Show the real drone moving in RViz using live telemetry from the Pixhawk 4.

Goal B:
Prepare the architecture so that environmental visualisation can be added later with external sensors such as depth camera, stereo camera, VIO, or LiDAR.

Deliverables:
1. A full ROS 1 package structure
2. launch files
3. a URDF or Xacro for the drone model
4. robot_state_publisher setup
5. TF tree design
6. MAVROS launch/integration setup
7. any small helper ROS nodes needed
8. RViz config file
9. clear run commands in order
10. comments in every file explaining why it exists

Important:
Do not give me a generic explanation only. Actually generate the implementation artifacts in code blocks.

Required architecture:
- Pixhawk 4 sends MAVLink telemetry
- MAVROS receives MAVLink and exposes ROS topics
- ROS publishes the correct TF tree
- robot_state_publisher publishes the drone model
- RViz visualises TF, RobotModel, IMU, pose, odometry, path, and any other useful state

Assumptions:
- Assume a companion computer running Ubuntu with ROS 1 installed
- Assume MAVROS is installed
- Assume Pixhawk connects over either serial or UDP
- If needed, provide both serial and UDP launch examples
- Use sensible default topic names and frame names
- Keep the first version as simple and robust as possible

What I want you to output:
Produce all of the following.

A. Workspace / package layout
Show the exact folder tree, for example:
catkin_ws/
  src/
    drone_viz/
      package.xml
      CMakeLists.txt
      launch/
      urdf/
      rviz/
      scripts/
      config/

B. package.xml
Generate a valid ROS 1 package.xml with the dependencies needed for:
- roscpp
- rospy
- tf
- tf2_ros
- robot_state_publisher
- joint_state_publisher if needed
- urdf / xacro
- mavros
- mavros_msgs
- nav_msgs
- sensor_msgs
- geometry_msgs
- rviz

C. CMakeLists.txt
Generate a valid CMakeLists.txt for the package.

D. Drone URDF / Xacro
Create a simple but valid drone model with:
- base_link
- imu_link
- gps_link if useful
- camera_link placeholder
- lidar_link placeholder
Use simple primitive geometry. The goal is not visual beauty, only a correct RViz-visible robot model.

E. TF design
Define and explain the TF tree you are implementing. Use at least:
- map
- odom
- base_link
- imu_link
- camera_link
- lidar_link
If there is a better minimal design for a first version, implement it and explain briefly.

F. ROS nodes
Create any helper nodes needed, such as:
1. a node that subscribes to MAVROS pose/local position topics and republishes or broadcasts transforms if needed
2. a node that builds a Path message for RViz from incoming pose data
3. optional stub nodes or placeholders for future sensor integration

Use Python for helper nodes unless there is a strong reason not to.

G. MAVROS integration
Provide a ROS 1 launch setup that connects MAVROS to the Pixhawk.
Include:
- one serial example
- one UDP example
- any required parameters
- comments showing where I edit device path, baud rate, and endpoint

Do not assume MAVProxy is required unless absolutely necessary.

H. RViz config
Create a basic RViz config that adds:
- TF
- RobotModel
- IMU if appropriate
- Pose or Odometry
- Path
- Marker / Axes if useful
- PointCloud2 or LaserScan placeholders for future expansion

I. Launch files
Create launch files for:
1. mavros only
2. robot model + state publisher
3. helper nodes
4. full bringup including RViz

J. Run order
Give the exact command sequence to test the system, in order, for example:
- roscore
- launch MAVROS
- launch robot model
- launch helper nodes
- launch RViz
But adapt it to your actual implementation.

K. Topic map
List the expected important topics used in the implementation, including likely MAVROS topics such as:
- /mavros/state
- /mavros/imu/data
- /mavros/local_position/pose
- /mavros/local_position/odom
- /mavros/global_position/global
and any custom topics you create

L. Common issues
After generating the implementation, include a short practical troubleshooting section for:
- no MAVROS connection
- wrong frame orientation
- robot model not appearing
- no TF
- RViz fixed frame mismatch
- NED vs ENU confusion
- why Pixhawk telemetry alone does not create a 3D map

Implementation constraints:
- ROS 1 first
- Do not switch the whole design to ROS 2
- Keep the implementation minimal but real
- The code should be reasonably runnable, not pseudocode
- Use proper shebangs and ROS Python style
- Add comments in the code
- Keep names consistent across launch files, TF frames, and URDF

Nice to have:
- include a static transform publisher where appropriate
- include a placeholder config section for future camera/LiDAR addition
- mention where SLAM or VIO would later plug in for Goal B

Output format:
1. First show the full folder tree
2. Then provide every file one by one with filename headings
3. Then provide the exact run commands
4. Then provide a short explanation of how the data flows through the system
5. Then provide troubleshooting notes

Do not skip files. Do not just describe them. Write them.
```

source devel/setup.bash
```
