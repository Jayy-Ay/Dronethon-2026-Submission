#!/usr/bin/env python3
"""ROS node that bridges MAVROS odometry to TF for RViz visualization.
Allows RViz to visualize the drone's pos and orientation using the TF frames, while also providing a clean odometry topic for aother nodes to use. Also includes checks for valid quaternions to avoid broadcasting invalid TF transforms. Used in the `drone_viz` package in `bringup.launch`

1. Subscribe to MAVROS odometry topic (default: /mavros/local_position/odom)
2. Republish odometry with consistent frame names (default: odom, base_link)
3. Broadcast TF transform from odom to `base_link` based on the odom data
4. Optionally publishes the normalized odom to a new topic for other nodes to use (default: /drone_viz/odom)

Parameters:
- source_topic: The MAVROS odom topic to subscribe to (default: /mavros/local_position/odom)
- odom_frame: The name of the odometry frame to use (default: odom)
- base_frame: The name of the base frame to use (default: base_link)
- publish_odom_topic: The topic name to publish the normalized odom (default: /drone_viz/odom). If empty, not published

Terminology:
- Odometry: A message type that contains info about the pos, orientation, and vel of drone, by MAVROS
- TF: Keep track of multiple coords frames over time. Transforms points, vectors, etc. between two coord frames at any time
- Quarternion: Representation to encode 3D orientations and rotations (x, y, z, w). Often to represent orientation in 3D space without suffering from gimbal lock issues in Euler angles
- base_link: Frame that represents the robot/drone itself. Here, it's frame that represents the drone's body in the TF tree
"""

import rospy                                   # ROS Python client library
import tf2_ros                                 # ROS TF library for broadcasting transforms
from geometry_msgs.msg import TransformStamped # Message type for transform between two frames
from nav_msgs.msg import Odometry              # Message type for odominformation (pos, orientation, vel)   

class MavrosTfBridge:
    """Normalize MAVROS odometry frames and rebroadcast them as TF."""

    def __init__(self):
        """Create ROS publishers, subscribers, and frame-name configuration."""
        self.source_topic = rospy.get_param("~source_topic", "/mavros/local_position/odom")
        self.odom_frame = rospy.get_param("~odom_frame", "odom")
        self.base_frame = rospy.get_param("~base_frame", "base_link")
        self.publish_odom_topic = rospy.get_param("~publish_odom_topic", "/drone_viz/odom")

        self.tf_broadcaster = tf2_ros.TransformBroadcaster()
        self.odom_pub = rospy.Publisher(self.publish_odom_topic, Odometry, queue_size=20)
        self.sub = rospy.Subscriber(self.source_topic, Odometry, self.odom_callback, queue_size=50)
        rospy.loginfo("mavros_tf_bridge listening on %s", self.source_topic)

    def odom_callback(self, msg):
        """Republish incoming odometry and broadcast the matching base_link transform."""
        normalized = Odometry()
        normalized.header = msg.header
        normalized.header.frame_id = self.odom_frame # The frame that represents world/ground (e.g. "odom")
        normalized.child_frame_id = self.base_frame  # The frame that represents drone itself (e.g. "base_link")
        normalized.pose = msg.pose
        normalized.twist = msg.twist
        self.odom_pub.publish(normalized)

        transform = TransformStamped()
        transform.header = normalized.header        # Use same header (timestamp and frame_id) as the normalized odom
        transform.child_frame_id = self.base_frame  # The frame of the drone itself (base_link)
        transform.transform.translation.x = normalized.pose.pose.position.x
        transform.transform.translation.y = normalized.pose.pose.position.y
        transform.transform.translation.z = normalized.pose.pose.position.z
        transform.transform.rotation = normalized.pose.pose.orientation

        if not self._has_valid_quaternion(transform): # Check for valid quaternion before broadcasting TF. Avoids RViz issue
            rospy.logwarn_throttle(
                5.0,
                "Skipping TF broadcast because MAVROS odometry contained an invalid quaternion.",
            )
            return
        self.tf_broadcaster.sendTransform(transform) # Broadcast the transform from odom to base_link based on the odom data

    @staticmethod
    def _has_valid_quaternion(transform):
        """Return True when the transform rotation contains a non-zero quaternion."""
        q = transform.transform.rotation
        norm_sq = q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w
        return norm_sq > 1e-12


if __name__ == "__main__":
    rospy.init_node("mavros_tf_bridge")
    MavrosTfBridge()
    rospy.spin() # Keepnode running/processing callbacks until shutdown
