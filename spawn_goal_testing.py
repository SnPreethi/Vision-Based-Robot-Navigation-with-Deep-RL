# SPAWNING GOAL MARKER
# This is an external goal manager. It means that the node runs outside the RL environment. On episode reset -> restart the node or call the service again.

#importing libraries
import rclpy
from rclpy.node import Node
import random
import math

from gazebo_msgs.srv import SpawnEntity, DeleteEntity
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose


# the 8 key directions
DIRECTIONS = {
    "N":  (0.0,  1.0),
    "S":  (0.0, -1.0),
    "E":  (1.0,  0.0),
    "W":  (-1.0, 0.0),
    "NE": (1.0,  1.0),
    "NW": (-1.0, 1.0),
    "SE": (1.0, -1.0),
    "SW": (-1.0, -1.0),
}

# goal marker visual
def goal_marker_sdf(name):
    return f"""
<sdf version="1.6">
  <model name="{name}">
    <static>true</static>
    <link name="link">
      <visual name="visual">
        <geometry>
          <box>
            <size>0.5 0.5 0.1</size>
          </box>
        </geometry>
        <material>
          <ambient>0.5 0.0 0.5 0.8</ambient>
          <diffuse>0.5 0.0 0.5 0.8</diffuse>
        </material>
      </visual>
    </link>
  </model>
</sdf>
"""


class RandomDirectionGoal(Node):

    def __init__(self):
        super().__init__('random_direction_goal')

        # this is used to check if distance is less than the tolerance value
        self.goal_tolerance = 0.5
        self.goal_active = False

        self.spawn_cli = self.create_client(SpawnEntity, '/spawn_entity')
        self.delete_cli = self.create_client(DeleteEntity, '/delete_entity')

        self.get_logger().info("Waiting for Gazebo services...")
        self.spawn_cli.wait_for_service()
        self.delete_cli.wait_for_service()

        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )

        self.robot_pose = None
        self.goal_pose = None
        self.goal_name = None

        self.spawn_random_goal()

    def spawn_random_goal(self):
        direction, (dx, dy) = random.choice(list(DIRECTIONS.items()))

        distance = random.uniform(2.0, 10.0)
        norm = math.sqrt(dx*dx + dy*dy)

        gx = distance * dx / norm
        gy = distance * dy / norm

        pose = Pose()
        pose.position.x = gx
        pose.position.y = gy
        pose.position.z = 1.0

        self.goal_pose = pose.position
        self.goal_name = f"goal_{direction}"

        req = SpawnEntity.Request()
        req.name = self.goal_name
        req.xml = goal_marker_sdf(self.goal_name)
        req.initial_pose = pose
        req.reference_frame = "world"

        self.spawn_cli.call_async(req)
        self.goal_active = True

        self.get_logger().info(
            f"Goal spawned at {direction} "
            f"({gx:.2f}, {gy:.2f})\n"
        )


    def odom_callback(self, msg):
        if not self.goal_active:
            return

        self.robot_pose = msg.pose.pose.position

        dx = self.robot_pose.x - self.goal_pose.x
        dy = self.robot_pose.y - self.goal_pose.y
        dist = math.sqrt(dx*dx + dy*dy)

        if dist < self.goal_tolerance:
            self.get_logger().info("Goal reached!")
            self.delete_goal()


    def delete_goal(self):
        req = DeleteEntity.Request()
        req.name = self.goal_name
        self.delete_cli.call_async(req)

        self.goal_active = False
        self.get_logger().info("Goal marker deleted")


def main():
    rclpy.init()
    node = RandomDirectionGoal()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()


'''
HIGH-LEVEL FLOW

start episode
↓
randomly choose direction
↓
spawn ONE goal marker
↓
continuously monitor robot pose
↓
if distance < tolerance:
    delete marker
    signal "goal reached"


ASSUMPTIONS
1. Goal is reached when distance(robot, goal) < goal_tolerance
2. world frame = world
3. marker is static visual-only
4. reset/spawn a new goal per episode
'''