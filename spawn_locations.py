# SPAWNING 8 DIRECTION MARKERS

# importing libraries
import rclpy
from rclpy.node import Node
import random
import math
import time

from gazebo_msgs.srv import SpawnEntity, DeleteEntity
from geometry_msgs.msg import Pose

DIRECTIONS = {
    'N': (0.0, 1.0),
    'S': (0.0, -1.0),
    'E': (1.0, 0.0),
    'W': (-1.0, 0.0),
    'NE': (1.0, 1.0),
    'NW': (-1.0, 1.0),
    'SE': (1.0, -1.0),
    'SW': (-1.0, -1.0),
}

def marker_sdf(name, color):
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
          <ambient>{color}</ambient>
          <diffuse>{color}</diffuse>
        </material>
      </visual>
    </link>
  </model>
</sdf>
"""

class DirectionMarkers(Node):

    def __init__(self):
        super().__init__('direction_markers')

        self.spawn_cli = self.create_client(SpawnEntity, '/spawn_entity')
        self.delete_cli = self.create_client(DeleteEntity, '/delete_entity')

        self.get_logger().info("Waiting for Gazebo services...")
        self.spawn_cli.wait_for_service()
        self.delete_cli.wait_for_service()

        self.marker_names = []

        self.spawn_all_markers()
        self.get_logger().info("Markers spawned. Alive for 60 seconds...")
        time.sleep(60)
        self.delete_all_markers()
        self.get_logger().info("Markers deleted.")


    def spawn_all_markers(self):
        for name, (dx, dy) in DIRECTIONS.items():

            dist = random.uniform(2.0, 10.0)
            norm = math.sqrt(dx*dx + dy*dy)

            x = dist * dx / norm
            y = dist * dy / norm

            pose = Pose()
            pose.position.x = x
            pose.position.y = y
            pose.position.z = 1.0

            req = SpawnEntity.Request()
            req.name = f"marker_{name}"
            req.xml = marker_sdf(req.name, "0.5 0.0 0.5 0.8")
            req.initial_pose = pose
            req.reference_frame = "world"

            self.spawn_cli.call_async(req)
            self.marker_names.append(req.name)

            self.get_logger().info(
                f"Spawned {req.name} at ({x:.2f}, {y:.2f})"
            )


    def delete_all_markers(self):
        for name in self.marker_names:
            req = DeleteEntity.Request()
            req.name = name
            self.delete_cli.call_async(req)


def main():
    rclpy.init()
    node = DirectionMarkers()
    rclpy.shutdown()


if __name__ == "__main__":
    main()