# SPAWNING GOAL MARKER

#importing libraries
import rclpy
from rclpy.node import Node
import random
import math
import time
import subprocess

from gazebo_msgs.srv import SpawnEntity, DeleteEntity
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose, Point, Twist
from std_srvs.srv import Empty
from std_msgs.msg import Bool

# the 8 cardinal + diagonal directions
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

# Goal name
GOAL_NAME = 'rl_goal'

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
            <size>0.4 0.4 0.1</size>
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

class GoalManager(Node):
    def __init__(self):
        super().__init__('goal_manager')

        # Goal params
        self.goal_tolerance = 1.0
        self.min_dist = 2.0
        self.max_dist = 8.0
        self.goal_active = False
        self.goal_xy = None
        self.robot_xy = None
        self.reset_count = 0

        # Gazebo services
        self.spawn_cli  = self.create_client(SpawnEntity,  '/spawn_entity')
        self.delete_cli = self.create_client(DeleteEntity, '/delete_entity')

        # Publishers
        self.goal_pos_pub = self.create_publisher(Point, '/goal_xy', 10)
        self.goal_pub = self.create_publisher(Bool, '/goal_reached', 10)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # Wait for Gazebo spawn / delete services
        self.get_logger().info('Waiting for Gazebo spawn/delete services...')
        self.spawn_cli.wait_for_service()
        self.delete_cli.wait_for_service()
        self.get_logger().info('Gazebo services ready.')

        # Subscriptions
        self.create_subscription(Odometry, '/odom_ground_truth_absolute', self.odom_callback, 10)

        # Services exposed to training env
        self.create_service(Empty, '/reset_goal', self.reset_goal_cb)

        self.get_logger().info(
            f'Goal Manager ready. '
            f'Call /reset_goal to reset world and spawn new goal.'
        )

    # CALLBACKS
    def odom_callback(self, msg):      
        p = msg.pose.pose.position
        self.robot_xy = (p.x, p.y)

        if not self.goal_active or self.goal_xy is None:
            return

        dx = p.x - self.goal_xy[0]
        dy = p.y - self.goal_xy[1]
        if math.sqrt(dx*dx + dy*dy) < self.goal_tolerance:
            self.get_logger().info('Goal reached!!!')
            self._delete_goal()
            msg = Bool()
            msg.data = True
            self.goal_pub.publish(msg)
    
    def reset_goal_cb(self, request, response):
        self.reset_count += 1
        self.get_logger().info(f'=== Episode reset #{self.reset_count} ===')

        # stop the rover first
        self._stop_robot()

        # delete existing goal
        self._delete_goal()
        
        # world reset
        self._reset_world()
        
        # spawn new goal
        self.spawn_goal()
        
        # publish reset signal
        msg = Bool()
        msg.data = False
        self.goal_pub.publish(msg)
        
        return response

    def _reset_world(self, max_retries=3):
        """
        Calls Gazebo's /reset_world service via subprocess.
        This resets the entire simulation: robot returns to its URDF spawn
        position (origin), physics state is cleared. No teleport needed.
        """
        self.get_logger().info('Calling /reset_world ...')

        for attempt in range(max_retries):
            try:
                result = subprocess.run(
                    ["ros2", "service", "call", "/reset_world", "std_srvs/srv/Empty"],
                    capture_output=True,
                    text=True,
                    timeout=30.0
                )

                if result.returncode == 0 and "unhandled exception" not in result.stderr:
                    self.get_logger().info('World reset successful.')
                    # Give simulation time to settle after reset
                    time.sleep(1.5)
                    return True
                else:
                    self.get_logger().warn(
                        f'World reset attempt {attempt + 1} failed: {result.stderr}'
                    )

            except subprocess.TimeoutExpired:
                self.get_logger().warn(f'World reset attempt {attempt + 1} timed out (30s).')

            except Exception as e:
                self.get_logger().error(f'World reset attempt {attempt + 1} error: {e}')

            # Wait before retry
            if attempt < max_retries - 1:
                time.sleep(1.0)

        self.get_logger().error(
            'All world reset attempts failed! '
            'Robot may not be at origin. Check that /reset_world service is available.'
        )
        return False

        # ROBOT CONTROLS
    
    def _stop_robot(self):
        # Publish zero velocity to stop robot
        self.cmd_pub.publish(Twist())
    
    # GOAL MANAGEMENT
    def spawn_goal(self):
        # spawn the goal marker at random position
        if self.goal_active:
            self.get_logger().info('spawn_goal called while goal already active - ignoring')
            return
        
        direction, (dx, dy) = random.choice(list(DIRECTIONS.items()))
        distance = random.uniform(self.min_dist, self.max_dist)
        norm = math.sqrt(dx*dx + dy*dy)
        gx = distance * dx / norm
        gy = distance * dy / norm

        pose = Pose()
        pose.position.x = gx
        pose.position.y = gy
        pose.position.z = 0.2

        req = SpawnEntity.Request()
        req.name = GOAL_NAME
        req.xml = goal_marker_sdf(GOAL_NAME)
        req.initial_pose = pose
        req.reference_frame = "world"

        self.spawn_cli.call_async(req)
        
        self.goal_xy = (gx, gy)
        self.goal_active = True

        # publishing the goal multiple times to ensure the env receives it
        pt = Point(x=gx, y=gy, z=0.0)
        for _ in range(5):
            self.goal_pos_pub.publish(pt)
            time.sleep(0.05)

        self.get_logger().info(f"Goal spawned [{direction}] at ({gx:.2f}, {gy:.2f}) | dist = {distance:.2f}m")

    def _delete_goal(self):
        if not self.goal_active:
            return
        
        req = DeleteEntity.Request()
        req.name = GOAL_NAME
        self.delete_cli.call_async(req)

        self.goal_active = False
        self.goal_xy = None

        self.get_logger().info(f"Goal marker {GOAL_NAME} deleted.")

def main():
    rclpy.init()
    node = GoalManager()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()