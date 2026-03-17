# GYM-LIKE ENV WRAPPER

# importing libraries
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import time
import rclpy

from geometry_msgs.msg import Twist, Point
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty

from _2_preprocess import preprocess
from _3_frame_stack import FrameStack
from _6_reward import RewardFunction
from vision_goal_utility import detect_goal_marker

# utility function for explicit action scaling
def scale_action(action, linear_limit, angular_limit):
    # Maps [-1, 1] -> [-linear_limit, +linear_limit]. This will allow backward motion.
    v = (action[0]) * linear_limit
    omega = action[1] * angular_limit
    return np.array([v, omega], dtype=np.float32)

class RosVisionEnv(gym.Env):
    metadata = {'render_modes': []}

    def __init__(
            self,
            node,
            camera_node,
            frame_stack_k=4,
            max_steps=600,
            linear_limit=0.8,
            angular_limit=0.5,
            reward_cfg=None,
            bounds_x=100.0,        # Max |x| coordinate before out-of-bounds
            bounds_y=100.0,        # Max |y| coordinate before out-of-bounds
            bounds_enabled=True,
    ):
        super().__init__()

        # core references
        self.node = node
        self.camera = camera_node

        # Frame Stack
        self.stack = FrameStack(frame_stack_k)

        # Reward
        self.reward_fn = RewardFunction(**(reward_cfg or {}))

        # Episode config
        self.max_steps = max_steps
        self.step_count = 0
        self.robot_xy = None
        self.goal_xy = None
        self.prev_frame = None
        self.goal_reached = False

        # Safety bounds
        self.bounds_enabled = bounds_enabled
        self.bounds_x = bounds_x
        self.bounds_y = bounds_y
        self.out_of_bounds_count = 0

        # Normalized Action space for PPO: [-1, 1]
        self.action_space = spaces.Box(low = -1.0, high = 1.0, shape = (2,), dtype = np.float32,)

        # physical action limits (used only for scaling)
        self.linear_limit = linear_limit
        self.angular_limit = angular_limit

        # Observation Space
        self.observation_space = spaces.Dict({
            'image': spaces.Box(
                low=0.0, high=1.0,
                shape=(frame_stack_k * 3, 84, 84),
                dtype=np.float32,
            ),
            'goal_distance': spaces.Box(
                low=0.0, high=1.0,
                shape=(1,),
                dtype=np.float32,
            ),
        })

        # ROS publishers / subscribers
        self.cmd_pub = node.create_publisher(Twist, "/cmd_vel", 10)

        node.create_subscription(Odometry, '/odom_ground_truth_absolute', self.odom_cb, 10,)
        node.create_subscription(Point, '/goal_xy', self.goal_xy_cb, 10)

        self.reset_goal_client = node.create_client(Empty, "/reset_goal")

    # ROS CALLBACKS
    def odom_cb(self, msg):
        p = msg.pose.pose.position
        self.robot_xy = (p.x, p.y)
    
    def goal_xy_cb(self, msg):
        self.goal_xy = (msg.x, msg.y)
    
    # UTILITIES
    def normalize_goal_distance(self, dist, max_dist=15.0):
        return np.clip(dist / max_dist, 0.0, 1.0)

    def _stop_robot(self):
        self.cmd_pub.publish(Twist())

    def _pixel_change(self, frame):
        if self.prev_frame is None:
            return 0.0
        diff = np.mean(np.abs(frame - self.prev_frame))
        return float(np.clip(diff, 0.0, 1.0))
    
    def _check_bounds(self):
        if not self.bounds_enabled or self.robot_xy is None:
            return False, None        
        x, y = self.robot_xy
        # check XY bounds
        if abs(x) > self.bounds_x or abs(y) > self.bounds_y:
            return True, f'XY out of bounds: ({x:.2f}, {y:.2f})'  
        return False, None
    
    def _build_obs(self):
        # Build the observation dict from current frame stack and goal distance
        img_obs = self.stack.get()
        if self.robot_xy is not None and self.goal_xy is not None:
            goal_dist = np.linalg.norm(
                np.array(self.robot_xy) - np.array(self.goal_xy)
            )
        else:
            goal_dist = 0.0
        return {
            'image': img_obs,
            'goal_distance': np.array(
                [self.normalize_goal_distance(goal_dist)],
                dtype=np.float32
            ),
        }
    
    # GYM API: RESET
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.step_count = 0
        self.prev_frame = None
        self.goal_reached = False

        self.reward_fn.reset()
        self.stack.reset()
        self._stop_robot()

        # Reset goal
        if self.reset_goal_client.wait_for_service(timeout_sec=5.0):
            self.reset_goal_client.call_async(Empty.Request())
            self.node.get_logger().info('Reset request sent to GoalManager')
        else:
            self.node.get_logger().error('/reset_goal service not available! Is GoalManager running?')

        # Waiting for the robot to appear at origin
        self.node.get_logger().info('Waiting for robot to reach origin after world reset...')
        origin_start = time.time()
        while True:
            rclpy.spin_once(self.node, timeout_sec=0.05)
            if self.robot_xy is not None:
                dist_to_origin = np.linalg.norm(np.array(self.robot_xy))
                if dist_to_origin < 0.5:  # within 0.5m of origin
                    self.node.get_logger().info(f'Robot confirmed at origin: {self.robot_xy}, dist={dist_to_origin:.2f}m')
                    break
            if time.time() - origin_start > 15.0:
                self.node.get_logger().warn(
                    f'Origin confirmation timed out after 15s. '
                    f'Robot at {self.robot_xy}. '
                    f'Check that /reset_world is working correctly in Gazebo.'
                )
                break

        # Forcing fresh goal position receive
        self.goal_xy = None
        goal_start = time.time()
        self.node.get_logger().info('Waiting for goal position on /goal_xy ...')
        while self.goal_xy is None:
            rclpy.spin_once(self.node, timeout_sec=0.05)
            if time.time() - goal_start > 10.0:
                self.node.get_logger().warn('Goal position not received after 10s! Check GoalManager is running and publishing /goal_xy')
                # Failsafe: assign a default so the episode can still run
                self.goal_xy = (3.0, 3.0)
                self.node.get_logger().warn(f'Using failsafe goal_xy = {self.goal_xy}')
                break

        # Wait for camera to produce fresh frames after reset
        cam_start = time.time()
        frames_received = 0
        target_frames = 3 # waiting for 3 refresh frames before proceeding

        while frames_received < target_frames:
            prev_id = self.camera.frame_id
            rclpy.spin_once(self.node, timeout_sec=0.05)
            with self.camera.lock:
                curr_id = self.camera.frame_id
            if curr_id > prev_id:
                frames_received += 1
            if time.time() - cam_start > 8.0:
                self.node.get_logger().warn(f'Camera only recovered {frames_received}/{target_frames} frames after 8s. Proceeding anyway.')
                break
        self.node.get_logger().info(f'Camera ready: {frames_received} fresh frames received')

        # Warm up frame stack with fresh frames
        for _ in range(self.stack.k):
            with self.camera.lock:
                raw = self.camera.latest_frame
            if raw is not None:
                frame = preprocess(raw)
                self.stack.push(frame)
            rclpy.spin_once(self.node, timeout_sec=0.01)

        # ensuring frame stack if full
        if len(self.stack.frames) < self.stack.k:
            dummy = np.zeros((3, 84, 84), dtype=np.float32)
            while len(self.stack.frames) < self.stack.k:
                self.stack.push(dummy)

        # building initial observation
        obs = self._build_obs()
        self.prev_frame = self.stack.frames[-1]

        # log goal visibility at episode start
        goal_visible_at_start = False
        if self.camera.latest_frame is not None:
            goal_visible_at_start, _ = detect_goal_marker(self.camera.latest_frame)
        goal_dist = (
            np.linalg.norm(np.array(self.robot_xy) - np.array(self.goal_xy))
            if self.robot_xy is not None and self.goal_xy is not None
            else 0.0
        )

        self.node.get_logger().info(
            f'=== EPISODE START === '
            f'Robot at {self.robot_xy} | '
            f'Goal at {self.goal_xy} | '
            f'Dist = {goal_dist:.2f}m | '
            f'Goal visible = {goal_visible_at_start}'
        )

        return obs, {}

    # GYM API: STEP
    def step(self, action):
        info = {}
        self.step_count += 1

        # goal_xy must be valid before we can compute anything
        if self.goal_xy is None:
            self.node.get_logger().warn(f'Step {self.step_count}: goal_xy is none - Skipping step, returning zero-reward.')
            obs = self._build_obs()
            return obs, 0.0, False, False, {'no_goal': True}

        # capture current frame ID before taking action
        with self.camera.lock:
            initial_frame_id = self.camera.frame_id

        # scale action
        v, omega = scale_action(action, self.linear_limit, self.angular_limit)

        # publish action
        cmd = Twist()
        cmd.linear.x = float(v)
        cmd.angular.z = float(omega)
        self.cmd_pub.publish(cmd)

        # deterministic spinning to give ROS a moment to process the command
        rclpy.spin_once(self.node, timeout_sec=0.0)

        # wait for NEW frame that arrived after the action
        wait_count = 0
        max_wait = 30 # max ~300ms wait
        current_frame_id = initial_frame_id

        while current_frame_id == initial_frame_id and wait_count < max_wait:
            rclpy.spin_once(self.node, timeout_sec=0.01)
            with self.camera.lock:
                current_frame_id = self.camera.frame_id
                raw_frame = self.camera.latest_frame
            wait_count += 1

        # Handle frame timeout case (no new frame recieved)
        if current_frame_id == initial_frame_id:
            self.node.get_logger().warn(
                f'No new frame received after action.'
                f'Waited {wait_count * 10}ms. Using stale frame.'
                f'Step = {self.step_count}'
            )
            # if still no frame, Return current observation with small penalty
            obs = self._build_obs()
            return obs, -0.01, False, False, {'no_new_frame': True, 'frame_stale': True}
        
        # handle missing frame data
        if raw_frame is None:
            self.node.get_logger().error('Camera frame is None!')
            obs = self._build_obs()
            return obs, -0.01, False, False, {'camera_timeout': True}
        
        # preprocess the new frame
        frame = preprocess(raw_frame)
        self.stack.push(frame)
        
        # build observation
        obs = self._build_obs()

        # visual goal detection
        goal_visible, goal_x_offset = detect_goal_marker(raw_frame)

        # pixel change for exploration reward
        pixel_change = self._pixel_change(frame)
        self.prev_frame = frame

        # check safety bounds
        is_out_of_bounds, bounds_reason = self._check_bounds()
        if is_out_of_bounds:
            self.out_of_bounds_count += 1
            self.node.get_logger().error(
                f'OUT OF BOUNDS (#{self.out_of_bounds_count}): {bounds_reason} | '
                f'Episode {self.step_count} steps | Forcing episode end'
            )
            self._stop_robot()
            # Force episode termination with large penalty
            return obs, -100.0, False, True, {
                'out_of_bounds': True,
                'bounds_reason': bounds_reason,
                'goal_distance': np.linalg.norm(np.array(self.robot_xy) - np.array(self.goal_xy)),
            }
        
        # check timeout
        timeout = self.step_count >= self.max_steps

        goal_dist = (
            np.linalg.norm(np.array(self.robot_xy) - np.array(self.goal_xy))
            if self.robot_xy is not None
            else 0.0
        )

        # Compute reward
        reward, reward_info = self.reward_fn.compute(
            robot_xy = self.robot_xy or (0.0, 0.0),
            goal_xy = self.goal_xy,
            action = np.array([v, omega], dtype=np.float32),
            goal_visible = goal_visible,
            goal_x_offset = goal_x_offset,
            pixel_change = pixel_change,
            timeout=timeout,
            step_count=self.step_count
        )

        # determine episode end conditions
        if reward_info.get('goal_reached', False):
            self.goal_reached = True
        
        terminated = self.goal_reached
        truncated = timeout

        if terminated or truncated:
            self._stop_robot()
        
        # update info dict
        info.update(reward_info)
        info['frame_id'] = current_frame_id  # Track which frame we used
        info['frame_wait_ms'] = wait_count * 10  # Track how long we waited

        # Peiordic DEBUG logging
        if self.step_count % 20 == 0:
            self.node.get_logger().info(
                f"Step={self.step_count}, v={v:.2f}, omega={omega:.2f}, "
                f"goal_visible={goal_visible}, frame_wait={wait_count * 10}ms"
                f'pos=({self.robot_xy[0]:.1f},{self.robot_xy[1]:.1f}) '
            )

        return obs, reward, terminated, truncated, info

    def close(self):
        self._stop_robot()