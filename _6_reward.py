# REWARD FUNCTION

'''
The world is empty with a single goal marker visible.
Vision-basde navigation.
The robot must learn TWO distinct behaviors:
    1. SEARCH: When goal not visible → rotate to find it
    2. APPROACH: When goal visible → align and drive toward it
Reward Components (in order of importance):
- Distance Progress: Primary learning signal
- Goal Visibility: Vision utilization
- Goal Alignment: Precise tracking
- Forward Velocity: Active approach
- Exploration: Search behavior
- Stuck Prevention: Movement requirement
- Time Efficiency: Episode brevity
- Terminal Success: Sparse goal reward
'''

# importing libraries
import numpy as np

class RewardFunction:
    def __init__(
            self,
            goal_threshold=1.0,      # success distance
            max_stuck_steps=20,      # stuck detection window
            visibility_history=5):   # frames to track visibility

        # goal params
        self.goal_threshold = goal_threshold
        
        # movement tracking
        self.max_stuck_steps = max_stuck_steps
        self.position_history = []

        # progress tracking
        self.prev_dist = None
        self.best_dist = None # track closest approach

        # vision tracking
        self.visibility_history = []
        self.max_visibility_history = visibility_history

        # episode statistics
        self.cumulative_reward = 0.0
        self.steps_with_goal_visible = 0
    
    # Utility functions
    @staticmethod
    def euclidean_distance(p1, p2):
        return np.linalg.norm(np.array(p1) - np.array(p2))
    
    # main reward function
    def compute(
            self,
            robot_xy,
            goal_xy,
            action,
            goal_visible=False,
            goal_x_offset=0.0,
            pixel_change=0.0,
            timeout=False,
            step_count=0):
        
        '''       
        :: Description
        :param robot_xy: float values tuple from ground-truth odometry
        :param goal_xy: float values tuple
        :param action: float valued tuple for continuous action
        :param goal_visible: bool if goal is visible or not
        :param goal_x_offset: Normalized horizontal offset of goal in image [-1, 1], 0 = centered
        :param pixel_change: Normalized pixel difference between frames [0, 1]
        :param timeout: Bool episode timeout flag

        Returns
        reward, info
        '''
        reward = 0.0
        info = {}
        v, omega = action

        # distance computation
        curr_dist = self.euclidean_distance(robot_xy, goal_xy)
        
        # initialization on first step
        if self.prev_dist is None:
            self.prev_dist = curr_dist
            self.best_dist = curr_dist

        delta_dist = self.prev_dist - curr_dist # positive means moving closer

        # track best approach
        if curr_dist < self.best_dist:
            self.best_dist = curr_dist

        # --- GOAL RELATED CONDITIONS --- #

        # TERMINAL reward
        """Large reward for reaching goal. This is the ultimate objective but too sparse for early learning."""
        if curr_dist < self.goal_threshold:
            reward += 200.0
            info['goal_reached'] = True
            info['success'] = True
            # bonus for efficiency (reaching faster)
            time_bonus = max(0, 50.0 * (1.0 - step_count / 1000.0))
            reward += time_bonus

            self.prev_dist = curr_dist
            return reward, info
        else:
            info['goal_reached'] = False
            info['success'] = False
        
        # --- Distance progress --- #
        progress_reward = 10.0 * delta_dist
        reward += progress_reward
        # Penalty for moving away at ANY distance when goal is visible
        if delta_dist < 0:  # Moving away
            base_penalty = 3.0 * abs(delta_dist)
            # Double penalty when goal is visible
            if goal_visible:
                base_penalty *= 2.0
            # Triple penalty when close
            if curr_dist < 5.0:  # Raised from 3.0
                base_penalty *= 1.5
            reward -= base_penalty
        info['progress_reward'] = progress_reward
        info['delta_distance'] = delta_dist

        # --- Goal visibility --- #
        """Encourage keeping goal in view. Vision is the primary sensor - robot must learn to use it."""
        if goal_visible:
            reward += 0.2 # constant visibility bonus
            self.steps_with_goal_visible += 1
            # Bonus for maintaining visibility
            if self.visibility_history and self.visibility_history[-1]:
                reward += 0.2  # Continuity bonus
        else:
            # Small penalty for losing sight of goal when close
            if curr_dist < 4.0:
                reward -= 0.3
        # Update visibility history
        self.visibility_history.append(goal_visible)
        if len(self.visibility_history) > self.max_visibility_history:
            self.visibility_history.pop(0)
        
        info['goal_visible'] = goal_visible

        # --- ALIGNMENT Reward --- #
        """When goal is visible, encourage centering it in frame. This teaches precise tracking before approach. goal_x_offset: -1 (left), 0 (center), +1 (right), alignment_score: 0 (edge) → 1 (center)"""
        if goal_visible:
            alignment_score = 1.0 - abs(goal_x_offset)  # 1.0 when centered
            alignment_reward = 0.3 * alignment_score
            reward += alignment_reward
            
            # Bonus for very good alignment
            if abs(goal_x_offset) < 0.2:  # Within 20% of center
                reward += 0.5
            
            info['alignment_score'] = alignment_score
        else:
            info['alignment_score'] = 0.0

        # --- Forward Velocity --- #
        """When goal is visible AND aligned, encourage forward movement. Prevents robot from just tracking without approaching."""
        velocity_reward = 0.0
        if goal_visible:
            if abs(goal_x_offset) < 0.3:  # Well-aligned
                if v > 0:  # Moving forward
                    velocity_reward = 5.0 * max(0, v)
                    reward += velocity_reward
                    # Extra bonus for good speed
                    if v > 0.4:
                        reward += 1.0
                    info['velocity_reward'] = velocity_reward
                else:  # Moving backward when aligned with visible goal
                    backward_penalty = 5.0 * abs(v)  # Matches forward reward
                    reward -= backward_penalty
                    velocity_reward = -backward_penalty
                    # Extra penalty when close
                    if curr_dist < 5.0:
                        reward -= 3.0
            else:  # Goal visible but not well-aligned
                # Still penalize backward motion, just less
                if v < 0:
                    penalty = 2.0 * abs(v)
                    reward -= penalty
                    velocity_reward -= penalty
        info['velocity_reward'] = velocity_reward

        # --- GENERAL BACKWARD MOTION PENALTY --- #
        """Explicit penalty for backward motion. Backward should only be used as last resort (e.g., stuck)"""
        if v < -0.1:  # Significant backward motion
            general_backward_penalty = 1.5 * abs(v)
            reward -= general_backward_penalty
            # Extra penalty if goal is visible (should be moving forward!)
            if goal_visible:
                reward -= 2.0
            info['backward_penalty'] = general_backward_penalty
        else:
            info['backward_penalty'] = 0.0
        
        # --- Exploration --- #
        """When goal not visible, encourage exploratory behavior. Pixel change indicates camera movement (rotation/translation)."""
        # computing movement magnitude
        movement_magnitude = abs(v) + abs(omega)
        if not goal_visible:
            exploration_reward = 0.0
            # rewarding exploration only if rover moves meaningfully
            if movement_magnitude > 0.15:
                # rewarding rotation slightly for searching behaviour
                if abs(omega) > 0.2:
                    exploration_reward += 0.2 * abs(omega)
                # rewarding new visual change (seeing new area)
                if pixel_change > 0.02:
                    exploration_reward += 0.2 * pixel_change
                reward += exploration_reward
            # discourage drifting backwards while searching
            if v < -0.1:
                reward -= 1.0 * abs(v) 
            info['exploration_reward'] = exploration_reward
        else:
            info['exploration_reward'] = 0.0
        
        # --- Stuck penalty --- #
        """Prevent robot from getting stuck in one place. Track position history and penalize lack of movement."""
        self.position_history.append(robot_xy)
        if len(self.position_history) > self.max_stuck_steps:
            self.position_history.pop(0)
        
        # Check if stuck (moved less than 0.1m in last N steps)
        if len(self.position_history) >= self.max_stuck_steps:
            positions = np.array(self.position_history)
            movement_range = np.ptp(positions, axis=0)  # Range in x and y
            total_movement = np.sum(movement_range)
            
            if total_movement < 0.1:  # Less than 10cm movement
                reward -= 2.0  # Stuck penalty
                info['stuck'] = True
            else:
                info['stuck'] = False
        else:
            info['stuck'] = False

        # --- Action Smoothness ---
        """Small penalty for erratic control. Helps with sim-to-real transfer and efficiency."""
        # useless spin penalty
        if abs(omega) > 0.8 and abs(delta_dist) < 0.01 and not goal_visible:
            reward -= 0.5
        
        # --- Time penalty --- #
        """Encourage efficiency - solve task in fewer steps. Must be small to not overwhelm progress rewards. Small increasing penalty for time efficiency"""
        #reward -= (step_count / 256) #0.01
        reward -= 0.005

        # --- Timeout Penalty --- #
        """Strong penalty for exceeding episode limit."""
        if timeout:
            reward -= 10.0
            info['timeout'] = True
        else:
            info['timeout'] = False

        # --- Diagnostics --- #
        info.update({
            'goal_distance': curr_dist,
            'best_distance': self.best_dist,
            'distance_from_best': curr_dist - self.best_dist,
            'near_goal': curr_dist < self.goal_threshold * 2,  # Within 2m
            'visibility_ratio': self.steps_with_goal_visible / max(1, step_count),
            'linear_velocity': v,
            'angular_velocity': omega,
            'total_reward_this_step': reward,
        })

        # Updating states
        self.prev_dist = curr_dist
        self.cumulative_reward += reward

        return reward, info
    
    # Reset all tracking vars for new episode
    def reset(self):
        self.prev_dist = None
        self.best_dist = None
        self.position_history = []
        self.visibility_history = []
        self.cumulative_reward = 0.0
        self.steps_with_goal_visible = 0