'''
TESTING TRAINED MODELS ON
1. Performance metrics (success rate, efficiency, reward)
2. Behavioral analysis (search, approach, alignment)
3. Visual analysis (trajectories, temporal plots, distributions)
4. Comparison (multiple checkpoints, baselines)

TESTING SUITE STRUCTURE
1. Quick Test (5 episodes) - Sanity check
2. Standard Test (50 episodes) - Main evaluation
3. Stress Test (100 episodes) - Statistical significance
4. Behavioral Analysis - Detailed single-episode study
5. Comparison Test - Multiple checkpoints

Usage:
    python test_ppo_model.py --checkpoint checkpoints/ppo_best.pt --episodes 50
    python test_ppo_model.py --compare checkpoints/ppo_ep_00100.pt checkpoints/ppo_ep_01000.pt
'''

# importing libraries
import rclpy
import torch
import numpy as np
import argparse
import os
import json
from datetime import datetime
from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import seaborn as sns
import json

from _1_camera_node import CameraNode
from _7_env_ros2 import RosVisionEnv
from _4A_policy_cnn_ppo import CNNPolicy

# setting styles for better plots
# Set style for better plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# CONFIG
class TestConfig:
    # CONFIG FOR TESTING
    
    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Evaluation settings
    DETERMINISTIC = True  # Use mean action (no sampling)
    RENDER_INTERVAL = 5   # Save trajectory plot every N episodes
    
    # Output directories
    OUTPUT_DIR = "test_results"
    PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
    TRAJECTORIES_DIR = os.path.join(OUTPUT_DIR, "trajectories")
    LOGS_DIR = os.path.join(OUTPUT_DIR, "logs")
    
    # Metrics to track
    METRICS = [
        'success_rate',
        'avg_episode_reward',
        'avg_episode_length',
        'avg_goal_distance',
        'avg_visibility_ratio',
        'avg_alignment_score',
        'avg_forward_velocity',
        'efficiency_ratio',
    ]

# EPISODE DATA COLLECTOR TO collect and store all data from a single episode
class EpisodeData:    
    def __init__(self):
        self.reset()
    
    def reset(self):
        # Trajectory
        self.positions = []  # [(x, y), ...]
        self.goal_position = None
        
        # Actions
        self.actions = []  # [(v, omega), ...]
        
        # States
        self.distances = []  # Distance to goal over time
        self.rewards = []
        self.goal_visible = []  # Boolean timeline
        self.goal_offsets = []  # Alignment over time
        
        # Episode outcome
        self.success = False
        self.episode_length = 0
        self.total_reward = 0.0
        self.termination_reason = "unknown"
        
    def add_step(self, robot_xy, goal_xy, action, reward, info):
        # Add data from one step
        self.positions.append(robot_xy)
        if self.goal_position is None:
            self.goal_position = goal_xy
        
        self.actions.append(action)
        self.rewards.append(reward)
        self.distances.append(info.get('goal_distance', 0.0))
        self.goal_visible.append(info.get('goal_visible', False))
        self.goal_offsets.append(info.get('alignment_score', 0.0))
        
        self.episode_length += 1
        self.total_reward += reward
    
    def finalize(self, success, reason):
        # Mark episode as complete
        self.success = success
        self.termination_reason = reason
    
    def get_summary(self):
        # Get summary statistics for this episode
        return {
            'success': self.success,
            'episode_length': self.episode_length,
            'total_reward': self.total_reward,
            'final_distance': self.distances[-1] if self.distances else 0.0,
            'min_distance': min(self.distances) if self.distances else 0.0,
            'visibility_ratio': np.mean(self.goal_visible) if self.goal_visible else 0.0,
            'avg_alignment': np.mean(self.goal_offsets) if self.goal_offsets else 0.0,
            'avg_forward_vel': np.mean([a[0] for a in self.actions]) if self.actions else 0.0,
            'termination_reason': self.termination_reason,
        }

# MODEL EVALUATOR
class ModelEvaluator:
    # Evaluates a trained model and collects metrics
    
    def __init__(self, env, policy, device, deterministic=True):
        self.env = env
        self.policy = policy
        self.device = device
        self.deterministic = deterministic
        
        # Storage
        self.episodes_data = []
        self.summary_stats = {}
        
    def evaluate(self, num_episodes, desc="Evaluating"):
        # Run evaluation for specified number of episodes
        print(f"\n{'='*60}")
        print(f"   EVALUATION: {num_episodes} episodes")
        print(f"   Deterministic: {self.deterministic}")
        print(f"   Device: {self.device}")
        print(f"{'='*60}\n")
        
        self.episodes_data = []
        
        for ep_idx in tqdm(range(num_episodes), desc=desc):
            episode_data = self._run_episode()
            self.episodes_data.append(episode_data)
            
            # Print summary every 10 episodes
            if (ep_idx + 1) % 10 == 0:
                self._print_progress(ep_idx + 1)
        
        # Compute summary statistics
        self._compute_summary_stats()
        
        print(f"\n Evaluation complete!")
        return self.summary_stats
    
    def _run_episode(self):
        # Run a single episode and collect data
        episode_data = EpisodeData()
        
        obs, _ = self.env.reset()
        done = False
        
        while not done:
            # Get action from policy
            with torch.no_grad():
                obs_t = {
                    "image": torch.from_numpy(obs["image"]).float().to(self.device).unsqueeze(0),
                    "goal_distance": torch.from_numpy(obs["goal_distance"]).float().to(self.device).unsqueeze(0),
                }
                
                mean, std, value = self.policy(obs_t)
                
                if self.deterministic:
                    # Use mean action (no exploration)
                    action = torch.tanh(mean)
                else:
                    # Sample from distribution
                    from torch.distributions import Normal
                    dist = Normal(mean, std)
                    raw_action = dist.sample()
                    action = torch.tanh(raw_action)
            
            action_np = action.squeeze(0).cpu().numpy()
            
            # Step environment
            next_obs, reward, terminated, truncated, info = self.env.step(action_np)
            done = terminated or truncated
            
            # Store data
            episode_data.add_step(
                robot_xy=self.env.robot_xy or (0.0, 0.0),
                goal_xy=self.env.goal_xy or (0.0, 0.0),
                action=action_np,
                reward=reward,
                info=info
            )
            
            obs = next_obs
        
        # Finalize episode
        reason = "success" if terminated else "timeout"
        episode_data.finalize(success=terminated, reason=reason)
        
        return episode_data
    
    def _compute_summary_stats(self):
        """Compute aggregate statistics across all episodes."""
        summaries = [ep.get_summary() for ep in self.episodes_data]
        
        self.summary_stats = {
            'num_episodes': len(summaries),
            'success_rate': np.mean([s['success'] for s in summaries]),
            'avg_episode_reward': np.mean([s['total_reward'] for s in summaries]),
            'std_episode_reward': np.std([s['total_reward'] for s in summaries]),
            'avg_episode_length': np.mean([s['episode_length'] for s in summaries]),
            'std_episode_length': np.std([s['episode_length'] for s in summaries]),
            'avg_final_distance': np.mean([s['final_distance'] for s in summaries]),
            'avg_min_distance': np.mean([s['min_distance'] for s in summaries]),
            'avg_visibility_ratio': np.mean([s['visibility_ratio'] for s in summaries]),
            'avg_alignment': np.mean([s['avg_alignment'] for s in summaries]),
            'avg_forward_vel': np.mean([s['avg_forward_vel'] for s in summaries]),
        }
        
        # Compute efficiency (straight line distance / actual path length)
        efficiencies = []
        for ep in self.episodes_data:
            if len(ep.positions) > 1 and ep.goal_position:
                straight_line = np.linalg.norm(
                    np.array(ep.positions[0]) - np.array(ep.goal_position)
                )
                path_length = sum(
                    np.linalg.norm(np.array(ep.positions[i+1]) - np.array(ep.positions[i]))
                    for i in range(len(ep.positions) - 1)
                )
                if path_length > 0:
                    efficiencies.append(straight_line / path_length)
        
        self.summary_stats['efficiency_ratio'] = np.mean(efficiencies) if efficiencies else 0.0
    
    def _print_progress(self, episode_num):
        """Print progress update."""
        recent = self.episodes_data[-10:]
        recent_success = np.mean([ep.success for ep in recent])
        recent_reward = np.mean([ep.total_reward for ep in recent])
        
        print(f"  Episodes {episode_num-9}-{episode_num}: "
              f"Success={recent_success:.1%}, "
              f"Avg Reward={recent_reward:.1f}")

# VISUALIZATION
class Visualizer:
    """Creates visualizations from evaluation data."""
    
    def __init__(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def plot_summary_metrics(self, summary_stats, save_path=None):
        """Plot summary bar chart of key metrics."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Model Performance Summary', fontsize=16, fontweight='bold')
        
        # Success Rate
        ax = axes[0, 0]
        success_rate = summary_stats['success_rate'] * 100
        colors = ['green' if success_rate > 50 else 'orange' if success_rate > 25 else 'red']
        ax.bar(['Success Rate'], [success_rate], color=colors)
        ax.set_ylim(0, 100)
        ax.set_ylabel('Percentage (%)')
        ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
        ax.text(0, success_rate + 5, f'{success_rate:.1f}%', ha='center', fontweight='bold')
        
        # Average Reward
        ax = axes[0, 1]
        avg_reward = summary_stats['avg_episode_reward']
        std_reward = summary_stats['std_episode_reward']
        ax.bar(['Avg Reward'], [avg_reward], yerr=[std_reward], capsize=5)
        ax.set_ylabel('Reward')
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.text(0, avg_reward + std_reward + 5, f'{avg_reward:.1f}±{std_reward:.1f}', 
                ha='center', fontweight='bold')
        
        # Episode Length
        ax = axes[0, 2]
        avg_length = summary_stats['avg_episode_length']
        std_length = summary_stats['std_episode_length']
        ax.bar(['Avg Length'], [avg_length], yerr=[std_length], capsize=5, color='purple')
        ax.set_ylabel('Steps')
        ax.text(0, avg_length + std_length + 20, f'{avg_length:.0f}±{std_length:.0f}', 
                ha='center', fontweight='bold')
        
        # Goal Distance
        ax = axes[1, 0]
        final_dist = summary_stats['avg_final_distance']
        min_dist = summary_stats['avg_min_distance']
        ax.bar(['Final', 'Closest'], [final_dist, min_dist], color=['coral', 'lightblue'])
        ax.set_ylabel('Distance (m)')
        ax.set_title('Distance to Goal')
        
        # Visibility Ratio
        ax = axes[1, 1]
        vis_ratio = summary_stats['avg_visibility_ratio'] * 100
        ax.bar(['Visibility'], [vis_ratio], color='skyblue')
        ax.set_ylim(0, 100)
        ax.set_ylabel('Percentage (%)')
        ax.text(0, vis_ratio + 5, f'{vis_ratio:.1f}%', ha='center', fontweight='bold')
        
        # Efficiency
        ax = axes[1, 2]
        efficiency = summary_stats['efficiency_ratio'] * 100
        ax.bar(['Efficiency'], [efficiency], color='gold')
        ax.set_ylim(0, 100)
        ax.set_ylabel('Percentage (%)')
        ax.set_title('Path Efficiency')
        ax.text(0, efficiency + 5, f'{efficiency:.1f}%', ha='center', fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Summary plot saved: {save_path}")
        
        return fig
    
    def plot_episode_trajectory(self, episode_data, save_path=None, title=None):
        """Plot robot trajectory for a single episode."""
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        if title:
            fig.suptitle(title, fontsize=14, fontweight='bold')
        
        positions = np.array(episode_data.positions)
        goal_pos = episode_data.goal_position
        
        # Left plot: Trajectory
        ax = axes[0]
        ax.plot(positions[:, 0], positions[:, 1], 'b-', linewidth=2, label='Robot path')
        ax.plot(positions[0, 0], positions[0, 1], 'go', markersize=12, label='Start')
        ax.plot(positions[-1, 0], positions[-1, 1], 'ro', markersize=12, label='End')
        
        if goal_pos:
            # Goal with tolerance circle
            circle = patches.Circle(goal_pos, 1.0, color='purple', alpha=0.3, label='Goal (1m radius)')
            ax.add_patch(circle)
            ax.plot(goal_pos[0], goal_pos[1], 'p', color='purple', markersize=15)
        
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        ax.set_title(f'Episode Trajectory\n'
                     f'Success: {episode_data.success} | '
                     f'Length: {episode_data.episode_length} steps | '
                     f'Reward: {episode_data.total_reward:.1f}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        
        # Right plot: Metrics over time
        ax = axes[1]
        steps = range(len(episode_data.distances))
        
        # Distance
        ax.plot(steps, episode_data.distances, 'b-', linewidth=2, label='Distance to goal')
        ax.axhline(y=1.0, color='purple', linestyle='--', alpha=0.5, label='Goal threshold')
        ax.set_xlabel('Steps')
        ax.set_ylabel('Distance (m)', color='b')
        ax.tick_params(axis='y', labelcolor='b')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # Visibility (on secondary axis)
        ax2 = ax.twinx()
        visibility_viz = [1 if v else 0 for v in episode_data.goal_visible]
        ax2.fill_between(steps, visibility_viz, alpha=0.3, color='orange', label='Goal visible')
        ax2.set_ylabel('Goal Visible', color='orange')
        ax2.set_ylim(-0.1, 1.1)
        ax2.set_yticks([0, 1])
        ax2.set_yticklabels(['No', 'Yes'])
        ax2.tick_params(axis='y', labelcolor='orange')
        ax2.legend(loc='upper right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_action_distributions(self, episodes_data, save_path=None):
        """Plot distribution of actions taken."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Action Distribution Analysis', fontsize=16, fontweight='bold')
        
        # Collect all actions
        all_actions = []
        for ep in episodes_data:
            all_actions.extend(ep.actions)
        all_actions = np.array(all_actions)
        
        linear_vel = all_actions[:, 0]
        angular_vel = all_actions[:, 1]
        
        # Linear velocity distribution
        ax = axes[0, 0]
        ax.hist(linear_vel, bins=50, edgecolor='black', alpha=0.7)
        ax.axvline(linear_vel.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {linear_vel.mean():.3f}')
        ax.set_xlabel('Linear Velocity')
        ax.set_ylabel('Frequency')
        ax.set_title('Linear Velocity Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Angular velocity distribution
        ax = axes[0, 1]
        ax.hist(angular_vel, bins=50, edgecolor='black', alpha=0.7, color='orange')
        ax.axvline(angular_vel.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {angular_vel.mean():.3f}')
        ax.set_xlabel('Angular Velocity')
        ax.set_ylabel('Frequency')
        ax.set_title('Angular Velocity Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2D action space scatter
        ax = axes[1, 0]
        scatter = ax.scatter(linear_vel, angular_vel, alpha=0.1, s=1)
        ax.set_xlabel('Linear Velocity')
        ax.set_ylabel('Angular Velocity')
        ax.set_title('Action Space Coverage')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        
        # Action magnitude over time (averaged across episodes)
        ax = axes[1, 1]
        max_len = max(len(ep.actions) for ep in episodes_data)
        linear_over_time = np.zeros(max_len)
        angular_over_time = np.zeros(max_len)
        counts = np.zeros(max_len)
        
        for ep in episodes_data:
            for i, action in enumerate(ep.actions):
                linear_over_time[i] += abs(action[0])
                angular_over_time[i] += abs(action[1])
                counts[i] += 1
        
        linear_over_time /= (counts + 1e-8)
        angular_over_time /= (counts + 1e-8)
        
        ax.plot(linear_over_time, label='|Linear velocity|', linewidth=2)
        ax.plot(angular_over_time, label='|Angular velocity|', linewidth=2)
        ax.set_xlabel('Episode Step')
        ax.set_ylabel('Absolute Action Value')
        ax.set_title('Average Action Magnitude Over Episode')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Action distribution plot saved: {save_path}")
        
        return fig
    
    def plot_reward_analysis(self, episodes_data, save_path=None):
        """Analyze reward components and temporal patterns."""
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(3, 2, figure=fig)
        fig.suptitle('Reward Analysis', fontsize=16, fontweight='bold')
        
        # Episode rewards distribution
        ax = fig.add_subplot(gs[0, 0])
        rewards = [ep.total_reward for ep in episodes_data]
        ax.hist(rewards, bins=30, edgecolor='black', alpha=0.7, color='green')
        ax.axvline(np.mean(rewards), color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {np.mean(rewards):.1f}')
        ax.set_xlabel('Total Episode Reward')
        ax.set_ylabel('Frequency')
        ax.set_title('Episode Reward Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Success vs failure rewards
        ax = fig.add_subplot(gs[0, 1])
        success_rewards = [ep.total_reward for ep in episodes_data if ep.success]
        failure_rewards = [ep.total_reward for ep in episodes_data if not ep.success]
        
        bp = ax.boxplot([success_rewards, failure_rewards], 
                        labels=['Success', 'Failure'],
                        patch_artist=True)
        bp['boxes'][0].set_facecolor('lightgreen')
        bp['boxes'][1].set_facecolor('lightcoral')
        ax.set_ylabel('Total Reward')
        ax.set_title('Reward by Outcome')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Reward accumulation over episode
        ax = fig.add_subplot(gs[1, :])
        max_len = max(len(ep.rewards) for ep in episodes_data)
        cumulative_rewards = np.zeros((len(episodes_data), max_len))
        
        for i, ep in enumerate(episodes_data):
            cumsum = np.cumsum(ep.rewards)
            cumulative_rewards[i, :len(cumsum)] = cumsum
            cumulative_rewards[i, len(cumsum):] = cumsum[-1] if len(cumsum) > 0 else 0
        
        mean_cumulative = np.mean(cumulative_rewards, axis=0)
        std_cumulative = np.std(cumulative_rewards, axis=0)
        
        ax.plot(mean_cumulative, linewidth=2, label='Mean')
        ax.fill_between(range(max_len), 
                        mean_cumulative - std_cumulative,
                        mean_cumulative + std_cumulative,
                        alpha=0.3)
        ax.set_xlabel('Episode Step')
        ax.set_ylabel('Cumulative Reward')
        ax.set_title('Reward Accumulation Over Episode')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Step rewards distribution
        ax = fig.add_subplot(gs[2, 0])
        all_step_rewards = []
        for ep in episodes_data:
            all_step_rewards.extend(ep.rewards)
        
        ax.hist(all_step_rewards, bins=50, edgecolor='black', alpha=0.7, color='skyblue')
        ax.axvline(0, color='black', linestyle='-', linewidth=1)
        ax.axvline(np.mean(all_step_rewards), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {np.mean(all_step_rewards):.3f}')
        ax.set_xlabel('Step Reward')
        ax.set_ylabel('Frequency')
        ax.set_title('Per-Step Reward Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Episode length vs reward correlation
        ax = fig.add_subplot(gs[2, 1])
        lengths = [ep.episode_length for ep in episodes_data]
        rewards = [ep.total_reward for ep in episodes_data]
        colors = ['green' if ep.success else 'red' for ep in episodes_data]
        
        ax.scatter(lengths, rewards, c=colors, alpha=0.6)
        ax.set_xlabel('Episode Length (steps)')
        ax.set_ylabel('Total Reward')
        ax.set_title('Episode Length vs Reward')
        ax.grid(True, alpha=0.3)
        
        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Success'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Failure')
        ]
        ax.legend(handles=legend_elements)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Reward analysis plot saved: {save_path}")
        
        return fig
    
    def plot_behavioral_analysis(self, episodes_data, save_path=None):
        """Analyze behavioral patterns."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Behavioral Analysis', fontsize=16, fontweight='bold')
        
        # Visibility ratio per episode
        ax = axes[0, 0]
        visibility_ratios = [np.mean(ep.goal_visible) * 100 for ep in episodes_data]
        colors = ['green' if ep.success else 'red' for ep in episodes_data]
        ax.scatter(range(len(visibility_ratios)), visibility_ratios, c=colors, alpha=0.6)
        ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Goal Visibility (%)')
        ax.set_title('Goal Visibility Ratio Per Episode')
        ax.grid(True, alpha=0.3)
        
        # Alignment score over time
        ax = axes[0, 1]
        max_len = max(len(ep.goal_offsets) for ep in episodes_data)
        alignment_over_time = np.zeros(max_len)
        counts = np.zeros(max_len)
        
        for ep in episodes_data:
            for i, score in enumerate(ep.goal_offsets):
                alignment_over_time[i] += score
                counts[i] += 1
        
        alignment_over_time /= (counts + 1e-8)
        ax.plot(alignment_over_time, linewidth=2)
        ax.set_xlabel('Episode Step')
        ax.set_ylabel('Average Alignment Score')
        ax.set_title('Goal Alignment Over Episode')
        ax.grid(True, alpha=0.3)
        
        # Distance reduction rate
        ax = axes[1, 0]
        distance_reduction_rates = []
        for ep in episodes_data:
            if len(ep.distances) > 1:
                initial_dist = ep.distances[0]
                final_dist = ep.distances[-1]
                reduction = (initial_dist - final_dist) / initial_dist * 100
                distance_reduction_rates.append(reduction)
        
        ax.hist(distance_reduction_rates, bins=30, edgecolor='black', alpha=0.7, color='teal')
        ax.axvline(0, color='red', linestyle='--', linewidth=1)
        ax.axvline(np.mean(distance_reduction_rates), color='blue', linestyle='--', linewidth=2,
                   label=f'Mean: {np.mean(distance_reduction_rates):.1f}%')
        ax.set_xlabel('Distance Reduction (%)')
        ax.set_ylabel('Frequency')
        ax.set_title('Distance Reduction Rate')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Success rate by visibility
        ax = axes[1, 1]
        visibility_bins = [0, 25, 50, 75, 100]
        success_by_visibility = []
        
        for i in range(len(visibility_bins) - 1):
            low, high = visibility_bins[i], visibility_bins[i+1]
            episodes_in_bin = [
                ep for ep in episodes_data
                if low <= np.mean(ep.goal_visible) * 100 < high
            ]
            if episodes_in_bin:
                success_rate = np.mean([ep.success for ep in episodes_in_bin]) * 100
                success_by_visibility.append(success_rate)
            else:
                success_by_visibility.append(0)
        
        bin_labels = ['0-25%', '25-50%', '50-75%', '75-100%']
        bars = ax.bar(bin_labels, success_by_visibility, edgecolor='black', alpha=0.7)
        
        # Color bars
        for i, bar in enumerate(bars):
            if success_by_visibility[i] > 75:
                bar.set_color('green')
            elif success_by_visibility[i] > 50:
                bar.set_color('yellow')
            else:
                bar.set_color('red')
        
        ax.set_xlabel('Visibility Ratio Bins')
        ax.set_ylabel('Success Rate (%)')
        ax.set_title('Success Rate by Visibility')
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Behavioral analysis plot saved: {save_path}")
        
        return fig

# CHECKPOINT COMPARISON
class CheckpointComparator:
    """Compare multiple checkpoints side-by-side."""
    
    def __init__(self, env, device):
        self.env = env
        self.device = device
        self.results = {}
    
    def compare_checkpoints(self, checkpoint_paths, episodes_per_checkpoint=50):
        """Evaluate multiple checkpoints."""
        print(f"\n{'='*60}")
        print(f"   CHECKPOINT COMPARISON")
        print(f"   Checkpoints: {len(checkpoint_paths)}")
        print(f"   Episodes per checkpoint: {episodes_per_checkpoint}")
        print(f"{'='*60}\n")
        
        for checkpoint_path in checkpoint_paths:
            print(f"\nLoading: {checkpoint_path}")
            
            # Load model
            policy = CNNPolicy(
                in_channels=12,
                action_dim=2,
                input_height=84,
                input_width=84,
            ).to(self.device)
            
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            policy.load_state_dict(checkpoint['policy_state_dict'])
            policy.eval()
            
            # Evaluate
            evaluator = ModelEvaluator(self.env, policy, self.device, deterministic=True)
            summary = evaluator.evaluate(
                episodes_per_checkpoint,
                desc=f"Evaluating {os.path.basename(checkpoint_path)}"
            )
            
            # Store results
            checkpoint_name = os.path.basename(checkpoint_path)
            self.results[checkpoint_name] = {
                'summary': summary,
                'episodes_data': evaluator.episodes_data,
            }
        
        return self.results
    
    def plot_comparison(self, save_path=None):
        """Create comparison plots."""
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        fig.suptitle('Checkpoint Comparison', fontsize=16, fontweight='bold')
        
        checkpoints = list(self.results.keys())
        
        # Success Rate
        ax = axes[0, 0]
        success_rates = [self.results[cp]['summary']['success_rate'] * 100 for cp in checkpoints]
        bars = ax.bar(range(len(checkpoints)), success_rates)
        for i, bar in enumerate(bars):
            if success_rates[i] > 75:
                bar.set_color('green')
            elif success_rates[i] > 50:
                bar.set_color('yellow')
            else:
                bar.set_color('red')
        ax.set_xticks(range(len(checkpoints)))
        ax.set_xticklabels(checkpoints, rotation=45, ha='right')
        ax.set_ylabel('Success Rate (%)')
        ax.set_ylim(0, 100)
        ax.set_title('Success Rate')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Average Reward
        ax = axes[0, 1]
        rewards = [self.results[cp]['summary']['avg_episode_reward'] for cp in checkpoints]
        ax.bar(range(len(checkpoints)), rewards, color='skyblue')
        ax.set_xticks(range(len(checkpoints)))
        ax.set_xticklabels(checkpoints, rotation=45, ha='right')
        ax.set_ylabel('Average Reward')
        ax.set_title('Average Episode Reward')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Episode Length
        ax = axes[0, 2]
        lengths = [self.results[cp]['summary']['avg_episode_length'] for cp in checkpoints]
        ax.bar(range(len(checkpoints)), lengths, color='orange')
        ax.set_xticks(range(len(checkpoints)))
        ax.set_xticklabels(checkpoints, rotation=45, ha='right')
        ax.set_ylabel('Average Length (steps)')
        ax.set_title('Average Episode Length')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Goal Distance
        ax = axes[1, 0]
        final_dists = [self.results[cp]['summary']['avg_final_distance'] for cp in checkpoints]
        ax.bar(range(len(checkpoints)), final_dists, color='purple')
        ax.set_xticks(range(len(checkpoints)))
        ax.set_xticklabels(checkpoints, rotation=45, ha='right')
        ax.set_ylabel('Final Distance (m)')
        ax.set_title('Average Final Distance to Goal')
        ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Goal threshold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # Visibility
        ax = axes[1, 1]
        visibility = [self.results[cp]['summary']['avg_visibility_ratio'] * 100 for cp in checkpoints]
        ax.bar(range(len(checkpoints)), visibility, color='gold')
        ax.set_xticks(range(len(checkpoints)))
        ax.set_xticklabels(checkpoints, rotation=45, ha='right')
        ax.set_ylabel('Visibility (%)')
        ax.set_title('Average Goal Visibility')
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Efficiency
        ax = axes[1, 2]
        efficiency = [self.results[cp]['summary']['efficiency_ratio'] * 100 for cp in checkpoints]
        ax.bar(range(len(checkpoints)), efficiency, color='teal')
        ax.set_xticks(range(len(checkpoints)))
        ax.set_xticklabels(checkpoints, rotation=45, ha='right')
        ax.set_ylabel('Efficiency (%)')
        ax.set_title('Path Efficiency')
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Comparison plot saved: {save_path}")
        
        return fig

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)

# REPORT GENERATOR
def generate_report(summary_stats, output_path):
    """Generate a text report of evaluation results."""
    report = f"""
{'='*70}
EVALUATION REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*70}

OVERVIEW
--------
Total Episodes: {summary_stats['num_episodes']}

PERFORMANCE METRICS
-------------------
Success Rate:           {summary_stats['success_rate']*100:.2f}%
Average Episode Reward: {summary_stats['avg_episode_reward']:.2f} ± {summary_stats['std_episode_reward']:.2f}
Average Episode Length: {summary_stats['avg_episode_length']:.1f} ± {summary_stats['std_episode_length']:.1f} steps

DISTANCE METRICS
----------------
Average Final Distance: {summary_stats['avg_final_distance']:.3f} m
Average Min Distance:   {summary_stats['avg_min_distance']:.3f} m
Path Efficiency:        {summary_stats['efficiency_ratio']*100:.2f}%

BEHAVIORAL METRICS
------------------
Goal Visibility Ratio:  {summary_stats['avg_visibility_ratio']*100:.2f}%
Average Alignment:      {summary_stats['avg_alignment']:.3f}
Avg Forward Velocity:   {summary_stats['avg_forward_vel']:.3f}

INTERPRETATION
--------------
"""
    
    # Add interpretation
    if summary_stats['success_rate'] > 0.75:
        report += "EXCELLENT: Model shows strong performance with >75% success rate.\n"
    elif summary_stats['success_rate'] > 0.5:
        report += "GOOD: Model performs well with >50% success rate.\n"
    elif summary_stats['success_rate'] > 0.25:
        report += "MODERATE: Model shows learning but needs improvement.\n"
    else:
        report += "❌ POOR: Model needs more training or tuning.\n"
    
    if summary_stats['avg_visibility_ratio'] > 0.7:
        report += "Model effectively maintains visual contact with goal.\n"
    elif summary_stats['avg_visibility_ratio'] > 0.4:
        report += "Model sometimes loses sight of goal.\n"
    else:
        report += "Model struggles to keep goal in view.\n"
    
    if summary_stats['efficiency_ratio'] > 0.7:
        report += "Model takes efficient paths to goal.\n"
    elif summary_stats['efficiency_ratio'] > 0.5:
        report += "Model's paths are somewhat indirect.\n"
    else:
        report += "Model takes very indirect paths.\n"
    
    report += "\n" + "="*70 + "\n"
    
    # Write to file
    with open(output_path, 'w') as f:
        f.write(report)
    
    print(f"📄 Report saved: {output_path}")
    
    return report

# MAIN TESTING FUNCTIONS
def test_single_checkpoint(checkpoint_path, num_episodes=50, output_dir=None):
    """Test a single checkpoint comprehensively."""
    
    # Setup output directory
    if output_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = os.path.join(TestConfig.OUTPUT_DIR, f'test_{timestamp}')
    
    os.makedirs(output_dir, exist_ok=True)
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"   TESTING CHECKPOINT: {checkpoint_path}")
    print(f"   Episodes: {num_episodes}")
    print(f"   Output: {output_dir}")
    print(f"{'='*70}\n")
    
    # Initialize ROS
    rclpy.init()
    camera_node = CameraNode()
    
    # Create environment
    env = RosVisionEnv(
        node=camera_node,
        camera_node=camera_node,
        frame_stack_k=4,
        max_steps=1000,
    )
    
    # Load model
    policy = CNNPolicy(
        in_channels=12,
        action_dim=2,
        input_height=84,
        input_width=84,
    ).to(TestConfig.DEVICE)
    
    checkpoint = torch.load(checkpoint_path, map_location=TestConfig.DEVICE)
    policy.load_state_dict(checkpoint['policy_state_dict'])
    policy.eval()
    
    print(f"Model loaded from episode {checkpoint.get('episode', 'unknown')}")
    
    # Evaluate
    evaluator = ModelEvaluator(env, policy, TestConfig.DEVICE, deterministic=True)
    summary_stats = evaluator.evaluate(num_episodes)
    
    # Print summary
    print(f"\n{'='*70}")
    print("RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"Success Rate:     {summary_stats['success_rate']*100:.2f}%")
    print(f"Avg Reward:       {summary_stats['avg_episode_reward']:.2f} ± {summary_stats['std_episode_reward']:.2f}")
    print(f"Avg Length:       {summary_stats['avg_episode_length']:.1f} ± {summary_stats['std_episode_length']:.1f} steps")
    print(f"Path Efficiency:  {summary_stats['efficiency_ratio']*100:.2f}%")
    print(f"Goal Visibility:  {summary_stats['avg_visibility_ratio']*100:.2f}%")
    print(f"{'='*70}\n")
    
    # Generate visualizations
    visualizer = Visualizer(plots_dir)
    
    print("Generating visualizations...")
    
    # 1. Summary metrics
    visualizer.plot_summary_metrics(
        summary_stats,
        save_path=os.path.join(plots_dir, 'summary_metrics.png')
    )
    
    # 2. Action distributions
    visualizer.plot_action_distributions(
        evaluator.episodes_data,
        save_path=os.path.join(plots_dir, 'action_distributions.png')
    )
    
    # 3. Reward analysis
    visualizer.plot_reward_analysis(
        evaluator.episodes_data,
        save_path=os.path.join(plots_dir, 'reward_analysis.png')
    )
    
    # 4. Behavioral analysis
    visualizer.plot_behavioral_analysis(
        evaluator.episodes_data,
        save_path=os.path.join(plots_dir, 'behavioral_analysis.png')
    )
    
    # 5. Individual episode trajectories (best, worst, median)
    # Best episode
    best_ep = max(evaluator.episodes_data, key=lambda ep: ep.total_reward)
    visualizer.plot_episode_trajectory(
        best_ep,
        save_path=os.path.join(plots_dir, 'trajectory_best.png'),
        title=f'Best Episode (Reward: {best_ep.total_reward:.1f})'
    )
    
    # Worst episode
    worst_ep = min(evaluator.episodes_data, key=lambda ep: ep.total_reward)
    visualizer.plot_episode_trajectory(
        worst_ep,
        save_path=os.path.join(plots_dir, 'trajectory_worst.png'),
        title=f'Worst Episode (Reward: {worst_ep.total_reward:.1f})'
    )
    
    # Median episode
    sorted_eps = sorted(evaluator.episodes_data, key=lambda ep: ep.total_reward)
    median_ep = sorted_eps[len(sorted_eps) // 2]
    visualizer.plot_episode_trajectory(
        median_ep,
        save_path=os.path.join(plots_dir, 'trajectory_median.png'),
        title=f'Median Episode (Reward: {median_ep.total_reward:.1f})'
    )
    
    # Generate report
    report = generate_report(
        summary_stats,
        os.path.join(output_dir, 'report.txt')
    )
    print(report)
    
    # Save detailed results as JSON
    results_json = {
        'checkpoint': checkpoint_path,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'summary': summary_stats,
        'episodes': [ep.get_summary() for ep in evaluator.episodes_data],
    }
    
    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(results_json, f, indent=2, cls=NumpyEncoder)
    
    print(f"Results saved to: {output_dir}")
    
    # Cleanup
    env.close()
    rclpy.shutdown()
    
    return summary_stats, evaluator.episodes_data


def compare_multiple_checkpoints(checkpoint_paths, episodes_per_checkpoint=50):
    """Compare multiple checkpoints side-by-side."""
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(TestConfig.OUTPUT_DIR, f'comparison_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize ROS
    rclpy.init()
    camera_node = CameraNode()
    
    # Create environment
    env = RosVisionEnv(
        node=camera_node,
        camera_node=camera_node,
        frame_stack_k=4,
        max_steps=1000,
    )
    
    # Compare
    comparator = CheckpointComparator(env, TestConfig.DEVICE)
    results = comparator.compare_checkpoints(checkpoint_paths, episodes_per_checkpoint)
    
    # Create comparison plot
    comparator.plot_comparison(
        save_path=os.path.join(output_dir, 'checkpoint_comparison.png')
    )
    
    # Save comparison report
    with open(os.path.join(output_dir, 'comparison_report.txt'), 'w') as f:
        f.write(f"CHECKPOINT COMPARISON\n")
        f.write(f"{'='*70}\n\n")
        
        for checkpoint_name, data in results.items():
            f.write(f"\n{checkpoint_name}\n")
            f.write(f"{'-'*70}\n")
            f.write(f"Success Rate:     {data['summary']['success_rate']*100:.2f}%\n")
            f.write(f"Avg Reward:       {data['summary']['avg_episode_reward']:.2f}\n")
            f.write(f"Avg Length:       {data['summary']['avg_episode_length']:.1f}\n")
            f.write(f"Path Efficiency:  {data['summary']['efficiency_ratio']*100:.2f}%\n")
            f.write(f"\n")
    
    print(f"\nComparison results saved to: {output_dir}")
    
    # Cleanup
    env.close()
    rclpy.shutdown()
    
    return results


# COMMAND LINE INTERFACE
def main():
    parser = argparse.ArgumentParser(description='Test trained PPO navigation model')
    parser.add_argument('--checkpoint', type=str, help='Path to checkpoint file')
    parser.add_argument('--episodes', type=int, default=50, help='Number of test episodes')
    parser.add_argument('--compare', nargs='+', help='Compare multiple checkpoints')
    parser.add_argument('--output', type=str, help='Output directory')
    
    args = parser.parse_args()
    
    if args.compare:
        # Comparison mode
        compare_multiple_checkpoints(args.compare, args.episodes)
    
    elif args.checkpoint:
        # Single checkpoint test
        test_single_checkpoint(args.checkpoint, args.episodes, args.output)
    
    else:
        print("Error: Must specify --checkpoint or --compare")
        print("\nExamples:")
        print("  # Test single checkpoint:")
        print("  python test_ppo_model.py --checkpoint checkpoints/ppo_best.pt --episodes 50")
        print("\n  # Compare multiple checkpoints:")
        print("  python test_ppo_model.py --compare checkpoints/ppo_ep_00100.pt checkpoints/ppo_ep_01000.pt --episodes 30")


if __name__ == "__main__":
    main()