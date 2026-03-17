# MAIN TRAINING ENTRY POINT (PPO + ROS2 + Vision)

# importing libraries
import rclpy
import torch
import numpy as np
import os, logging
from datetime import datetime
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from _1_camera_node import CameraNode
from _7_env_ros2 import RosVisionEnv
from _4A_policy_cnn_ppo import CNNPolicy
from _5_ppo_agent import PPOAgent
from _7_env_ros2 import scale_action
from rollout_buffer import RolloutBuffer

# GLOBAL TRAINING VARS
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TOTAL_EPISODES = 100000

# Update Frequency for PPO
UPDATE_EVERY_STEPS = 2048  # PPO Update frequency
MIN_BUFFER_SIZE = 512  # Minimum transitions before first update

def setup_logger(log_dir='logs'):
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(
        log_dir,
        f'train_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    )
    logging.basicConfig(
        level = logging.INFO,
        format = '%(asctime)s [%(levelname)s] %(message)s',
        handlers = [
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger('RL-TRAIN')

def save_checkpoint(agent, policy, episode, global_step, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "policy_state_dict": policy.state_dict(),
        "optimizer_state_dict": agent.optim.state_dict(),
        "episode": episode,
        "global_step": global_step,
    }, path)

def load_checkpoint(agent, policy, path):
    checkpoint = torch.load(path)
    policy.load_state_dict(checkpoint["policy_state_dict"])
    agent.optim.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["episode"], checkpoint["global_step"]

def main():
    # Logger setup
    logger = setup_logger()
    logger.info('Starting RL Training')

    # Device setup
    logger.info(f"Using device: {DEVICE}")
    if DEVICE.type == "cuda":
        logger.info(f"GPU name: {torch.cuda.get_device_name(0)}")
    else:
        logger.warning("CUDA not available. Training on CPU.")

    # Tensorboard setup
    tb_writer = SummaryWriter(log_dir='tensorboard')
    logger.info("TensorBoard logging enabled (tensorboard/)")

    # Checkpoint config
    CHECKPOINT_PATH = "checkpoints"
    SAVE_EVERY_EPISODES = 5
    RESUME = True # turn this to True if loading from a specific checkpoint
    RESUME_CHECKPOINT = 'checkpoints/ppo_ep_02355.pt'

    # ROS2 INIT
    rclpy.init()

    # Camera node (sensor interface)
    camera_node = CameraNode()

    # ENVIRONMENT
    env = RosVisionEnv(
        node=camera_node,
        camera_node=camera_node,
        frame_stack_k=4,
        max_steps=1000,
    )

    # SANITY CHECK - Checking the action space info
    #print(f'Action space: {env.action_space}')

    # SANITY CHECK - Environment check
    '''try:
        check_env(env)
        print('Environment check passed!')
    except Exception as e:
        print(f'Environment check failed: {e}')'''
    
    # PPO POLICY & AGENT
    policy = CNNPolicy(
        in_channels=12,   # 4 frames × 3 RGB channels
        action_dim=2,     # [v, omega]
        input_height=84,
        input_width=84,
    ).to(DEVICE)

    agent = PPOAgent(
        policy,
        lr=1e-4,
        entropy_coef=0.003,
        epochs=6,
        batch_size=128
    )

    # Rollout buffer
    buffer = RolloutBuffer()

    episode_idx = 0
    global_step = 0
    best_episode_reward = -float('inf')
    BEST_MODEL_PATH = 'checkpoints/ppo_best.pt'

    if RESUME:
        if not os.path.exists(RESUME_CHECKPOINT):
            raise FileNotFoundError(f'Checkpoint not found {RESUME_CHECKPOINT}')
        episode_idx, global_step = load_checkpoint(agent, policy, RESUME_CHECKPOINT)
        logger.info(
            f'Resumed training from checkpoint: {RESUME_CHECKPOINT} | '
            f'episode = {episode_idx}, global_step = {global_step}')

    # TRAINING LOOP
    obs, _ = env.reset()
    episode_reward = 0.0
    episode_steps = 0
    
    episode_rewards_history = []
    episode_lengths_history = []
    success_history = []

    with tqdm(
        total = TOTAL_EPISODES,
        initial = episode_idx,
        desc = 'Training Progress',
        unit = 'episode',
    ) as pbar:
        
        while rclpy.ok() and episode_idx < TOTAL_EPISODES:

            # Convert observation to GPU tensors
            with torch.no_grad():
                obs_t = {
                    "image": torch.from_numpy(obs["image"]).float().to(DEVICE),
                    "goal_distance": torch.from_numpy(obs["goal_distance"]).float().to(DEVICE),
                }
                obs_gpu = {
                    'image': obs_t['image'].unsqueeze(0),
                    'goal_distance': obs_t['goal_distance'].unsqueeze(0),
                }
                action, log_prob, value, raw_action = agent.act(obs_gpu)

            action_np = action.squeeze(0).cpu().numpy()

            # Step environment
            next_obs, reward, terminated, truncated, info = env.step(action_np)
            
            # Periodic DEBUG logging
            if episode_steps % 200 == 0:
                logger.info(f"Step {episode_steps}: reward={reward:.3f}, dist={info.get('goal_distance', 0):.2f}")
            
            # tensorboard logging
            if 'goal_distance' in info:
                tb_writer.add_scalar('Diagnostics/GoalDistance', info['goal_distance'], global_step)

            # SANITY CHECKS
            assert not torch.isnan(action).any(), "NaN in action!"
            assert not np.isnan(reward), "NaN in reward!"
            assert obs["image"].min() >= 0.0 and obs["image"].max() <= 1.0, \
                f'Image obs out of bounds: min={obs["image"].min()}, max={obs["image"].max()}'
            assert 0.0 <= obs["goal_distance"][0] <= 1.0, \
                f'Goal distance out of bounds: {obs["goal_distance"]}'

            done = terminated or truncated

            # Store transition (skipping no_goal steps as they carry no signal)
            if not info.get('no_goal', False):
                buffer.add(
                    obs = obs_t,
                    action = action.squeeze(0).cpu(),
                    log_prob = log_prob.squeeze(0).cpu(),
                    value = value.item(),
                    reward = reward,
                    done = done,
                )

            episode_reward += reward
            episode_steps += 1
            global_step += 1
            obs = next_obs

            # Action monitoring
            if global_step % 10000 == 0:
                # Policy behaviour (exploration, saturation)
                # Robot behaviour (actual motion)
                logger.info(
                    f'Policy action (normalized): '
                    f'v = {action[0,0].item():+.3f}, w = {action[0,1].item():+.3f} | '
                    f'Cmd_vel (scaled): '
                    f'v = {action_np[0]:+.3f} m/s, w = {action_np[1]:+.3f} rad/s'
                )

            if global_step % 200 == 0:
                # plots for Linear vs angular actions
                tb_writer.add_scalar("Action/Linear_v", action[0, 0].item(), global_step)
                tb_writer.add_scalar("Action/Angular_w", action[0, 1].item(), global_step)

                v, omega = scale_action(action_np, env.linear_limit, env.angular_limit)
                # plots for raw action vs scaled action
                # policy saturation, exploration decay, smoothness over time
                tb_writer.add_scalar("ActionRaw/v", action[0, 0].item(), global_step)
                tb_writer.add_scalar("ActionRaw/w", action[0, 1].item(), global_step)
                tb_writer.add_scalar("ActionScaled/v", v, global_step)
                tb_writer.add_scalar("ActionScaled/w", omega, global_step)


            # PPO UPDATE EVERY N STEPS
            if global_step % UPDATE_EVERY_STEPS == 0 and len(buffer.rewards) >= MIN_BUFFER_SIZE:
                logger.info(f'Performing PPO update at global step {global_step}')
                # compute returns and advantages
                (
                    obs_batch, 
                    action_batch, 
                    logprob_batch, 
                    return_batch, 
                    advantage_batch, 
                    value_batch,
                ) = buffer.compute_returns_and_advantages(gamma=0.99, lam=0.95,)
                
                # Sanity Check
                assert not torch.isnan(advantage_batch).any(), 'NaN in advantages!'

                losses = agent.update({
                    "obs": obs_batch,
                    "actions": action_batch,
                    "log_probs": logprob_batch,
                    "returns": return_batch,
                    "advantages": advantage_batch,
                    "old_values": value_batch,
                })

                tb_writer.add_scalar("Loss/Policy", losses['policy_loss'], global_step)
                tb_writer.add_scalar("Loss/Value", losses['value_loss'], global_step)
                tb_writer.add_scalar("Loss/Entropy", losses['entropy'], global_step)
                logger.info(
                    f'PPO losses — policy: {losses["policy_loss"]:.4f}, '
                    f'value: {losses["value_loss"]:.4f}, '
                    f'entropy: {losses["entropy"]:.4f}'
                )

                # clearing buffer after update
                buffer.clear()
                logger.info(f'PPO update completed. Buffer cleared.')

            # Episode end handling (now just for logging and reset)
            if done:
                logger.info(
                    f"Episode {episode_idx} COMPLETE | "
                    f"Reward {episode_reward:.2f} | "
                    f"Steps {episode_steps} | "
                    f"Terminated {terminated} | "
                    f"Truncated {truncated}"
                )

                # Tensorboard episode metrics
                tb_writer.add_scalar("Episode/Reward", episode_reward, episode_idx)
                tb_writer.add_scalar("Episode/Length", episode_steps, episode_idx)
                tb_writer.add_scalar("Episode/Success", int(terminated), episode_idx)

                # Track episode history
                episode_rewards_history.append(episode_reward)
                episode_lengths_history.append(episode_steps)
                success_history.append(terminated)

                # Rolling averages (last 100 episodes)
                if len(episode_rewards_history) > 100:
                    avg_reward = np.mean(episode_rewards_history[-100:])
                    avg_length = np.mean(episode_lengths_history[-100:])
                    success_rate = np.mean(success_history[-100:])
                    tb_writer.add_scalar("Metrics/AvgReward100", avg_reward, episode_idx)
                    tb_writer.add_scalar("Metrics/AvgLength100", avg_length, episode_idx)
                    tb_writer.add_scalar("Metrics/SuccessRate100", success_rate, episode_idx)

                # Checkpoint saving
                if episode_idx % SAVE_EVERY_EPISODES == 0 and episode_idx > 0:
                    checkpoint_path = f'{CHECKPOINT_PATH}/ppo_ep_{episode_idx:05d}.pt'
                    save_checkpoint(agent, policy, episode_idx, global_step, checkpoint_path)
                    logger.info(f"Checkpoint saved: {checkpoint_path}")
                
                # save best model
                if terminated and episode_reward > best_episode_reward:
                    best_episode_reward = episode_reward
                    save_checkpoint(agent,policy, episode_idx, global_step, BEST_MODEL_PATH)
                    logger.info(f'New BEST model saved! Reward = {episode_reward:.2f} (episode {episode_idx})')

                # update progress bar
                pbar.update(1)
                pbar.set_postfix({
                    'reward': f'{episode_reward:.1f}',
                    'steps': episode_steps,
                    'success': 'SUCCESS' if terminated else 'NO SUCCESS FOR NOW'
                })

                # Reset for next episode
                obs, _ = env.reset()
                episode_reward = 0.0
                episode_steps = 0
                episode_idx += 1
            
            # Memory cleanup
            if global_step % 100 == 0 and DEVICE.type == "cuda":
                torch.cuda.empty_cache()

    # CLEAN SHUTDOWN
    env.close()
    rclpy.shutdown()
    tb_writer.close()


if __name__ == "__main__":
    main()