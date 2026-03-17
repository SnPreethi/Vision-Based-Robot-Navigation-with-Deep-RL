# PPO ROLLOUT BUFFER
'''
Store transitions collected during interaction
Store value predictions from the critic
Compute returns and advantages using GAE
Output tensors ready for PPO updates
Reset cleanly between episodes
It must not interact with ROS, call the policy, update the network
'''

# importing libraries
import torch

class RolloutBuffer:
    def __init__(self):
        self.clear()

    # reuse in every episode
    def clear(self):
        self.obs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []

    # transition method called every environment step
    def add(self, obs, action, reward, done, log_prob, value):
        self.obs.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)
    
    # GAE implementation
    def compute_returns_and_advantages(self, gamma=0.99, lam=0.95):
        returns = []
        advantages = []

        gae = 0.0
        next_value = 0.0

        for t in reversed(range(len(self.rewards))):
            mask = 1.0 - float(self.dones[t])

            delta = (
                self.rewards[t]
                + gamma * next_value * mask
                - self.values[t]
            )

            gae = delta + gamma * lam * mask * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + self.values[t])

            next_value = self.values[t]

        obs_image = torch.stack([o["image"] for o in self.obs])
        obs_goal = torch.stack([o["goal_distance"] for o in self.obs])
        obs_batch = {
            "image": obs_image,
            "goal_distance": obs_goal,
        }

        return (
            obs_batch,
            torch.stack(self.actions),
            torch.stack(self.log_probs),
            torch.tensor(returns, dtype=torch.float32, device=obs_image.device),
            torch.tensor(advantages, dtype=torch.float32, device=obs_image.device),
            torch.tensor(self.values, dtype=torch.float32, device=obs_image.device),
        )
