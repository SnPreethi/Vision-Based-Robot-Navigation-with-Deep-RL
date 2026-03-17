# PPO LOGIC (continuous action spaces)

# importing libraries
import torch
import torch.nn.functional as F
from torch.distributions import Normal

class PPOAgent:
    def __init__(
            self,
            policy,
            lr=3e-4,
            clip_eps=0.2,
            value_coef=0.5,
            entropy_coef=0.01,
            epochs=4,
            batch_size=64,):
        
        self.policy = policy
        self.optim = torch.optim.Adam(policy.parameters(), lr=lr)

        self.clip_eps = clip_eps
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.epochs = epochs
        self.batch_size = batch_size

    def act(self, obs):
        mean, std, value = self.policy(obs)
        dist = Normal(mean, std)
        raw_action = dist.rsample()
        action = torch.tanh(raw_action)
        log_prob = dist.log_prob(raw_action) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(-1)

        return action, log_prob, value.squeeze(-1), raw_action

    def update(self, batch):
        # explicitly adding train mode
        self.policy.train()

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        update_count = 0

        # unpacking batch
        obs = batch["obs"]   
        actions = batch["actions"]
        old_log_probs = batch["log_probs"]
        returns = batch["returns"]
        advantages = batch["advantages"]

        old_values = batch.get('old_values', None)

        # Advantage normalization for stability by keeping advantages mean centered
        advantages = advantages - advantages.mean()
        adv_std = advantages.std(unbiased=False)
        if adv_std > 1e-6:
            advantages = advantages / (adv_std + 1e-8)

        N = obs['image'].size(0)
        idxs = torch.randperm(N)

        for _ in range(self.epochs):
            for start in range(0, N, self.batch_size):
                end = start + self.batch_size
                mb_idx = idxs[start:end]

                device = next(self.policy.parameters()).device

                mb_obs = {
                    'image': obs['image'][mb_idx].to(device),
                    'goal_distance': obs['goal_distance'][mb_idx].to(device),
                }
                mb_actions = actions[mb_idx].to(device)
                mb_old_log_probs = old_log_probs[mb_idx].to(device)
                mb_returns = returns[mb_idx].to(device)
                mb_advantages = advantages[mb_idx].to(device)

                # Forward pass
                mean, std, values = self.policy(mb_obs)
                values = values.squeeze(-1)

                dist = Normal(mean, std)
                eps = 1e-6
                raw_actions = torch.atanh(torch.clamp(mb_actions, -1 + eps, 1 - eps))
                new_log_probs = dist.log_prob(raw_actions)
                new_log_probs -= torch.log(1 - mb_actions.pow(2) + eps)
                new_log_probs = new_log_probs.sum(-1)
                
                entropy = dist.entropy().sum(-1).mean()

                # Probability ratio
                ratio = torch.exp(new_log_probs - mb_old_log_probs)

                # Clipped surrogate objective
                unclipped = ratio * mb_advantages
                clipped = torch.clamp(
                    ratio,
                    1.0 - self.clip_eps,
                    1.0 + self.clip_eps
                ) * mb_advantages

                policy_loss = -torch.min(unclipped, clipped).mean()

                # Value function loss
                # Value-function clipping from the PPO paper
                if old_values is not None:
                    mb_old_values = old_values[mb_idx].to(device)
                    
                    value_pred_clipped = mb_old_values + torch.clamp(
                        values - mb_old_values,
                        -self.clip_eps,
                        self.clip_eps
                    )
                    
                    value_loss = torch.max(
                        (values - mb_returns).pow(2),
                        (value_pred_clipped - mb_returns).pow(2)
                    ).mean()
                
                else:
                    value_loss = F.mse_loss(values, mb_returns)

                # Total PPO loss
                loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    - self.entropy_coef * entropy
                )

                # Optimization step
                self.optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optim.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                update_count += 1
        
        n = max(1, update_count)
        return {
            'policy_loss': total_policy_loss / n,
            'value_loss': total_value_loss / n,
            'entropy': total_entropy / n,
        }