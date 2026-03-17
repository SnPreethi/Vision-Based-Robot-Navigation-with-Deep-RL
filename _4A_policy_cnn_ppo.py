# CNN BACKBONE (Atari-style pixel based) - PPO

# importing libraries
import torch
import torch.nn as nn

# Image dimensions
input_height = 84
input_width = 84

class CNNPolicy(nn.Module):
    def __init__(self, in_channels, action_dim, input_height=input_height, input_width=input_width):
        # initialization
        super().__init__()

        # feature extractor of Nature CNN
        # 4 channels (4x1) for grayscale and 12 channels (4x3) for RGB
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )

        # Dynamically computing CNN output size
        with torch.no_grad():
            dummy_input = torch.zeros(1, in_channels, input_height, input_width)
            cnn_out_dim = self.cnn(dummy_input).shape[1]
        
        # Fully Connected Layer
        goal_dim = 1
        self.fc = nn.Sequential(
            nn.Linear(cnn_out_dim + goal_dim, 512),
            nn.ReLU()
        )

        # policy head (actor) - Diagonal Gaussian Policy for PPO agent
        self.mean = nn.Linear(512, action_dim)
        self.log_std_layer = nn.Linear(512, action_dim)

        # PPO safe bounds for log-std
        self.LOG_STD_MIN = -5.0
        self.LOG_STD_MAX = 3.0

        # outputting scalar state value estimate V(s)
        self.value = nn.Linear(512, 1)

    # forward pass
    def forward(self, obs):

        # unpacking observation dict
        x = obs['image'] # (B, C, H, W)
        goal_dist = obs['goal_distance'] # (B, 1)

        # SANITY CHECKS
        assert x.dim() == 4, f'Expected image (B, C, H, W), got {x.shape}'
        assert goal_dist.dim() == 2, f'Expected (B, 1), got {goal_dist.shape}'

        # cnn feature extraction (B, cnn_out_dim)
        x = self.cnn(x)

        # concatenate goal distance
        x = torch.cat([x, goal_dist], dim=1)
        
        # fully connected layers
        x = self.fc(x)

        # actor
        mean = self.mean(x)
        log_std = torch.clamp(
            self.log_std_layer(x),
            self.LOG_STD_MIN,
            self.LOG_STD_MAX
        )
        std = torch.clamp(torch.exp(log_std), min=1e-3, max=2.0)

        # critic
        value = self.value(x)

        return mean, std, value