# CNN BACKBONE (Atari-style pixel based Nature CNN) - DQN

# importing libraries
import torch
import torch.nn as nn

class DQNPolicy(nn.Module):
    def __init__(self, in_channels, action_dim):
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

        # Fully Connected Layer
        # 3136 is hard-coded. Which means input resolution is 84 x 84
        # 64 x 7 x 7 = 3136 -> changing input resolution will break the network
        self.fc = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU()
        )

        '''
        if changing the size from 84 x 84 compute the CNN output size dynamically
        with torch.no_grad():
            dummy_input = torch.zeros(1, in_channels, input_height, input_width)
            cnn_out_dim = self.cnn(dummy_input).shape[1]
        
        self.fc = nn.Sequential(
            nn.Linear(cnn_out_dim, 512),
            nn.ReLU()
        )
        '''

        # Discrete action head (Q-values)
        self.q_head = nn.Linear(512, action_dim)

    # forward pass
    def forward(self, x):
        x = self.fc(self.cnn(x))
        q_values = self.q_head(x)
        return q_values