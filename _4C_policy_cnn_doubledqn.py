# DOUBLE DQN AND DUELING ARCHITECTURE

'''
Double DQN
    - fixes Q-value overestimation
    - overestimation bias is significantly reduced
    - almost zero extra computation
    - this is just a training rule change
    - in DQN, the same network chooses action and evaluates its value
    - in Double DQN, online network chooses the action and target network evaluates the chosen action

Dueling Architecture
    - fixes state-action entanglement
    - In many states, the choice of action barely matters, but vanilla DQN still learns a full Q-value per action. This wastes capacity and slows learning.
    - The dueling idea is to decompose Q-values as V(s) -> how good the state is and A(s,a) -> how much better action a is than average
'''

# importing libraries
import torch
import torch.nn as nn

# size
input_height_output, input_width_output = 84, 84

class DuelingNatureCNNPolicy(nn.Module):
    def __init__(self, in_channels, action_dim, input_height=input_height_output, input_width=input_width_output):
        # initialization
        super().__init__()

        # Shared convolutional backbone of Nature CNN
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )

        # Dynamically compute CNN output size
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, input_height, input_width)
            cnn_out_dim = self.cnn(dummy).shape[1]

        # Shared fully connected layer
        self.fc = nn.Sequential(
            nn.Linear(cnn_out_dim, 512),
            nn.ReLU()
        )

        # Value stream V(s)
        self.value_head = nn.Linear(512, 1)

        # Advantage stream A(s, a)
        self.advantage_head = nn.Linear(512, action_dim)

    def forward(self, x):
        x = self.cnn(x)
        x = self.fc(x)

        value = self.value_head(x)                # (B, 1)
        advantage = self.advantage_head(x)        # (B, A)

        # Combine using mean-normalized advantage
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q_values # (B, action_dim)
    

'''
TRAINING STEP

# Online network
q_online_next = online_net(next_states)
next_actions = torch.argmax(q_online_next, dim=1)

# Target network
q_target_next = target_net(next_states)
q_next = q_target_next.gather(1, next_actions.unsqueeze(1)).squeeze(1)

# Double DQN target
q_target = rewards + gamma * (1 - dones) * q_next
'''
