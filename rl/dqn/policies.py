import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class DQNPolicy(nn.Module):
    def __init__(self, num_actions, obs_dim) -> None:
        super().__init__()
        self.num_actions = num_actions
        self.obs_dim = obs_dim

class CNNDQNNet(DQNPolicy):

    def __init__(self, num_actions:int=2, obs_dim:int = 60) -> None:
        super().__init__(num_actions, obs_dim)

        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3,  stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3,  stride=2),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(32*3*3, 128), 
            nn.ReLU(),
            nn.Linear(128, num_actions)
        )

    def forward(self, x):
        # make sure the data is a tensor on the correct device
        device = next(self.parameters()).device
        x = torch.tensor(x).to(device, dtype=torch.float32)
        # make sure data has always the correct dim (channel X width X height)
        if len(x.shape) == 2: x = x.unsqueeze(dim=0)
        # make sure data is always structured in batches 
        if len(x.shape) == 3: x = x.unsqueeze(dim=1)

        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class FCDQNNet(DQNPolicy):

    def __init__(self, num_actions:int=2, obs_dim:int = 60) -> None:
        super().__init__(num_actions, obs_dim)

        self.fc = nn.Sequential(
            nn.Linear(obs_dim*obs_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def forward(self, x):
        return self.fc(x.flatten())

if __name__ == '__main__':
    obs = torch.rand(size=(1, 1, 60, 60))
    cnn_net = CNNDQNNet()
    fc_net = FCDQNNet()
    print(cnn_net.forward(obs))
    print(fc_net.forward(obs))