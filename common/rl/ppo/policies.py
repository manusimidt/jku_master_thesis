import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical



class ActorNet(nn.Module):
    def __init__(self, obs_space: tuple, action_space: int, hidden_size: int):
        """
        :param obs_space: 3-dimensional tuple (CxHxW)
        :param action_space: size of the action space
        :param hidden_size: size of the hidden space
        """
        super(ActorNet, self).__init__()
        C, H, W = obs_space
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=C, out_channels=16, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Flatten()
        )
        # compute the feature shape by doing one forward pass
        with torch.no_grad():
            feature_shape = self.features(torch.rand(size=(1, C, H, W))).shape[1]

        self.fc = nn.Sequential(
            nn.Linear(feature_shape, hidden_size),
            nn.LayerNorm(hidden_size, elementwise_affine=False),
            nn.ReLU(),
            nn.Linear(hidden_size, action_space)
        )

    def forward(self, x):
        # make sure the data is a tensor on the correct device
        device = next(self.parameters()).device
        x = x.to(device, dtype=torch.float32)

        x = self.features(x)
        # convert the images to a matrix with the batch count as first dimension and the features as second dimension
        # x = x.view(x.size(0), -1)
        x = self.fc(x)
        # return torch.softmax(x, dim=-1) #-1 to take softmax of last dimension
        return x


class CriticNet(nn.Module):

    def __init__(self, obs_space: tuple, hidden_size: int):
        super(CriticNet, self).__init__()
        C, H, W = obs_space
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=C, out_channels=8, kernel_size=3, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Flatten()
        )
        # compute the feature shape by doing one forward pass
        with torch.no_grad():
            feature_shape = self.features(torch.rand(size=(1, C, H, W))).shape[1]

        self.fc = nn.Sequential(
            nn.Linear(feature_shape, hidden_size),
            nn.LayerNorm(hidden_size, elementwise_affine=False),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        # make sure the data is a tensor on the correct device
        device = next(self.parameters()).device
        x = x.to(device, dtype=torch.float32)

        x = self.features(x)
        # x = x.view(x.size(0), -1)
        return self.fc(x)


class ActorCriticNet(nn.Module):
    def __init__(self, obs_space, action_space, hidden_size):
        super(ActorCriticNet, self).__init__()
        self.actor = ActorNet(obs_space, action_space, hidden_size)
        self.critic = CriticNet(obs_space, hidden_size)

    def forward(self, x):
        raise NotImplementedError

    def act(self, state):
        action_logits = self.actor(state)
        dist = Categorical(logits=action_logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob

    def act_deterministic(self, state):
        action_logits = self.actor(state)
        action = torch.argmax(action_logits, dim=1)
        log_prob = torch.log(F.softmax(action_logits, dim=0))[:, action].squeeze()
        return action, log_prob

    def evaluate(self, state, action):
        action_logits = self.actor(state)
        policy_dist = Categorical(logits=action_logits)
        log_probs = policy_dist.log_prob(action)
        entropy = policy_dist.entropy()
        state_value = self.critic(state)
        return state_value, log_probs, entropy

