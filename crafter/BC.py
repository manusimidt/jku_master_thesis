import numpy as np
import os
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from env import VanillaEnv
from rl.ppo.policies import ActorCriticNet
from rl.ppo.ppo import PPO


class BCDataset(Dataset):
    def __init__(self, x, y):
        super().__init__()
        assert x.shape[0] == y.shape[0]
        self.x = x
        self.y = y

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]


def load_data(dataset_dir='./dataset/') -> tuple:
    """
    Loads the data and returns two torch.dataloader
    """
    states = actions = None

    for file in os.listdir(dataset_dir):
        with np.load(dataset_dir + file) as data:
            normalized_states = np.array(np.moveaxis(data['image'], -1, -3) / 255, dtype=np.float32)
            if states is None:
                states = normalized_states
                actions = data['action']
            else:
                states = np.concatenate([states, normalized_states], axis=0)
                actions = np.concatenate([actions, data['action']], axis=0)

    X = torch.tensor(states)
    y = torch.tensor(actions)
    data = BCDataset(X, y)
    train_set_length = int(len(data) * 0.8)
    train_set, val_set = torch.utils.data.random_split(data, [train_set_length, len(data) - train_set_length])
    train_loader: DataLoader = DataLoader(train_set, batch_size=64, shuffle=True)
    test_loader: DataLoader = DataLoader(val_set, batch_size=64, shuffle=True)
    return train_loader, test_loader


@torch.enable_grad()
def train(net: ActorCriticNet, dataLoader: DataLoader, optim_actor,
          loss_actor=nn.CrossEntropyLoss()) -> tuple:
    device = next(net.parameters()).device
    net.train()
    actor_errors = []

    for batch in dataLoader:
        # X is the observation
        # y contains the choosen action and the return estimate from the critic
        X, y = batch[0].to(device), batch[1].to(device)
        target_actions = y

        pred_action_logits = net.actor.forward(X)

        actor_error = loss_actor(pred_action_logits, target_actions.to(torch.int64))

        optim_actor.zero_grad()
        actor_error.backward()
        optim_actor.step()

        actor_errors.append(actor_error.item())

    return actor_errors


@torch.no_grad()
def evaluate(net: ActorCriticNet, data_loader: DataLoader, loss_actor) -> tuple:
    device = next(net.parameters()).device
    net.eval()
    actor_errors = []

    for batch in data_loader:
        X, target_actions = batch[0].to(device), batch[1].to(device)

        pred_action_logits = net.actor.forward(X)
        actor_errors.append(loss_actor(pred_action_logits, target_actions.to(torch.int64)).item())

    return actor_errors


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Training on ", device)

lr = 0.001
n_episodes = 10_000
n_epochs = 25

env = VanillaEnv()
trainloader, testloader = load_data()

model = ActorCriticNet(obs_space=(3, 64, 64), action_space=env.num_actions, hidden_size=256).to(device)
optim_actor = optim.Adam(model.actor.parameters(), lr=lr)
loss_actor = nn.CrossEntropyLoss()

for epoch in range(n_epochs):
    actor_errors = train(model, trainloader, optim_actor, loss_actor)
    val_actor_errors = evaluate(model, testloader, loss_actor)

    print(
        f"""Epoch {epoch} Train errors: Actor {np.mean(actor_errors):.3f}, Val errors: Actor {np.mean(val_actor_errors):.3f} """)

optimizer = optim.Adam(model.parameters(), lr=0.001)
ppo = PPO(model, env, optimizer)

ppo.save('./ckpts', 'BC test')
