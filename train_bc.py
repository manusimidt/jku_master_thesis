import numpy as np
from env import VanillaEnv, TRAIN_CONFIGURATIONS
from policy import ActorNet
from rl.common.buffer2 import Transition, Episode
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import List
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


def generate_expert_trajectories(env, n_episodes):
    gamma = 0.99

    episodes: List[Episode] = []

    for i in range(n_episodes):
        done = False
        episode = Episode(discount=gamma)
        obs = env.reset()
        obstacle_position = env.actualEnv.obstacle_position
        jumping_pixel = obstacle_position - 14
        step = 0
        while not done:
            action = 0 if step < jumping_pixel else 1
            next_obs, reward, done, _ = env.step(action)
            episode.append(Transition(obs, action, reward, 0))
            obs = next_obs
            env.render()
            step += 1

        episodes.append(episode)

    # get the states, returns and actions out of the episodes
    states, actions, values = [], [], []
    for episode in episodes:
        states += episode.states()
        actions += episode.actions()
        values += episode.calculate_return()

    states, actions, values = np.array(states), np.array(actions), np.array(values)
    # vertically add actions and values
    targets = np.column_stack((actions, values))

    X = torch.tensor(states)
    Y = torch.tensor(targets)
    data: BCDataset = BCDataset(X, Y)
    train_set_length = int(len(data) * 0.8)
    train_set, val_set = torch.utils.data.random_split(data, [train_set_length, len(data) - train_set_length])

    trainloader: DataLoader = DataLoader(train_set, batch_size=64, shuffle=True)
    testloader: DataLoader = DataLoader(val_set, batch_size=64, shuffle=True)
    return trainloader, testloader


@torch.enable_grad()
def train(net: ActorNet, dataLoader: DataLoader, optim_actor, loss_actor=nn.CrossEntropyLoss()):
    device = next(net.parameters()).device
    net.train()
    actor_errors = []

    for batch in dataLoader:
        # X is the observation
        # y contains the choosen action and the return estimate from the critic
        X, y = batch[0].to(device), batch[1].to(device)
        target_actions = y[:, 0]

        pred_action_logits = net.forward(X, contrastive=False)

        actor_error = loss_actor(pred_action_logits, target_actions.to(torch.int64))

        optim_actor.zero_grad()
        actor_error.backward()
        optim_actor.step()

        actor_errors.append(actor_error.item())

    return actor_errors


@torch.no_grad()
def evaluate(net: ActorNet, dataLoader: DataLoader, loss_actor):
    device = next(net.parameters()).device
    net.eval()
    actor_errors = []

    for batch in dataLoader:
        X, y = batch[0].to(device), batch[1].to(device)
        target_actions = y[:, 0]
        pred_action_logits = net.forward(X, contrastive=False)
        actor_errors.append(loss_actor(pred_action_logits, target_actions.to(torch.int64)).item())

    return actor_errors


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Training on ", device)

    lr = 0.001
    n_episodes = 5_000
    n_epochs = 8

    environment = 'vanilla_env'
    base_model = './ckpts/loss_my_psm_fb_yx.pth'

    model = ActorNet().to(device)
    if len(base_model) > 0:
        print("Loading model ", base_model)
        ckp = torch.load(base_model, map_location=device)
        conf = ckp['info']['conf']

        model.load_state_dict(ckp['state_dict'])
        model.disable_embedding_weights()
        optim_actor = optim.Adam(model.parameters(), lr=lr)

    else:
        print("Training BC model without any contrastive learning")
        optim_actor = optim.Adam(model.parameters(), lr=lr)
        base_model = './ckpts/pure_bc_without_pse-narrow_grid.pth'
        conf = TRAIN_CONFIGURATIONS['narrow_grid']

    env = VanillaEnv(configurations=list(conf))

    print(f"Generating {n_episodes} expert episodes...")
    train_loader, test_loader = generate_expert_trajectories(env, n_episodes)

    loss_actor = nn.CrossEntropyLoss()

    for epoch in range(n_epochs):
        actor_errors = train(model, train_loader, optim_actor, loss_actor)
        val_actor_errors = evaluate(model, test_loader, loss_actor)

        print(
            f"""Epoch {epoch} Train errors: Actor {np.mean(actor_errors):.8f}, Val errors: Actor {np.mean(val_actor_errors):.8f} """)

    state = {
        'state_dict': model.state_dict(),
        'optimizer': optim_actor.state_dict(),
        'info': {'conf': conf}
    }
    torch.save(state, base_model.replace('.pth', '-bc.pth'))
