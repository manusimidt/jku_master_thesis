import random

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from env import VanillaEnv
from typing import List

from policy import ActorNet
from rl.common.buffer2 import Episode, Transition
import torch.nn.functional as F

class ContrastiveDataset(Dataset):
    def __init__(self, statesX: torch.tensor, actionsX: torch.tensor, statesY: torch.tensor, actionsY: torch.tensor):
        super().__init__()
        assert statesX.shape[0] == actionsX.shape[0] == statesY.shape[0] == actionsY.shape[0]
        self.statesX = statesX
        self.actionsX = actionsX
        self.statesY = statesY
        self.actionsY = actionsY

    def __len__(self):
        return self.statesX.shape[0]

    def __getitem__(self, index):
        return self.statesX[index], self.actionsX[index], self.statesY[index], self.actionsY[index]


def generate_expert_trajectories(env1, env2, n_episodes):
    def generate_episodes(env) -> tuple[np.ndarray, np.ndarray]:
        total_states, total_actions = [], []
        for i in range(n_episodes):
            states, actions = [], []
            done = False
            obs = env.reset()
            obstacle_position = env.actualEnv.obstacle_position
            jumping_pixel = obstacle_position - 14
            step = 0
            while not done:
                action = 0 if step < jumping_pixel else 1
                next_obs, reward, done, _ = env.step(action)
                states.append(obs)
                actions.append(action)
                obs = next_obs
                env.render()
                step += 1
            total_states.append(states)
            total_actions.append(actions)

        return np.array(total_states), np.array(total_actions)

    statesX, actionsX = generate_episodes(env1)
    statesY, actionsY = generate_episodes(env2)

    data: ContrastiveDataset = ContrastiveDataset(torch.tensor(statesX), torch.tensor(actionsX), torch.tensor(statesY),
                                                  torch.tensor(actionsY))
    return DataLoader(data, batch_size=64, shuffle=True)


def psm_tot(x_arr, y_arr, gamma=0.99):
    storage = np.full(shape=(len(x_arr), len(y_arr)), fill_value=-1.0)

    def psm_dyn(x_idx, y_idx):
        tv = 0. if x_arr[x_idx] == y_arr[y_idx] else 1.
        if (x_idx == len(x_arr) - 1 and y_idx == len(y_arr) - 1):
            return tv
        else:
            next_x_idx = min(x_idx + 1, len(x_arr) - 1)
            next_y_idx = min(y_idx + 1, len(y_arr) - 1)
            next_psm = psm_dyn(next_x_idx, next_y_idx) if storage[next_x_idx, next_y_idx] == -1 else storage[
                next_x_idx, next_y_idx]
            return tv + gamma * next_psm

    for i in range(len(x_arr)):
        for j in range(len(y_arr)):
            storage[i, j] = psm_dyn(i, j)
    return storage


@torch.enable_grad()
def train(data, net):
    device = next(net.parameters()).device
    net.train()
    errors = []

    for batch in data:
        statesX, actionsX = batch[0], batch[1]
        statesY, actionsY = batch[2], batch[3]
        print(statesX.shape, actionsX.shape)
        # todo build positive and negative pairs

        # loop over each episode in the batch
        for episode_idx in range(statesX.shape[0]):
            curr_statesX, curr_actionX = statesX[episode_idx], actionsX[episode_idx]
            curr_statesY, curr_actionY = statesY[episode_idx], actionsY[episode_idx]

            # calculate psm
            psm = psm_tot(curr_actionX, curr_actionY)
            psm_metric = np.exp(-psm/beta)
            # loop over each state x
            for state_idx in range(curr_statesX.shape[0]):
                # best_batch = np.argmin(psm[state_idx])
                best_batch = np.argmax(psm_metric[state_idx])

                positive_pair = torch.stack((curr_statesX[state_idx], curr_statesY[best_batch]))
                negative_pairs = np.delete(curr_statesY, best_batch, axis=0)


                # pass the positive pairs through the network
                positive_logits = net.forward(positive_pair, contrastive=True)
                # this is s_\theta(x_y, y)
                s_theta = F.cosine_similarity(positive_logits[0], positive_logits[1], dim=0)

                nominator = psm_metric[state_idx, best_batch] * torch.exp(s_theta)

                

                print(positive_pair, negative_pairs)

if __name__ == '__main__':
    training_MDPs = [
        VanillaEnv(configurations=[(26, 12)]), VanillaEnv(configurations=[(29, 12)]),
        VanillaEnv(configurations=[(34, 12)]), VanillaEnv(configurations=[(26, 20)]),
        VanillaEnv(configurations=[(34, 20)]), VanillaEnv(configurations=[(26, 28)]),
        VanillaEnv(configurations=[(29, 28)]), VanillaEnv(configurations=[(34, 28)]),
    ]
    K = 40
    beta = .5
    net = ActorNet()

    for i in range(K):
        # Sample a pair of training MDPs
        Mx, My = random.sample(training_MDPs, 2)

        data = generate_expert_trajectories(Mx, My, 1_000)

        train(data, net)
