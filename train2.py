import random
import torch.nn as nn
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import torch.optim as optim
from env import VanillaEnv
from typing import List

from policy import ActorNet
from rl.common.buffer2 import Episode, Transition
import torch.nn.functional as F


def generate_expert_trajectory(env):
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
    return np.array(states), np.array(actions)


def psm_tot(x_arr, y_arr, gamma=0.99):
    storage = np.full(shape=(len(x_arr), len(y_arr)), fill_value=-1.0)

    def psm_dyn(x_idx, y_idx):
        tv = 0. if x_arr[x_idx] == y_arr[y_idx] else 1.
        if x_idx == len(x_arr) - 1 and y_idx == len(y_arr) - 1:
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
def train(Mx: VanillaEnv, My: VanillaEnv, net, optim) -> float:
    device = next(net.parameters()).device
    net.train()

    statesX, actionsX = generate_expert_trajectory(Mx)
    statesY, actionsY = generate_expert_trajectory(My)

    statesX, actionsX = torch.tensor(statesX).to(device), torch.tensor(actionsX)
    statesY, actionsY = torch.tensor(statesY).to(device), torch.tensor(actionsY)

    # calculate psm
    psm = torch.tensor(psm_tot(actionsY, actionsX)).to(device)
    psm_metric = torch.exp(-psm / beta)
    loss = 0
    # loop over each state x
    for state_idx in range(statesY.shape[0]):
        # best_match = np.argmax(psm[state_idx])
        best_match = torch.argmax(psm_metric[state_idx])

        target_y = statesY[state_idx] # this is y
        positive_x = statesX[best_match] # this is x_y
        negative_x = torch.cat((statesX[:best_match], statesX[best_match + 1:]), dim=0) # this are all x without x_y

        # pass the positive pairs through the network
        positive_x_logits, target_logits = net.forward(torch.stack((target_y, positive_x)), contrastive=True)
        negative_x_logits = net.forward(negative_x, contrastive=True)

        # this is s_\theta(x_y, y)
        positive_sim = F.cosine_similarity(positive_x_logits, target_logits, dim=0)
        nominator = psm_metric[state_idx, best_match] * torch.exp(inv_temp * positive_sim)

        negative_sim = F.cosine_similarity(negative_x_logits, target_logits, dim=1)
        # psm_metric_negative = np.delete(psm_metric.cpu().numpy(), best_match.cpu().item(), axis=0)
        psm_metric_negative = torch.cat((psm_metric[:best_match], psm_metric[best_match + 1:]), dim=0)

        sum_term = torch.sum(
            (1 - psm_metric_negative[:, state_idx]) * torch.exp(inv_temp * negative_sim))

        loss += -torch.log(nominator / (nominator + sum_term))
        

    total_loss = loss/statesY.shape[0]
    optim.zero_grad()
    total_loss.backward()
    optim.step()
    print(f"Loss episode: {total_loss.item():.4f}")
    return total_loss.item()


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Training on ", device)

    training_MDPs = [
        VanillaEnv(configurations=[(26, 12)]), VanillaEnv(configurations=[(29, 12)]),
        VanillaEnv(configurations=[(34, 12)]), VanillaEnv(configurations=[(26, 20)]),
        VanillaEnv(configurations=[(34, 20)]), VanillaEnv(configurations=[(26, 28)]),
        VanillaEnv(configurations=[(29, 28)]), VanillaEnv(configurations=[(34, 28)]),
    ]
    K = 500
    beta = 0.01
    inv_temp = 1.0  # lambda
    net = ActorNet().to(device)
    optim = optim.Adam(net.parameters(), lr=.01)
    total_errors = []
    for i in range(K):
        # Sample a pair of training MDPs
        Mx, My = random.sample(training_MDPs, 2)
        error = train(Mx, My, net, optim)
        print(f"Epoch {i}. Loss: {error:.3f} convX:{Mx.configurations}, convY:{My.configurations}")
        total_errors.append(error)

    xpoints = np.arange(len(total_errors))
    ypoints = np.array(total_errors)

    plt.plot(xpoints, ypoints)
    plt.title("Loss over training iterations")
    plt.show()
