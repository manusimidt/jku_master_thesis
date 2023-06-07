import random
import torch
import time

import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from env import VanillaEnv, TRAIN_CONFIGURATIONS, generate_expert_trajectory

from policy import ActorNet
import torch.nn.functional as F
import psm


@torch.enable_grad()
def train(Mx: VanillaEnv, My: VanillaEnv, net, optim, alpha, beta, inv_temp, psm_func, loss_bc):
    device = next(net.parameters()).device
    net.train()

    statesX, actionsX = generate_expert_trajectory(Mx)
    statesY, actionsY = generate_expert_trajectory(My)

    statesX, actionsX = torch.tensor(statesX).to(device), torch.tensor(actionsX)
    statesY, actionsY = torch.tensor(statesY).to(device), torch.tensor(actionsY)

    # calculate psm
    psm_metric = torch.tensor(psm_func(actionsX, actionsY)).to(device)
    psm_metric = torch.exp(-psm_metric / beta)
    loss = 0
    avg_pos_sim, avg_neg_sim = 0, 0
    # loop over each state x
    for state_idx in range(statesY.shape[0]):
        # best_match = np.argmax(psm[state_idx])
        current_psm_values = psm_metric[:, state_idx]
        best_match = torch.argmax(current_psm_values)

        target_y = statesY[state_idx]  # this is y
        positive_x = statesX[best_match]  # this is x_y
        negative_x = torch.cat((statesX[:best_match], statesX[best_match + 1:]), dim=0)  # this are all x without x_y

        # pass the positive pairs through the network
        # z_\theta(x_y), z_\theta(y)
        positive_x_logits, target_logits = net.forward(torch.stack((positive_x, target_y)), contrastive=True)
        negative_x_logits = net.forward(negative_x, contrastive=True)  # z_\theta(x')

        # this is s_\theta(x_y, y)
        positive_sim = F.cosine_similarity(positive_x_logits, target_logits, dim=0)
        avg_pos_sim += positive_sim
        nominator = current_psm_values[best_match] * torch.exp(inv_temp * positive_sim)

        # s_\theta(x', y)
        negative_sim = F.cosine_similarity(negative_x_logits, target_logits, dim=1)
        avg_neg_sim += torch.mean(negative_sim)
        psm_metric_negative = torch.cat((current_psm_values[:best_match], current_psm_values[best_match + 1:]), dim=0)
        sum_term = torch.sum(
            (1 - psm_metric_negative) * torch.exp(inv_temp * negative_sim))

        loss += -torch.log(nominator / (nominator + sum_term))

    contrastive_loss = (loss / statesY.shape[0])

    # Add BC loss
    # states_y_logits = net.forward(statesY, contrastive=False)
    # cross_entropy_loss = loss_bc(states_y_logits, actionsY.to(device).to(torch.int64))

    total_loss = alpha * contrastive_loss # + cross_entropy_loss
    optim.zero_grad()
    total_loss.backward()
    optim.step()
    avg_pos_sim = (avg_pos_sim / statesY.shape[0]).item()
    avg_neg_sim = (avg_neg_sim / statesY.shape[0]).item()
    # return total_loss.item(), contrastive_loss.item(), cross_entropy_loss.item(), avg_pos_sim, avg_neg_sim
    return total_loss.item(), contrastive_loss.item(), 0, avg_pos_sim, avg_neg_sim


def dict2mdtable(d, key='Name', val='Value'):
    rows = [f'| {key} | {val} |']
    rows += ['|--|--|']
    rows += [f'| {k} | {v} |' for k, v in d.items()]
    return "  \n".join(rows)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Training on ", device)

    hyperparams = {
        "K": 5000,
        "alpha": 4,  # alignment loss scaling
        "beta": 0.1,  # PSM scaling
        "lambda": 2,  # inverse temperature
        "psm": "paper",
        "conf": "narrow_grid",
        "script": "train_combined.py"
    }

    configurations = TRAIN_CONFIGURATIONS[hyperparams["conf"]]
    training_MDPs = []
    for conf in configurations:
        training_MDPs.append(VanillaEnv([conf]))

    psm_functions = {"fb": psm.psm_fb, "paper": psm.psm_paper}
    psm_func = psm_functions[hyperparams["psm"]]
    net = ActorNet().to(device)

    optim = optim.Adam(net.parameters(), lr=.004)
    loss_bc = nn.CrossEntropyLoss()

    tb = SummaryWriter()
    tb.add_text('info/args', dict2mdtable(hyperparams))

    for i in range(hyperparams['K']):
        # Sample a pair of training MDPs
        Mx, My = random.sample(training_MDPs, 2)
        # todo only for debugging!
        # Mx, My = training_MDPs[0], training_MDPs[1]
        info = train(Mx, My, net, optim, hyperparams['alpha'], hyperparams['beta'], hyperparams['lambda'], psm_func,
                     loss_bc)

        total_err, contrastive_err, cross_entropy_err, avg_pos_sim, avg_neg_sim = info
        print(f"Iteration {i}. Loss: {total_err:2.3f}")

        tb.add_scalar("loss/total", total_err, i)
        tb.add_scalar("loss/contrastive", contrastive_err, i)
        tb.add_scalar("loss/bc", cross_entropy_err, i)
        tb.add_scalar("debug/positive_similarity", avg_pos_sim, i)
        tb.add_scalar("debug/negative_similarity", avg_neg_sim, i)

    state = {
        'state_dict': net.state_dict(),
        'optimizer': optim.state_dict(),
        'info': {'conf': configurations}
    }
    torch.save(state, 'ckpts/' + tb.log_dir.split('\\')[-1] + ".pth")
