import random
import torch
import time
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import env
from common import dict2mdtable, set_seed
from env import VanillaEnv, TRAIN_CONFIGURATIONS, generate_expert_episode

from policy import ActorNet
import torch.nn.functional as F
import psm
from validate import validate


@torch.enable_grad()
def train(Mx: VanillaEnv, My: VanillaEnv, net, optim, alpha, beta, inv_temp, psm_func, bc_data, loss_bc):
    device = next(net.parameters()).device
    net.train()

    statesX, actionsX = generate_expert_episode(Mx)
    statesY, actionsY = generate_expert_episode(My)

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
    # idx = random.sample(range(len(bc_data[0])), 64)
    # bc_states, bc_actions = torch.tensor(bc_data[0][idx]).to(device), torch.tensor(bc_data[1][idx]).to(device)
    # states_y_logits = net.forward(bc_states, contrastive=False)
    # cross_entropy_loss = loss_bc(states_y_logits, bc_actions.to(torch.int64))

    total_loss = alpha * contrastive_loss # + cross_entropy_loss
    optim.zero_grad()
    total_loss.backward()
    optim.step()
    avg_pos_sim = (avg_pos_sim / statesY.shape[0]).item()
    avg_neg_sim = (avg_neg_sim / statesY.shape[0]).item()
    # return total_loss.item(), contrastive_loss.item(), cross_entropy_loss.item(), avg_pos_sim, avg_neg_sim
    return total_loss.item(), contrastive_loss.item(), 0, avg_pos_sim, avg_neg_sim


if __name__ == '__main__':
    seed = 31
    set_seed(seed, env=None)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Training on ", device)

    hyperparams = {
        "K": 3_000,
        "lr": 0.0026,
        "alpha": .1,  # alignment loss scaling
        "beta": 1.0,  # PSM scaling
        "lambda": 1.0,  # inverse temperature
        "psm": "fb",
        "conf": "wide_grid",
        "script": "train.py",
        "info": "Trained without BC!"
    }

    configurations = TRAIN_CONFIGURATIONS[hyperparams["conf"]]
    training_MDPs = []
    for conf in configurations:
        training_MDPs.append(env.AugmentingEnv([conf]))

    psm_functions = {"f": psm.psm_f_fast, "fb": psm.psm_fb_fast}
    psm_func = psm_functions[hyperparams["psm"]]
    net = ActorNet().to(device)

    optim = optim.Adam(net.parameters(), lr=hyperparams['lr'])
    loss_bc = nn.CrossEntropyLoss()

    tb = SummaryWriter()
    tb.add_text('info/args', dict2mdtable(hyperparams))

    bc_data = env.generate_bc_data(training_MDPs, 4096 * 2, balanced=True)

    for i in range(hyperparams['K']):
        # Sample a pair of training MDPs
        Mx, My = random.sample(training_MDPs, 2)
        # todo only for debugging!
        # Mx, My = training_MDPs[0], training_MDPs[1]
        info = train(Mx, My, net, optim, hyperparams['alpha'], hyperparams['beta'], hyperparams['lambda'], psm_func,
                     bc_data, loss_bc)

        total_err, contrastive_err, cross_entropy_err, avg_pos_sim, avg_neg_sim = info
        print(f"Iteration {i}. Loss: {total_err:2.3f}")

        tb.add_scalar("loss/total", total_err, i)
        tb.add_scalar("loss/contrastive", contrastive_err, i)
        tb.add_scalar("loss/bc", cross_entropy_err, i)
        tb.add_scalar("debug/positive_similarity", avg_pos_sim, i)
        tb.add_scalar("debug/negative_similarity", avg_neg_sim, i)

        # if i % 1000 == 0:
        #     fig, perf = validate(net, device, configurations)
        #     tb.add_scalar("val/generalization", perf, i)
        #     tb.add_image("val/fig", fig, i, dataformats="HWC")
    state = {
        'state_dict': net.state_dict(),
        'optimizer': optim.state_dict(),
        'info': {'conf': configurations}
    }
    torch.save(state, 'ckpts/' + tb.log_dir.split('\\')[-1] + ".pth")
