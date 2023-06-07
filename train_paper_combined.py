import random
import torch
import time
import torch.nn as nn

import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import psm
from env import VanillaEnv, TRAIN_CONFIGURATIONS, generate_expert_trajectory

from policy import ActorNet


def contrastive_loss(similarity_matrix, metric_values, temperature=1.0, beta=1.0):
    """
    Contrastive Loss with embedding similarity.
    Taken from Agarwal.et.al. rewritten in pytorch
    """
    # z_\theta(X): embedding_1 = nn_model.representation(X)
    # z_\theta(Y): embedding_2 = nn_model.representation(Y)
    # similarity_matrix = cosine_similarity(embedding_1, embedding_2
    # metric_values = PSM(X, Y)
    metric_shape = metric_values.size()
    similarity_matrix /= temperature
    neg_logits1 = similarity_matrix

    col_indices = torch.argmin(metric_values, dim=1)
    pos_indices1 = torch.stack(
        (torch.arange(metric_shape[0], dtype=torch.int32, device=device), col_indices), dim=1)
    pos_logits1 = similarity_matrix[pos_indices1[:, 0], pos_indices1[:, 1]]

    metric_values /= beta
    similarity_measure = torch.exp(-metric_values)
    pos_weights1 = -metric_values[pos_indices1[:, 0], pos_indices1[:, 1]]
    pos_logits1 += pos_weights1
    negative_weights = torch.log((1.0 - similarity_measure) + 1e-8)
    negative_weights[pos_indices1[:, 0], pos_indices1[:, 1]] = pos_weights1

    neg_logits1 += negative_weights

    neg_logits1 = torch.logsumexp(neg_logits1, dim=1)
    return torch.mean(neg_logits1 - pos_logits1)  # Equation 4


def cosine_similarity(a, b, eps=1e-8):
    """
    Computes cosine similarity between all pairs of vectors in x and y
    added eps for numerical stability
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt


@torch.enable_grad()
def train(Mx: VanillaEnv, My: VanillaEnv, net, optim, alpha, beta, inv_temp, psm_func, loss_bc):
    device = next(net.parameters()).device
    net.train()

    statesX, actionsX = generate_expert_trajectory(Mx)
    statesY, actionsY = generate_expert_trajectory(My)

    statesX, actionsX = torch.tensor(statesX).to(device), torch.tensor(actionsX)
    statesY, actionsY = torch.tensor(statesY).to(device), torch.tensor(actionsY)

    embedding_1 = net.forward(statesX, contrastive=True)  # z_theta(x)
    embedding_2 = net.forward(statesY, contrastive=True)  # z_theta(y)
    similarity_matrix = cosine_similarity(embedding_1, embedding_2)

    metric_values = torch.tensor(psm_func(actionsX, actionsY)).to(device)
    alignment_loss = contrastive_loss(similarity_matrix, metric_values, inv_temp, beta)

    # states_y_logits = net.forward(statesY, contrastive=False)
    # actionsY = actionsY.to(device).to(torch.int64)
    # # todo in the paper they have use other data for training bc (256 x 60 x 60 x 2) They probably do this because
    # #  they have a function that up-samples the training examples where the action has to jump
    # cross_entropy_loss = loss_bc(states_y_logits, actionsY)

    total_loss = alpha * alignment_loss  # + cross_entropy_loss

    optim.zero_grad()
    total_loss.backward()
    optim.step()
    return total_loss.item(), alignment_loss.item(), 0  # cross_entropy_loss.item()


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Training on ", device)

    hyperparams = {
        "K": 150000,
        "lr": 0.0026,
        "alpha": 4,  # alignment loss scaling
        "beta": 0.1,  # PSM scaling
        "lambda": 1,  # inverse temperature
        "psm": "paper",
        "conf": "narrow_grid",
        "script": "train_paper_combined.py"
    }

    psm_functions = {"fb": psm.psm_fb, "paper": psm.psm_paper}
    psm_func = psm_functions[hyperparams["psm"]]
    configurations = TRAIN_CONFIGURATIONS[hyperparams["conf"]]
    training_MDPs = []
    for conf in configurations:
        training_MDPs.append(VanillaEnv([conf]))

    net = ActorNet().to(device)

    optim = optim.Adam(net.parameters(), lr=hyperparams['lr'])
    loss_bc = nn.CrossEntropyLoss()
    start = time.time()
    tb = SummaryWriter()

    for i in range(hyperparams['K']):
        # Sample a pair of training MDPs
        Mx, My = random.sample(training_MDPs, 2)
        info = train(Mx, My, net, optim, hyperparams['alpha'], hyperparams['beta'], hyperparams['lambda'], psm_func,
                     loss_bc)

        total_err, contrastive_err, cross_entropy_err = info
        print(f"Iteration {i}. Loss: {total_err:.3f}")

        tb.add_scalar("loss/total", total_err, i)
        tb.add_scalar("loss/contrastive", contrastive_err, i)
        tb.add_scalar("loss/bc", cross_entropy_err, i)

    state = {
        'state_dict': net.state_dict(),
        'optimizer': optim.state_dict(),
        'info': {'conf': configurations}
    }
    torch.save(state, 'ckpts/' + tb.log_dir.split('\\')[-1] + ".pth")
