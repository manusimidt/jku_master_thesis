import random
import torch
import time

import torch.optim as optim

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
def train(Mx: VanillaEnv, My: VanillaEnv, net, optim) -> float:
    device = next(net.parameters()).device
    net.train()

    statesX, actionsX = generate_expert_trajectory(Mx)
    statesY, actionsY = generate_expert_trajectory(My)

    statesX, actionsX = torch.tensor(statesX).to(device), torch.tensor(actionsX)
    statesY, actionsY = torch.tensor(statesY).to(device), torch.tensor(actionsY)

    embedding_1 = net.forward(statesX, contrastive=True)  # z_theta(x)
    embedding_2 = net.forward(statesY, contrastive=True)  # z_theta(y)
    similarity_matrix = cosine_similarity(embedding_1, embedding_2)

    metric_values = torch.tensor(psm.psm_paper(actionsY, actionsX)).to(device)  # maybe actionsY, actionsX must be switched! (Nope, does not work better)
    loss = contrastive_loss(similarity_matrix, metric_values, temperature)

    optim.zero_grad()
    loss.backward()
    optim.step()
    return loss.item()


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Training on ", device)

    configurations = TRAIN_CONFIGURATIONS['wide_grid']
    training_MDPs = []
    for conf in configurations:
        training_MDPs.append(VanillaEnv([conf]))
    K = 30_000
    beta = 0.01
    temperature = 0.5
    learning_rate = 0.0026
    net = ActorNet().to(device)

    optim = optim.Adam(net.parameters(), lr=learning_rate)
    total_errors = []
    start = time.time()
    for i in range(K):
        # Sample a pair of training MDPs
        Mx, My = random.sample(training_MDPs, 2)
        error = train(Mx, My, net, optim)
        print(f"Iteration {i}. Loss: {error:.3f} convX:{Mx.configurations}, convY:{My.configurations}")
        total_errors.append(error)
    end = time.time()
    print(f"Elapsed time {end - start}")

    state = {
        'state_dict': net.state_dict(),
        'optimizer': optim.state_dict(),
        'info': {'conf': configurations}
    }
    torch.save(state, 'ckpts/train_paper-wide_grid.pth')

    plt.plot(np.arange(len(total_errors)), np.array(total_errors))
    plt.title("Loss over training iterations")
    plt.show()
