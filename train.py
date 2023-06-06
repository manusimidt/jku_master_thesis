import random
import torch
import time

import torch.optim as optim
from env import VanillaEnv, TRAIN_CONFIGURATIONS, generate_expert_trajectory

from policy import ActorNet
import torch.nn.functional as F
import psm


@torch.enable_grad()
def train(Mx: VanillaEnv, My: VanillaEnv, net, optim) -> float:
    device = next(net.parameters()).device
    net.train()

    statesX, actionsX = generate_expert_trajectory(Mx)
    statesY, actionsY = generate_expert_trajectory(My)

    statesX, actionsX = torch.tensor(statesX).to(device), torch.tensor(actionsX)
    statesY, actionsY = torch.tensor(statesY).to(device), torch.tensor(actionsY)

    # calculate psm
    psm_metric = torch.tensor(psm.psm_fb(actionsY, actionsX)).to(device)
    psm_metric = torch.exp(-psm_metric / beta)
    loss = 0
    # loop over each state x
    for state_idx in range(statesY.shape[0]):
        # best_match = np.argmax(psm[state_idx])
        best_match = torch.argmax(psm_metric[state_idx])

        target_y = statesY[state_idx]  # this is y
        positive_x = statesX[best_match]  # this is x_y
        negative_x = torch.cat((statesX[:best_match], statesX[best_match + 1:]), dim=0)  # this are all x without x_y

        # pass the positive pairs through the network
        # z_\theta(x_y), z_\theta(y)
        positive_x_logits, target_logits = net.forward(torch.stack((target_y, positive_x)), contrastive=True)
        negative_x_logits = net.forward(negative_x, contrastive=True)  # z_\theta(x')

        # this is s_\theta(x_y, y)
        positive_sim = F.cosine_similarity(positive_x_logits, target_logits, dim=0)
        nominator = psm_metric[state_idx, best_match] * torch.exp(inv_temp * positive_sim)

        # s_\theta(x', y)
        negative_sim = F.cosine_similarity(negative_x_logits, target_logits, dim=1)
        psm_metric_negative = torch.cat((psm_metric[:best_match], psm_metric[best_match + 1:]), dim=0)

        sum_term = torch.sum(
            (1 - psm_metric_negative[:, state_idx]) * torch.exp(inv_temp * negative_sim))

        loss += -torch.log(nominator / (nominator + sum_term))

    total_loss = loss / statesY.shape[0]
    optim.zero_grad()
    total_loss.backward()
    optim.step()
    return total_loss.item()


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Training on ", device)

    configurations = TRAIN_CONFIGURATIONS['narrow_grid']
    training_MDPs = []
    for conf in configurations:
        training_MDPs.append(VanillaEnv([conf]))
    K = 5000
    beta = 0.1
    inv_temp = 1 / 0.1  # lambda
    net = ActorNet().to(device)

    optim = optim.Adam(net.parameters(), lr=.001)
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
    xpoints = np.arange(len(total_errors))
    ypoints = np.array(total_errors)

    plt.plot(xpoints, ypoints)
    plt.title("Loss over training iterations")
    plt.show()

    state = {
        'state_dict': net.state_dict(),
        'optimizer': optim.state_dict(),
        'info': {'conf': configurations}
    }
    torch.save(state, 'ckpts/loss_my_psm_fb_yx.pth')
