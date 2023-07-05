import argparse
import copy
import os
import random
from multiprocessing import Pool

import torch
import time
import torch.nn as nn

import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

import env
import psm
from common import dict2mdtable, set_seed
from env import VanillaEnv, AugmentingEnv, TRAIN_CONFIGURATIONS, generate_expert_episode

from policy import ActorNet
from rl.common.logger import get_date_str
from validate import validate


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
        (torch.arange(metric_shape[0], dtype=torch.int32, device=col_indices.device), col_indices), dim=1)
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
def train(Mx: VanillaEnv, My: VanillaEnv, net, optim, alpha1, alpha2, beta, inv_temp, psm_func, bc_data, loss_bc):
    device = next(net.parameters()).device
    net.train()

    statesX, actionsX = generate_expert_episode(Mx)
    statesY, actionsY = generate_expert_episode(My)

    statesX, actionsX = torch.tensor(statesX).to(device), torch.tensor(actionsX)
    statesY, actionsY = torch.tensor(statesY).to(device), torch.tensor(actionsY)

    representation_1 = net.forward(statesX, contrastive=True)  # z_theta(x)
    representation_2 = net.forward(statesY, contrastive=True)  # z_theta(y)
    similarity_matrix = cosine_similarity(representation_1, representation_2)

    metric_values = torch.tensor(psm_func(actionsX, actionsY)).to(device)
    alignment_loss = contrastive_loss(similarity_matrix, metric_values, inv_temp, beta)

    # todo in the paper they have use other data for training bc (256 x 60 x 60 x 2) They probably do this because
    #  they have a function that up-samples the training examples where the action has to jump
    # states_y_logits = net.forward(statesY, contrastive=False)
    # actionsY = actionsY.to(device).to(torch.int64)
    # cross_entropy_loss = loss_bc(states_y_logits, actionsY)
    idx = random.sample(range(len(bc_data[0])), 256)
    bc_states, bc_actions = torch.tensor(bc_data[0][idx]).to(device), torch.tensor(bc_data[1][idx]).to(device)
    states_y_logits = net.forward(bc_states, contrastive=False)
    cross_entropy_loss = loss_bc(states_y_logits, bc_actions.to(torch.int64))

    total_loss = alpha1 * alignment_loss + alpha2 * cross_entropy_loss

    optim.zero_grad()
    total_loss.backward()
    optim.step()
    return total_loss.item(), alignment_loss.item(), cross_entropy_loss.item()


def main(hyperparams: dict):
    set_seed(hyperparams['seed'], None)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Training on ", device)

    psm_functions = {"f": psm.psm_f_fast, "fb": psm.psm_fb_fast}
    psm_func = psm_functions[hyperparams["psm"]]

    configurations = TRAIN_CONFIGURATIONS[hyperparams["conf"]]
    training_MDPs = []
    for conf in configurations:
        training_MDPs.append(VanillaEnv([conf]) if hyperparams['env'] == 'vanilla' else AugmentingEnv([conf]))

    net = ActorNet().to(device)

    optimizer = optim.Adam(net.parameters(), lr=hyperparams['learning_rate'])
    loss_bc = nn.CrossEntropyLoss()

    tb = SummaryWriter(log_dir=hyperparams['train_dir'] + os.sep + str(hyperparams['seed']))
    tb.add_text('info/args', dict2mdtable(hyperparams))

    bc_data = env.generate_bc_data(training_MDPs, 4096 * 8, balanced=False)

    for i in range(hyperparams['n_iterations']):
        # Sample a pair of training MDPs
        Mx, My = random.sample(training_MDPs, 2)
        info = train(Mx, My, net, optimizer, hyperparams['alpha1'], hyperparams['alpha2'], hyperparams['beta'],
                     hyperparams['lambda'], psm_func,
                     bc_data, loss_bc)

        total_err, contrastive_err, cross_entropy_err = info
        print(f"Iteration {i}. Loss: {total_err:.3f}")

        tb.add_scalar("loss/total", total_err, i)
        tb.add_scalar("loss/contrastive", contrastive_err, i)
        tb.add_scalar("loss/bc", cross_entropy_err, i)

        if i % 1000 == 0:
            fig, perf = validate(net, device, configurations)
            tb.add_scalar("val/generalization", perf, i)
            tb.add_image("val/fig", fig, i, dataformats="HWC")

    state = {
        'state_dict': net.state_dict(),
        'optimizer': optimizer.state_dict(),
        'info': {'conf': configurations}
    }
    torch.save(state, 'ckpts/' + tb.log_dir.split('\\')[-1] + ".pth")


if __name__ == '__main__':
    set_seed(123, None)  # Base seed

    parser = argparse.ArgumentParser()

    parser.add_argument("--train_dir", default=f'./experiments/{get_date_str()}/',
                        help="The directory to store the results from this run into")

    parser.add_argument("-e", "--env", choices=["vanilla", "random"], default="vanilla",
                        help="The environment to train on")
    parser.add_argument("-c", "--conf", choices=["narrow_grid", "wide_grid"], default="wide_grid",
                        help="The environment configuration to train on")

    parser.add_argument("-psm", "--psm", choices=["f", "fb"], default="f",
                        help="The PSM distance function to use (f=forward PSM, fb=forward-backward PSM)")

    parser.add_argument("-bs", "--batch_size", default=256, type=int,
                        help="Size of one Minibatch")
    parser.add_argument("-bc", "--batch_count", default=10, type=int,
                        help="Number of batches. BC will be trained on batch_size x batch_count samples")

    parser.add_argument("--balanced", default=True, type=bool, action=argparse.BooleanOptionalAction,
                        help="If true, the algorithm will be trained on a balanced dataset (1/3 action 1, 2/3 action 0 "
                             "examples)")

    parser.add_argument("-lr", "--learning_rate", default=0.0026, type=float,
                        help="Learning rate for the optimizer")
    parser.add_argument("-K", "--n_iterations", default=80_000, type=int,
                        help="Number of total training steps")

    parser.add_argument("-a1", "--alpha1", default=5., type=float,
                        help="Scaling factor for the alignment loss")

    parser.add_argument("-a2", "--alpha2", default=1., type=float,
                        help="Scaling factor for the BC loss")

    parser.add_argument("-b", "--beta", default=1.0, type=float,
                        help="Scaling factor for the PSM")

    parser.add_argument("-l", "--lambda", default=0.5, type=float,
                        help="Inverse temperature")

    parser.add_argument("-s", "--seed", default=[1], type=int, nargs='+',
                        help="The seed to train on. If multiple values are provided, the script will "
                             "spawn a new process for each seed")

    args = parser.parse_args()
    hyperparams = copy.deepcopy(vars(args))
    if len(args.seed) == 1:
        # convert the list to a single value
        hyperparams['seed'] = hyperparams['seed'][0]
        main(hyperparams)
    else:
        params_arr = []
        for i in range(len(args.seed)):
            curr_hyperparams = copy.deepcopy(hyperparams)
            curr_hyperparams['seed'] = hyperparams['seed'][i]
            params_arr.append((curr_hyperparams,))
        with Pool(4) as p:
            p.starmap(main, params_arr)
