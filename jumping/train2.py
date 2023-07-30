import argparse
import copy
import os
from multiprocessing import Pool

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from common import dict2mdtable, set_seed
from env import TRAIN_CONFIGURATIONS, JumpingExpertBuffer

from policy import ActorNet
import torch.nn.functional as F
import common.psm as psm
from rl.common.logger import get_date_str
from validate import validate, generate_image


@torch.enable_grad()
def train(net, optim, alpha1, alpha2, beta, inv_temp, psm_func, buffer, loss_bc, batch_size):
    net.train()

    statesX, actionsX = buffer.sample_trajectory()
    statesY, actionsY = buffer.sample_trajectory()

    # calculate psm
    psm_metric = psm_func(actionsX, actionsY)
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
    bc_states, bc_actions = buffer.sample(batch_size)
    states_y_logits = net.forward(bc_states, contrastive=False)
    cross_entropy_loss = loss_bc(states_y_logits, bc_actions.to(torch.int64))

    total_loss = alpha1 * contrastive_loss + alpha2 * cross_entropy_loss
    optim.zero_grad()
    total_loss.backward()
    optim.step()
    avg_pos_sim = (avg_pos_sim / statesY.shape[0]).item()
    avg_neg_sim = (avg_neg_sim / statesY.shape[0]).item()
    return total_loss.item(), contrastive_loss.item(), cross_entropy_loss.item(), avg_pos_sim, avg_neg_sim


def main(hyperparams: dict):
    set_seed(hyperparams['seed'], None)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Training on ", device)

    psm_functions = {"f": psm.psm_f_fast, "fb": psm.psm_fb_fast}
    psm_func = psm_functions[hyperparams["psm"]]

    training_conf = list(TRAIN_CONFIGURATIONS[hyperparams["conf"]])

    net = ActorNet().to(device)

    optimizer = optim.Adam(net.parameters(), lr=hyperparams['learning_rate'])
    loss_bc = nn.CrossEntropyLoss()

    tb = SummaryWriter(log_dir=hyperparams['train_dir'] + os.sep + str(hyperparams['seed']))
    tb.add_text('info/args', dict2mdtable(hyperparams))

    buffer = JumpingExpertBuffer(training_conf, device, hyperparams['seed'])

    for i in range(hyperparams['n_iterations']):
        info = train(net, optimizer, hyperparams['alpha1'], hyperparams['alpha2'], hyperparams['beta'],
                     hyperparams['lambda'], psm_func,
                     buffer, loss_bc, hyperparams['batch_size'])

        total_err, contrastive_err, cross_entropy_err, avg_pos_sim, avg_neg_sim = info

        tb.add_scalar("loss/total", total_err, i)
        tb.add_scalar("loss/contrastive", contrastive_err, i)
        tb.add_scalar("loss/bc", cross_entropy_err, i)
        tb.add_scalar("debug/positive_similarity", avg_pos_sim, i)
        tb.add_scalar("debug/negative_similarity", avg_neg_sim, i)

        print(f"Iteration {i}. Loss: {total_err:2.3f}", end='')
        if i % 250 == 0:
            grid, train_perf, test_perf, total_perf = validate(net, device, training_conf)
            print(f", train perf: {train_perf:.3f}, test perf: {test_perf:.3f}, total perf:  {total_perf:.3f}")
            tb.add_scalar("val/generalization", total_perf, i)
            tb.add_image("val/fig", generate_image(grid, training_conf), i, dataformats="HWC")
        else:
            print("")
    state = {
        'state_dict': net.state_dict(),
        'optimizer': optimizer.state_dict(),
        'info': {'conf': training_conf}
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

    parser.add_argument("-lr", "--learning_rate", default=0.001, type=float,
                        help="Learning rate for the optimizer")
    parser.add_argument("-K", "--n_iterations", default=3_000, type=int,
                        help="Number of total training steps")

    parser.add_argument("-a1", "--alpha1", default=5., type=float,
                        help="Scaling factor for the alignment loss")

    parser.add_argument("-a2", "--alpha2", default=1., type=float,
                        help="Scaling factor for the BC loss")

    parser.add_argument("-b", "--beta", default=1.0, type=float,
                        help="Scaling factor for the PSM")

    parser.add_argument("-l", "--lambda", default=1.0, type=float,
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
