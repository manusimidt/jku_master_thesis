import argparse
import copy
import json
import os
from multiprocessing import Pool

import pandas as pd
import torch
import torch.nn as nn

import torch.optim as optim

import augmentations
import wandb

import jumping.psm as psm
from common import set_seed, get_date_str
from env import TRAIN_CONFIGURATIONS, JumpingExpertBuffer

from policy import ActorNet
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
def train(net, optim, alpha1, alpha2, beta, inv_temp, psm_func, buffer, loss_bc, batch_size,
          augmentation=augmentations.identity):
    net.train()
    alignment_loss = cross_entropy_loss = torch.tensor(0)

    if alpha1 != 0:
        statesX, actionsX = buffer.sample_trajectory()
        statesY, actionsY = buffer.sample_trajectory()

        statesX, statesY = augmentation(statesX), augmentation(statesY)

        representation_1 = net.forward(statesX, contrastive=True)  # z_theta(x)
        representation_2 = net.forward(statesY, contrastive=True)  # z_theta(y)
        similarity_matrix = cosine_similarity(representation_1, representation_2)

        metric_values = psm_func(actionsX, actionsY)
        alignment_loss = contrastive_loss(similarity_matrix, metric_values, inv_temp, beta)

    if alpha2 != 0:
        bc_states, bc_actions = buffer.sample(batch_size)
        bc_states = augmentation(bc_states)
        # if alpha1 is zero, the first layers are not touched by PSE => train them with BC
        states_y_logits = net.forward(bc_states, contrastive=False, full_network=alpha1 == 0)
        cross_entropy_loss = loss_bc(states_y_logits, bc_actions.to(torch.int64))

    total_loss = alpha1 * alignment_loss + alpha2 * cross_entropy_loss

    optim.zero_grad()
    total_loss.backward()
    optim.step()

    return total_loss.item(), alignment_loss.item(), cross_entropy_loss.item()


@torch.no_grad()
def evaluate(net: ActorNet, buffer: JumpingExpertBuffer, batch_size: int, loss_actor=nn.CrossEntropyLoss()):
    """
    Evaluates the network on a randomly sampled batch without applying any augmentation!
    """
    net.eval()

    states, actions = buffer.sample(batch_size)

    pred_action_logits = net.forward(states, contrastive=False)
    actor_error = loss_actor(pred_action_logits, actions)

    return actor_error.item()


def main(hyperparams: dict, train_dir: str, experiment_id: str):
    set_seed(hyperparams['seed'])

    run_name = 'run-' + str(hyperparams['seed'])

    wandb.init(project="bc-generalization", dir=train_dir, group=experiment_id, config=hyperparams)
    losses = []

    psm_functions = {"f": psm.psm_f_fast, "fb": psm.psm_fb_fast}
    psm_func = psm_functions[hyperparams["psm"]]
    training_conf = list(TRAIN_CONFIGURATIONS[hyperparams["conf"]])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Training on ", device)
    net = ActorNet().to(device)
    optimizer = optim.Adam(net.parameters(), lr=hyperparams['learning_rate'])
    loss_bc = nn.CrossEntropyLoss()

    buffer = JumpingExpertBuffer(training_conf, device, hyperparams['seed'])

    for step in range(hyperparams['n_iterations']):
        # Sample a pair of training MDPs
        info = train(net, optimizer, hyperparams['alpha1'], hyperparams['alpha2'], hyperparams['beta'],
                     hyperparams['lambda'], psm_func, buffer, loss_bc, hyperparams['batch_size'],
                     augmentations.aug_map[hyperparams['augmentation']])

        total_err, contrastive_err, cross_entropy_err = info
        test_err = evaluate(net, buffer, hyperparams['batch_size'])

        loss_dict = {"Total loss": total_err, "Contrastive loss": contrastive_err, "BC loss": cross_entropy_err,
                     "Test loss": test_err}
        wandb.log(loss_dict, step=step)
        losses.append(loss_dict)

        print(f"Iteration {step}. Loss: {total_err:2.3f}")
        if step % 250 == 0:
            grid, train_perf, test_perf, total_perf, avg_jumps = validate(net, device, training_conf)
            print(
                f"Validation: train perf: {train_perf:.3f}, test perf: {test_perf:.3f}, total perf:  {total_perf:.3f}")
            wandb.log({
                "Generalization perf. on train": train_perf,
                "Generalization perf. on test": test_perf,
                "Total Generalization perf.": total_perf,
                "Average jumps per episode:": avg_jumps,
                "Generalization viz": wandb.Image(grid.T)
            }, step=step)
    state = {
        'state_dict': net.state_dict(),
        'optimizer': optimizer.state_dict(),
        'info': {'conf': training_conf}
    }
    torch.save(state, train_dir + os.sep + run_name + '.pth')
    # # save hyperparams
    df = pd.DataFrame(losses)
    df.to_csv(train_dir + os.sep + run_name + '.csv')


if __name__ == '__main__':
    set_seed(123, None)  # Base seed

    parser = argparse.ArgumentParser()

    parser.add_argument("--train_dir", default=f'./experiments/{get_date_str()}/',
                        help="The directory to store the results from this run into")

    parser.add_argument("-c", "--conf", choices=["narrow_grid", "wide_grid"], default="wide_grid",
                        help="The environment configuration to train on")

    parser.add_argument("-psm", "--psm", choices=["f", "fb"], default="f",
                        help="The PSM distance function to use (f=forward PSM, fb=forward-backward PSM)")

    parser.add_argument("-bs", "--batch_size", default=256, type=int,
                        help="Size of one Minibatch")

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

    parser.add_argument("-aug", "--augmentation", default="identity",
                        choices=list(augmentations.aug_map.keys()),
                        help="The augmentation that should be applied to the states")

    parser.add_argument("-s", "--seed", default=[1], type=int, nargs='+',
                        help="The seed to train on. If multiple values are provided, the script will "
                             "spawn a new process for each seed")

    args = parser.parse_args()
    _hyperparams = copy.deepcopy(vars(args))
    print(_hyperparams)
    _experiment_id = get_date_str()

    _train_dir = os.path.dirname(os.path.abspath(__file__)) + os.sep + 'experiments' + os.sep + _experiment_id
    if not os.path.exists(_train_dir):
        os.makedirs(_train_dir)

    # save hyperparams
    with open(_train_dir + os.sep + 'hyperparams.json', "w") as outfile:
        json.dump(_hyperparams, outfile, indent=2)

    if len(args.seed) == 1:
        # convert the list to a single value
        _hyperparams['seed'] = _hyperparams['seed'][0]
        main(_hyperparams, _train_dir, _experiment_id)

    else:
        params_arr = []
        for i in range(len(args.seed)):
            curr_hyperparams = copy.deepcopy(_hyperparams)
            curr_hyperparams['seed'] = _hyperparams['seed'][i]
            params_arr.append((curr_hyperparams, _train_dir, _experiment_id))
        with Pool(3) as p:
            p.starmap(main, params_arr)
