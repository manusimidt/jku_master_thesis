import argparse
import copy
import json
import os
from multiprocessing import Pool
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import torch.optim as optim

import wandb

import common.psm as psm
from coinrun.validate import validate
from common import set_seed, get_date_str, augmentations
from coinrun.env import CoinRunReplayBuffer

from coinrun.policy import CoinRunActor
from common.training_helpers import cosine_similarity, contrastive_loss_repository


@torch.enable_grad()
def train(net, optim, alpha1, alpha2, beta, inv_temp, psm_func, buffer, loss_bc, batch_size,
          augmentation=augmentations.identity):
    net.train()
    alignment_loss = cross_entropy_loss = torch.tensor(0)
    metric_values, similarity_matrix = None, None
    if alpha1 != 0:
        statesX, actionsX = buffer.sample_trajectory()
        statesY, actionsY = buffer.sample_trajectory()

        statesX, statesY = augmentation(statesX), augmentation(statesY)

        representation_1 = net.forward(statesX, contrastive=True)  # z_theta(x)
        representation_2 = net.forward(statesY, contrastive=True)  # z_theta(y)
        similarity_matrix = cosine_similarity(representation_1, representation_2)

        metric_values = psm_func(actionsX, actionsY)
        alignment_loss = contrastive_loss_repository(similarity_matrix, metric_values, inv_temp)

        # log_sim(cosine_similarity(representation_1, representation_2), metric_values)

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

    return total_loss.item(), alignment_loss.item(), cross_entropy_loss.item(), metric_values, similarity_matrix


@torch.no_grad()
def evaluate(net: CoinRunActor, buffer: CoinRunReplayBuffer, batch_size: int, loss_actor=nn.CrossEntropyLoss()):
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

    wandb.init(project="pse-coinrun", dir=train_dir, group=experiment_id, config=hyperparams)
    losses = []

    psm_functions = {"f": psm.psm_f_fast, "fb": psm.psm_fb_fast}
    psm_func = psm_functions[hyperparams["psm"]]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Training on ", device)
    net = CoinRunActor().to(device)
    optimizer = optim.Adam(net.parameters(), lr=hyperparams['learning_rate'], weight_decay=hyperparams['weight_decay'])
    lr_decay = optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=hyperparams['learning_decay'])

    loss_bc = nn.CrossEntropyLoss()

    buffer = CoinRunReplayBuffer(device, hyperparams['seed'], './coinrun/expert-dataset/easy')

    for step in range(hyperparams['n_iterations']):
        # Sample a pair of training MDPs
        info = train(net, optimizer, hyperparams['alpha1'], hyperparams['alpha2'], hyperparams['beta'],
                     hyperparams['lambda'], psm_func, buffer, loss_bc, hyperparams['batch_size'],
                     augmentations.aug_map[hyperparams['augmentation']])

        total_err, contrastive_err, cross_entropy_err, metric_values, similarity_matrix = info
        test_err = evaluate(net, buffer, hyperparams['batch_size'])

        loss_dict = {"Total loss": total_err, "Contrastive loss": contrastive_err, "BC loss": cross_entropy_err,
                     "Test loss": test_err, "Learning rate": lr_decay.get_last_lr()[0]}
        wandb.log(loss_dict, step=step)
        losses.append(loss_dict)
        if step % 100 == 0:
            lr_decay.step()

        print(f"Iteration {step}. Loss: {total_err:2.3f}")
        if step % 2000 == 0 and step != 0:
            solved_train, avg_reward_train, avg_steps_train = validate(net, start_level=100, num_levels=30)
            solved_test, avg_reward_test, avg_steps_test = validate(net, start_level=1000000, num_levels=30)
            wandb.log({
                "solved_train": solved_train,
                "solved_test": solved_test,
                "avg_reward_train": avg_reward_train,
                "avg_reward_test": avg_reward_test,
                "avg_steps_train": avg_steps_train,
                "avg_steps_test": avg_steps_test,
                "Sim matrix": wandb.Image(similarity_matrix),
                "Metric values": wandb.Image(metric_values),
            }, step=step)

            if step % 5000 == 0 and hyperparams['alpha1'] != 0:
                similarity_matrix = similarity_matrix.detach().cpu().numpy()
                metric_values = metric_values.detach().cpu().numpy()
                np.savez(train_dir + os.sep + run_name + f"-{step}", sim_matrix=similarity_matrix,
                         metric_values=metric_values)
    state = {
        'state_dict': net.state_dict(),
        'optimizer': optimizer.state_dict(),
        'info': {}
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

    parser.add_argument("-c", "--conf", choices=["narrow_grid", "wide_grid", "random_grid"], default="wide_grid",
                        help="The environment configuration to train on")

    parser.add_argument("-psm", "--psm", choices=["f", "fb"], default="fb",
                        help="The PSM distance function to use (f=forward PSM, fb=forward-backward PSM)")

    parser.add_argument("-bs", "--batch_size", default=256, type=int,
                        help="Size of one Minibatch")

    parser.add_argument("-lr", "--learning_rate", default=0.01, type=float,
                        help="Learning rate for the optimizer")
    parser.add_argument("-K", "--n_iterations", default=100_000, type=int,
                        help="Number of total training steps")

    parser.add_argument("-a1", "--alpha1", default=5., type=float,
                        help="Scaling factor for the alignment loss")

    parser.add_argument("-a2", "--alpha2", default=1., type=float,
                        help="Scaling factor for the BC loss")

    parser.add_argument("-b", "--beta", default=1.0, type=float,
                        help="Scaling factor for the PSM")

    parser.add_argument("-ld", "--learning_decay", default=0.995, type=float,
                        help="learning rate decay")

    parser.add_argument("-wd", "--weight_decay", default=0.0, type=float,
                        help="weight decay")

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
