import argparse
import copy
from multiprocessing import Pool

import numpy as np

from common import set_seed
from env import VanillaEnv, TRAIN_CONFIGURATIONS, BCDataset, generate_bc_dataset, AugmentingEnv
from policy import ActorNet
from rl.common.buffer2 import Transition, Episode
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import List
from rl.ppo.ppo import PPO


@torch.enable_grad()
def train(net: ActorNet, dataLoader: DataLoader, optim_actor, loss_actor=nn.CrossEntropyLoss()):
    device = next(net.parameters()).device
    net.train()
    actor_errors = []

    for batch in dataLoader:
        # X is the observation
        # y contains the choosen action and the return estimate from the critic
        X, y = batch[0].to(device), batch[1].to(device)

        pred_action_logits = net.forward(X, contrastive=False, full_network=True)

        actor_error = loss_actor(pred_action_logits, y.to(torch.int64))

        optim_actor.zero_grad()
        actor_error.backward()
        optim_actor.step()

        actor_errors.append(actor_error.item())

    return actor_errors


@torch.no_grad()
def evaluate(net: ActorNet, dataLoader: DataLoader, loss_actor):
    device = next(net.parameters()).device
    net.eval()
    actor_errors = []

    for batch in dataLoader:
        X, y = batch[0].to(device), batch[1].to(device)
        pred_action_logits = net.forward(X, contrastive=False)
        actor_errors.append(loss_actor(pred_action_logits, y.to(torch.int64)).item())

    return actor_errors


def main(model, hyperparams: dict, train_loader, test_loader):
    set_seed(hyperparams['seed'], env=None)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Training on ", device)

    model = model.to(device)
    optim_actor = optim.Adam(model.parameters(), lr=hyperparams['learning_rate'])

    loss_actor = nn.CrossEntropyLoss()

    for epoch in range(hyperparams['n_epochs']):
        actor_errors = train(model, train_loader, optim_actor, loss_actor)
        val_actor_errors = evaluate(model, test_loader, loss_actor)

        print(f"Epoch {epoch} Train errors: Actor {np.mean(actor_errors):.8f}, "
              f"Val errors: Actor {np.mean(val_actor_errors):.8f}")

    state = {
        'state_dict': model.state_dict(),
        'optimizer': optim_actor.state_dict(),
        'info': {'conf': hyperparams['conf']}
    }
    torch.save(state, f'ckpts/bc-agent-{hyperparams["seed"]}.pth')


if __name__ == '__main__':
    _base_model = ''
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--env", choices=["vanilla", "random"], default="vanilla",
                        help="The environment to train on")
    parser.add_argument("-c", "--conf", choices=["narrow_grid", "wide_grid"], default="wide_grid",
                        help="The environment configuration to train on")

    parser.add_argument("-bs", "--batch_size", default=256, type=int,
                        help="Size of one Minibatch")
    parser.add_argument("-bc", "--batch_count", default=10, type=int,
                        help="Number of batches. BC will be trained on batch_size x batch_count samples")

    parser.add_argument("-b", "--balanced", default=False, type=bool,
                        help="If true, the algorithm will be trained on a balanced dataset (1/3 action 1, 2/3 action 0 "
                             "examples)")

    parser.add_argument("-lr", "--learning_rate", default=0.0004, type=float,
                        help="Learning rate for the optimizer")
    parser.add_argument("-ne", "--n_epochs", default=20, type=int,
                        help="Number of epochs (Number of times the training will run over the entire dataset)")

    parser.add_argument("-s", "--seed", default=[1], type=int, nargs='+',
                        help="The seed to train on. If multiple values are provided, the script will "
                             "spawn a new process for each seed")

    args = parser.parse_args()
    hyperparams = copy.deepcopy(vars(args))

    conf = list(TRAIN_CONFIGURATIONS[hyperparams['conf']])
    hyperparams['conf'] = conf
    if hyperparams['env'] == 'vanilla':
        env = VanillaEnv(conf)
    else:
        env = AugmentingEnv(conf)

    print(f"Generating expert episodes...")
    train_loader, test_loader = generate_bc_dataset([env], hyperparams['batch_size'], hyperparams['batch_count'],
                                                    balanced=hyperparams['balanced'])

    if len(args.seed) == 1:
        # convert the list to a single value
        hyperparams['seed'] = hyperparams['seed'][0]
        main(ActorNet(), hyperparams, train_loader, test_loader)
    else:
        params_arr = []
        for i in range(len(args.seed)):
            curr_hyperparams = copy.deepcopy(hyperparams)
            curr_hyperparams['seed'] = hyperparams['seed'][i]
            params_arr.append((ActorNet(), curr_hyperparams, train_loader, test_loader))
        with Pool(5) as p:
            results = p.starmap(main, params_arr)
            print(results)
