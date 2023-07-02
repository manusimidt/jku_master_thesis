import argparse
import copy
import json
from multiprocessing import Pool

import numpy as np

from common import set_seed
from env import VanillaEnv, TRAIN_CONFIGURATIONS, generate_bc_dataset, AugmentingEnv
from policy import ActorNet
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

from rl.common.logger import CSVLogger, get_date_str


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


def main(model, hyperparams: dict, logger: CSVLogger):
    set_seed(hyperparams['seed'], env=None)
    conf = list(TRAIN_CONFIGURATIONS[hyperparams['conf']])
    if hyperparams['env'] == 'vanilla':
        env = VanillaEnv(conf)
    else:
        env = AugmentingEnv(conf)

    print(f"Generating expert episodes...")
    train_loader, test_loader = generate_bc_dataset([env], hyperparams['batch_size'], hyperparams['batch_count'],
                                                    balanced=hyperparams['balanced'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Training on ", device)

    model = model.to(device)
    optim_actor = optim.Adam(model.parameters(), lr=hyperparams['learning_rate'])

    loss_actor = nn.CrossEntropyLoss()

    total_train_loss, total_test_loss = [], []
    for epoch in range(hyperparams['n_epochs']):
        train_loss = np.mean(train(model, train_loader, optim_actor, loss_actor))
        test_loss = np.mean(evaluate(model, test_loader, loss_actor))

        print(f"Epoch {epoch} Train errors: {train_loss:.8f}, Val errors: {test_loss:.8f}")
        total_train_loss.append(train_loss)
        total_test_loss.append(test_loss)
        logger.on_epoch_end(epoch, train_loss=train_loss, test_loss=test_loss)

    state = {
        'state_dict': model.state_dict(),
        'optimizer': optim_actor.state_dict(),
        'info': {'conf': conf}
    }
    torch.save(state, logger.filepath.replace('.csv', '.pth'))
    # save hyperparams
    with open(logger.filepath.replace('.csv', '.json'), "w") as outfile:
        json.dump(hyperparams, outfile, indent=2)
    return total_train_loss, total_test_loss


if __name__ == '__main__':
    set_seed(123, None)  # Base seed
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--env", choices=["vanilla", "random"], default="vanilla",
                        help="The environment to train on")
    parser.add_argument("-c", "--conf", choices=["narrow_grid", "wide_grid"], default="wide_grid",
                        help="The environment configuration to train on")

    parser.add_argument("-bs", "--batch_size", default=256, type=int,
                        help="Size of one Minibatch")
    parser.add_argument("-bc", "--batch_count", default=10, type=int,
                        help="Number of batches. BC will be trained on batch_size x batch_count samples")

    parser.add_argument("--balanced", default=True, type=bool, action=argparse.BooleanOptionalAction,
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
    print(hyperparams)
    if len(args.seed) == 1:
        # convert the list to a single value
        hyperparams['seed'] = hyperparams['seed'][0]
        logger = CSVLogger(f'./experiments/{get_date_str()}/', f'train_bc{hyperparams["seed"]}',
                           columns=["train_loss", "test_loss"])
        main(ActorNet(), hyperparams, logger)

    else:
        params_arr = []
        for i in range(len(args.seed)):
            curr_hyperparams = copy.deepcopy(hyperparams)
            curr_hyperparams['seed'] = hyperparams['seed'][i]

            logger = CSVLogger(f'./experiments/{get_date_str()}/', f'train_bc{curr_hyperparams["seed"]}',
                               columns=["train_loss", "test_loss"])
            params_arr.append((ActorNet(), curr_hyperparams, logger))
        with Pool(4) as p:
            p.starmap(main, params_arr)

