import argparse
import copy
import json
from multiprocessing import Pool
import wandb
import augmentations
from common import set_seed
from env import TRAIN_CONFIGURATIONS, JumpingExpertBuffer
from policy import ActorNet
import torch
from torch.optim import Optimizer, Adam
import torch.nn as nn

from rl.common.logger import CSVLogger, get_date_str
from validate import validate


@torch.enable_grad()
def train(net: ActorNet, buffer: JumpingExpertBuffer, batch_size: int, optim_actor: Optimizer,
          loss_actor=nn.CrossEntropyLoss(), augmentation=augmentations.identity):
    """
    Trains the network on a randomly sampled batch and applies the given augmentation to each batch
    """
    net.train()

    states, actions = buffer.sample(batch_size)
    aug_states = augmentation(states)

    pred_action_logits = net.forward(aug_states, contrastive=False, full_network=True)
    actor_error = loss_actor(pred_action_logits, actions)

    optim_actor.zero_grad()
    actor_error.backward()
    optim_actor.step()

    return actor_error.item()


@torch.no_grad()
def evaluate(net: ActorNet, buffer: JumpingExpertBuffer, batch_size: int, optim_actor: Optimizer,
             loss_actor=nn.CrossEntropyLoss()):
    """
    Evaluates the network on a randomly sampled batch without applying any augmentation!
    """
    net.eval()

    states, actions = buffer.sample(batch_size)

    pred_action_logits = net.forward(states, contrastive=False)
    actor_error = loss_actor(pred_action_logits, actions)

    return actor_error.item()


def main(model, hyperparams: dict, logger: CSVLogger):
    set_seed(hyperparams['seed'], env=None)
    wandb.init(project="bc-generalization", group=logger.filepath.split('/')[-2].replace('\\', ''), config=hyperparams)

    conf = list(TRAIN_CONFIGURATIONS[hyperparams['conf']])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Training on ", device)

    buffer = JumpingExpertBuffer(conf, device, hyperparams['seed'])

    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=hyperparams['learning_rate'])

    loss_func = nn.CrossEntropyLoss()

    for step in range(hyperparams['steps']):
        train_loss = train(model, buffer, hyperparams['batch_size'], optimizer, loss_func,
                           augmentation=augmentations.aug_map[hyperparams['augmentation']])
        test_loss = evaluate(model, buffer, hyperparams['batch_size'], optimizer, loss_func)

        print(f"Step {step} Train err: {train_loss:.6f}, Test err: {test_loss:.6f}")
        wandb.log({
            "Train loss": train_loss,
            "Test loss": test_loss
        }, step=step)

        if step % 250 == 0:
            grid, train_perf, test_perf, total_perf, avg_jumps = \
                validate(model, device, TRAIN_CONFIGURATIONS[hyperparams['conf']])
            print(
                f"Validation: train perf: {train_perf:.3f}, test perf: {test_perf:.3f}, total perf:  {total_perf:.3f}")

            wandb.log({
                "Generalization perf. on train": train_perf,
                "Generalization perf. on test": test_perf,
                "Total Generalization perf.": total_perf,
                "Average jumps per episode:": avg_jumps,
                "Generalization viz": wandb.Image(grid.T)
            }, step=step)

        logger.on_epoch_end(step, train_loss=train_loss, test_loss=test_loss)

    state = {
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'info': {'conf': conf}
    }
    torch.save(state, logger.filepath.replace('.csv', '.pth'))
    # # save hyperparams
    with open(logger.filepath.replace('.csv', '.json'), "w") as outfile:
        json.dump(hyperparams, outfile, indent=2)
    # return total_train_loss, total_test_loss


if __name__ == '__main__':
    set_seed(123, None)  # Base seed
    parser = argparse.ArgumentParser()
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
    parser.add_argument("-K", "--steps", default=2001, type=int,
                        help="Number of training steps")

    parser.add_argument("-aug", "--augmentation", default="identity",
                        choices=list(augmentations.aug_map.keys()),
                        help="The augmentation that should be applied to the states")

    parser.add_argument("-s", "--seed", default=[1], type=int, nargs='+',
                        help="The seed to train on. If multiple values are provided, the script will "
                             "spawn a new process for each seed")

    args = parser.parse_args()
    hyperparams = copy.deepcopy(vars(args))
    print(hyperparams)
    date_str = get_date_str()

    if len(args.seed) == 1:
        # convert the list to a single value
        hyperparams['seed'] = hyperparams['seed'][0]
        logger = CSVLogger(f'./experiments/{date_str}/', f'train_bc{hyperparams["seed"]}',
                           columns=["train_loss", "test_loss"])
        main(ActorNet(), hyperparams, logger)

    else:
        params_arr = []
        for i in range(len(args.seed)):
            curr_hyperparams = copy.deepcopy(hyperparams)
            curr_hyperparams['seed'] = hyperparams['seed'][i]

            logger = CSVLogger(f'./experiments/{date_str}/', f'train_bc{curr_hyperparams["seed"]}',
                               columns=["train_loss", "test_loss"])
            params_arr.append((ActorNet(), curr_hyperparams, logger))
        with Pool(3) as p:
            p.starmap(main, params_arr)
