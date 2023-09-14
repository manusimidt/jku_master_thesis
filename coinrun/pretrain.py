import argparse
import copy
import logging
from torch import optim

from coinrun.env import VanillaEnv
from coinrun.policy import CoinRunActor, CoinRunCritic
from coinrun.validate import validate
from common.rl.ppo.ppo import PPO
from common.rl.buffer2 import RolloutBuffer
from common.rl.ppo.policies import ActorCriticNet
from rl.logger import ConsoleLogger, Tracker, WandBLogger
from rl.utils import set_seed

logging.basicConfig(level=logging.INFO)
RANDOM_SEED = 1
START_SEED = 17

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--num_levels", default=1024, type=int,
                        help="Number of different seeds/levels used for training")
    parser.add_argument("--n_epochs", default=3, type=int,
                        help="Number of epochs per training")
    parser.add_argument("-lr", "--learning_rate", default=0.004, type=float,
                        help="Learning rate for the optimizer")
    parser.add_argument("-ld", "--learning_decay", default=1, type=float,
                        help="learning rate decay")
    parser.add_argument("-bs", "--batch_size", default=256, type=int,
                        help="Size of one Minibatch")

    parser.add_argument("--gamma", default=0.999, type=float,
                        help="Discount factor")
    parser.add_argument('--min_transitions', default=500, type=int,
                        help="Minimum number of transitions collected before doing the training")
    set_seed(None, RANDOM_SEED)
    args = parser.parse_args()
    _hyperparams = copy.deepcopy(vars(args))
    print(_hyperparams)

    policy: ActorCriticNet = ActorCriticNet(obs_space=(3, 64, 64), action_space=15, hidden_size=256)
    policy.actor = CoinRunActor()
    policy.critic = CoinRunCritic()

    optimizer = optim.Adam(policy.parameters(), lr=_hyperparams["learning_rate"])
    lr_decay = optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=_hyperparams['learning_decay'])
    buffer = RolloutBuffer(capacity=_hyperparams['min_transitions']+100, batch_size=_hyperparams["batch_size"],
                           min_transitions=_hyperparams['min_transitions'])

    logger1 = ConsoleLogger(log_every=50, average_over=50)
    logger2 = WandBLogger(project='ppo-coin-run', info=_hyperparams)
    tracker = Tracker(logger1, logger2)

    env = VanillaEnv(start_level=START_SEED, num_levels=_hyperparams['num_levels'])

    alg = PPO(policy, env, optimizer, lr_decay, seed=124, tracker=tracker, buffer=buffer,
              n_epochs=_hyperparams['n_epochs'], gamma=_hyperparams['gamma'])
    #alg.load(f'./runs/ppo-{_hyperparams["num_levels"]}.pth')
    logging.info("Training on " + alg.device)
    for i in range(10):
        alg.learn(500)
        # alg.learn(200)
        alg.save('./runs/', f'ppo-{_hyperparams["num_levels"]}')

        solved_train, avg_reward_train, avg_steps_train = validate(policy, start_level=START_SEED,
                                                                   num_levels=_hyperparams['num_levels'])
        solved_test, avg_reward_test, avg_steps_test = validate(policy, start_level=100000,
                                                                num_levels=_hyperparams['num_levels'])

        logger2.log_custom({
            "val/solved_train": solved_train, "val/solved_test": solved_test,
            "val/avg_reward_train": avg_reward_train, "val/avg_reward_test": avg_reward_test,
            "val/avg_steps_train": avg_steps_train, "val/avg_steps_test": avg_steps_test,
        })
