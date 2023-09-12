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

logging.basicConfig(level=logging.INFO)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--num_levels", default=64, type=int,
                        help="Number of different seeds/levels used for training")
    parser.add_argument("-lr", "--learning_rate", default=0.02, type=float,
                        help="Learning rate for the optimizer")
    parser.add_argument("-ld", "--learning_decay", default=0.995, type=float,
                        help="learning rate decay")
    parser.add_argument("-bs", "--batch_size", default=256, type=int,
                        help="Size of one Minibatch")
    parser.add_argument("--n_epochs", default=5, type=int,
                        help="Size of one Minibatch")

    args = parser.parse_args()
    _hyperparams = copy.deepcopy(vars(args))
    print(_hyperparams)

    policy: ActorCriticNet = ActorCriticNet(obs_space=(3, 64, 64), action_space=15, hidden_size=256)
    policy.actor = CoinRunActor()
    policy.critic = CoinRunCritic()

    optimizer = optim.Adam(policy.parameters(), lr=_hyperparams["learning_rate"])
    lr_decay = optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=_hyperparams['learning_decay'])
    buffer = RolloutBuffer(capacity=2000, batch_size=_hyperparams["batch_size"], min_transitions=2000)

    logger1 = ConsoleLogger(log_every=100, average_over=100)
    logger2 = WandBLogger(project='ppo-coin-run', info=_hyperparams)
    tracker = Tracker(logger1, logger2)

    env = VanillaEnv(start_level=0, num_levels=_hyperparams['num_levels'])

    alg = PPO(policy, env, optimizer, lr_decay, seed=124, tracker=tracker, buffer=buffer,
              n_epochs=_hyperparams['n_epochs'])

    logging.info("Training on " + alg.device)
    for i in range(120):
        alg.learn(500)
        alg.save('./runs/', f'ppo-{_hyperparams["num_levels"]}')

        solved_train, avg_reward_train, avg_steps_train = validate(policy.actor, start_level=0,
                                                                   num_levels=_hyperparams['num_levels'],
                                                                   iterations=50)
        solved_test, avg_reward_test, avg_steps_test = validate(policy.actor, start_level=1000000,
                                                                num_levels=_hyperparams['num_levels'],
                                                                iterations=50)

        logger2.log_custom({
            "val/solved_train": solved_train, "val/solved_test": solved_test,
            "val/avg_reward_train": avg_reward_train, "val/avg_reward_test": avg_reward_test,
            "val/avg_steps_train": avg_steps_train, "val/avg_steps_test": avg_steps_test,
        })
