import argparse
import copy
import logging
import gymnasium as gym
import numpy as np
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed

from coinrun.env import VanillaEnv

logging.basicConfig(level=logging.INFO)


def validate(model, start_level, num_levels, iterations=100):
    device = next(model.parameters())
    env = VanillaEnv(start_level=start_level, num_levels=num_levels)

    avg_reward = []
    avg_iterations = []
    solved = 0

    for _ in range(iterations):
        cum_reward = 0  # cumulative reward
        num_iterations = 0
        states = []
        actions = []
        obs = env.reset()
        while True:
            action = model.predict(obs, deterministic=True)[0]
            next_obs, rew, done, info = env.step(action)

            num_iterations += 1
            cum_reward += rew
            states.append(obs)
            actions.append(action)

            obs = next_obs

            if done:
                if info["success"]:
                    solved += 1
                break
        avg_reward.append(cum_reward)
        avg_iterations.append(num_iterations)
    return solved / iterations, np.mean(avg_reward), np.mean(avg_iterations)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--num_levels", default=128, type=int,
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

    env = VanillaEnv(start_level=0, num_levels=_hyperparams['num_levels'])

    model = PPO("MlpPolicy", env, verbose=1)

    for _ in range(25):
        model.learn(total_timesteps=10_000)
        solved_train, avg_reward_train, avg_steps_train = validate(model.policy, start_level=0,
                                                                   num_levels=_hyperparams['num_levels'],
                                                                   iterations=50)
        solved_test, avg_reward_test, avg_steps_test = validate(model.policy, start_level=1000000,
                                                                num_levels=_hyperparams['num_levels'],
                                                                iterations=50)
        print(f"Solved: {solved_train}/{solved_test}, Reward: {avg_reward_train}/{avg_reward_test}, Steps: {avg_steps_train}/{avg_steps_test}")