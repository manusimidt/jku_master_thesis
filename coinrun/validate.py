import os

import torch
import numpy as np
from coinrun.env import VanillaEnv
from coinrun.policy import CoinRunActor, CoinRunCritic
from common.rl.ppo.policies import ActorCriticNet


def validate(model, start_level, num_levels, iterations=100, record_optimal=False):
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
            action_logits = model.forward(torch.FloatTensor(obs).to(device).unsqueeze(0))
            action = torch.argmax(action_logits)
            next_obs, rew, done, info = env.step(action.item())

            num_iterations += 1
            cum_reward += rew
            states.append(obs)
            actions.append(action)

            obs = next_obs

            if done:
                if info["success"]:
                    solved += 1
                    print(f"Seed {info['level_seed']} solved!")
                    if record_optimal:
                        target_folder = f'./dataset/{num_levels}'
                        file_path = f'{target_folder}/{info["level_seed"]}-{_get_episode_nr(target_folder, info["level_seed"])}.npz'
                        np.savez(file_path, state=np.array(states), action=np.array(actions))
                break
        avg_reward.append(cum_reward)
        avg_iterations.append(num_iterations)
    return solved / iterations, np.mean(avg_reward), np.mean(avg_iterations)


def _get_episode_nr(target_folder: str, seed: int) -> int:
    """
    Each episode is named after the seed of the env and an appended increment.
    This function looks into the target_folder and returns the next increment
    """
    return len([filename for filename in os.listdir(target_folder) if filename.startswith(str(seed) + '-')])


if __name__ == '__main__':
    _model = ActorCriticNet(obs_space=(3, 64, 64), action_space=15, hidden_size=256)
    _model.actor = CoinRunActor()
    _model.critic = CoinRunCritic()
    _num_seeds = 100

    ckp = torch.load(f'./runs/ppo-{_num_seeds}.pth')
    _model.load_state_dict(ckp['state_dict'])

    print(validate(_model.actor, start_level=12345678, num_levels=_num_seeds, record_optimal=False, iterations=1000))
