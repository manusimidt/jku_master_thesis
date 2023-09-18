import os

import torch
import numpy as np
from coinrun.env import VanillaEnv
from coinrun.policy import CoinRunActor, CoinRunCritic
from common.rl.ppo.policies import ActorCriticNet


def validate(model, start_level, num_levels, record_optimal=False, render_mode="rgb_array"):
    device = next(model.parameters())

    avg_reward = []
    avg_iterations = []
    solved = 0

    for i in range(num_levels):
        env = VanillaEnv(start_level=start_level + i, num_levels=1, render_mode=render_mode)
        cum_reward = 0  # cumulative reward
        num_iterations = 0
        states = []
        actions = []
        obs = env.reset()
        while True:
            if hasattr(model, 'act'): action = model.act(torch.FloatTensor(obs).to(device).unsqueeze(0))[0].item()
            else: action = torch.argmax(model.forward(torch.FloatTensor(obs).to(device).unsqueeze(0))).item()

            next_obs, rew, done, info = env.step(action)

            num_iterations += 1
            cum_reward += rew
            states.append(obs)
            actions.append(action)

            obs = next_obs

            if done:
                print(f"Validation run with seed {start_level+i}: success: {info['success']}, reward: {cum_reward}, iterations: {num_iterations}")
                if info["success"]:
                    solved += 1
                    if record_optimal:
                        target_folder = f'./dataset/{num_levels}'
                        file_path = f'{target_folder}/{info["level_seed"]}-{_get_episode_nr(target_folder, info["level_seed"])}.npz'
                        np.savez(file_path, state=np.array(states), action=np.array(actions))
                break
        avg_reward.append(cum_reward)
        avg_iterations.append(num_iterations)
    return solved / num_levels, np.mean(avg_reward), np.mean(avg_iterations)


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
    _num_seeds = 62

    ckp = torch.load(f'./runs/ppo-{_num_seeds}.pth')
    _model.load_state_dict(ckp['state_dict'])

    print(validate(_model, start_level=17, num_levels=_num_seeds, record_optimal=True))
