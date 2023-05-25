from collections import deque

import gym
import numpy as np
import torch
import crafter
from gym import spaces
from typing import List
import itertools
import torchvision.transforms.functional as fn
from torchvision.transforms.functional import InterpolationMode


class VanillaEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, rendering=False):
        super().__init__()

        self.num_actions = 17
        self.action_space = spaces.Discrete(self.num_actions)
        self.observation_space = spaces.Box(low=0, high=1,
                                            shape=(3, 64, 64), dtype=np.float32)
        self.actualEnv = crafter.Recorder(
            gym.make('CrafterReward-v1'),
            './logdir',
            save_stats=True,
            save_video=False,
            save_episode=False,
        )

    def step(self, action) -> tuple:
        obs, r, done, info = self.actualEnv.step(action)
        return np.array(np.moveaxis(obs, -1, -3) / 255, dtype=np.float32), float(r), done, info

    def reset(self) -> np.ndarray:
        obs = self.actualEnv.reset()
        return np.array(np.moveaxis(obs, -1, -3) / 255, dtype=np.float32)

    def render(self, mode="human"):
        pass

    def close(self):
        self.actualEnv.close()


if __name__ == '__main__':

    _envs = [VanillaEnv()]
    for _env in _envs:
        _obs_arr = [_env.reset(), _env.step(0)[0], _env.step(0)[0], _env.step(0)[0], _env.step(1)[0]]
        for _obs in _obs_arr:
            assert _obs.dtype == np.float32, "Incorrect datatype"
            assert _obs.shape == (3, 64, 64), "Incorrect shape"

