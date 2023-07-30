import os
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

    def __init__(self, seed):
        super().__init__()

        self.num_actions = 17
        self.action_space = spaces.Discrete(self.num_actions)
        self.observation_space = spaces.Box(low=0, high=1,
                                            shape=(3, 64, 64), dtype=np.float32)
        self.actualEnv = crafter.Recorder(
            crafter.Env(seed=seed),
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


class CrafterReplayBuffer:
    def __init__(self, device, seed, data_dir):
        self.device = device
        self.seed = seed
        self.data_dir = data_dir
        self.buffer_size = 62002  # Size of the imitation dataset i downloaded
        self.rng = np.random.default_rng(seed=seed)

        self.states = torch.empty((self.buffer_size, 3, 64, 64), device=device, dtype=torch.float32)
        self.actions = torch.empty(self.buffer_size, device=device, dtype=torch.int64)

        self.episode_start_idx = []

        self._populate()

    def _populate(self):
        i = 0
        for file in os.listdir(self.data_dir):
            with np.load(self.data_dir + os.sep + file) as data:
                self.episode_start_idx.append(i)
                for image, action in zip(data['image'], data['action']):
                    self.states[i] = torch.tensor(np.moveaxis(image, -1, -3) / 255, dtype=torch.float32,
                                                  device=self.device)
                    self.actions[i] = torch.tensor(action, device=self.device)
                    i += 1
        assert i == self.buffer_size, "Not all data was loaded!"

    def sample(self, batch_size, replace=False) -> tuple:
        """
        :param batch_size:
        :param replace: if replace is set to true, a batch can contain the same state twice
        """
        ind = self.rng.choice(range(self.buffer_size), size=batch_size, replace=replace)
        return self.states[ind], self.actions[ind]

    def sample_trajectory(self) -> tuple:
        """
        Samples an entire expert trajectory
        """
        idx = self.rng.choice(np.arange(len(self.episode_start_idx)))
        next_idx = idx + 1 if idx != len(self.episode_start_idx) - 1 else None

        if next_idx:
            return self.states[self.episode_start_idx[idx]:self.episode_start_idx[next_idx]], \
                self.actions[self.episode_start_idx[idx]:self.episode_start_idx[next_idx]]
        else:
            return self.states[self.episode_start_idx[idx]:], self.actions[self.episode_start_idx[idx]:]


if __name__ == '__main__':
    buffer = CrafterReplayBuffer('cpu', 0, './dataset')
    buffer.sample(batch_size=23)
    buffer.sample_trajectory()

    _envs = [VanillaEnv()]
    for _env in _envs:
        _obs_arr = [_env.reset(), _env.step(0)[0], _env.step(0)[0], _env.step(0)[0], _env.step(1)[0]]
        for _obs in _obs_arr:
            assert _obs.dtype == np.float32, "Incorrect datatype"
            assert _obs.shape == (3, 64, 64), "Incorrect shape"