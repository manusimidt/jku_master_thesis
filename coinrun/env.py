import os
import gym
import torch
import numpy as np


class VanillaEnv(gym.Env):
    def __init__(self, start_level=0, num_levels=0):
        self.actual_env = gym.make('procgen:procgen-coinrun-v0', start_level=start_level,
                                   num_levels=num_levels, distribution_mode="easy")

    def step(self, action):
        obs, r, done, info = self.actual_env.step(action)
        info["success"] = r == 10
        #if r == 0: r = -0.02  # slightly punish the agent for each time step
        return np.array(np.moveaxis(obs, -1, -3) / 255, dtype=np.float32), r, done, info

    def reset(self, **kwargs):
        obs = self.actual_env.reset()
        return np.array(np.moveaxis(obs, -1, -3) / 255, dtype=np.float32)

    def close(self):
        self.actual_env.close()


class CoinRunReplayBuffer:
    def __init__(self, device, seed, data_dir):
        self.device = device
        self.data_dir = data_dir
        self.buffer_size = self._calc_buffer_size()  # Size of the imitation dataset i downloaded
        self.rng = np.random.default_rng(seed=seed)

        self.states = torch.empty((self.buffer_size, 3, 64, 64), device=device, dtype=torch.float32)
        self.actions = torch.empty(self.buffer_size, device=device, dtype=torch.int64)

        self.episode_start_idx = []

        self._populate()

    def _calc_buffer_size(self):
        size = 0
        for file in os.listdir(self.data_dir):
            with np.load(self.data_dir + os.sep + file) as data:
                size += len(data['state'])
        return size

    def _populate(self):
        i = 0
        for file in os.listdir(self.data_dir):
            with np.load(self.data_dir + os.sep + file) as data:
                self.episode_start_idx.append(i)
                for image, action in zip(data['state'], data['action']):
                    self.states[i] = torch.tensor(image, device=self.device)
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
    buffer = CoinRunReplayBuffer('cpu', 0, './dataset/10')
    print(buffer.sample(batch_size=23)[0].shape)
    print(buffer.sample_trajectory()[0].shape)

    _env = VanillaEnv()
    _obs_arr = [_env.reset(), _env.step(0)[0], _env.step(0)[0], _env.step(0)[0], _env.step(1)[0]]
    for _obs in _obs_arr:
        assert _obs.dtype == np.float32, "Incorrect datatype"
        assert _obs.shape == (3, 64, 64), "Incorrect shape"
