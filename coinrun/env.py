import os
import gym
import torch
import numpy as np
from gym import spaces


class VanillaEnv(gym.Env):

    def __init__(self, start_level=0, num_levels=0, distribution_mode="easy", render_mode="rgb_array"):
        # dist mode hard => 17
        # dist mode easy => 13
        self.actual_env = gym.make('procgen:procgen-coinrun-v0', start_level=start_level, paint_vel_info=True,
                                   num_levels=num_levels, distribution_mode=distribution_mode, render_mode=render_mode)

        self.action_space = self.actual_env.action_space
        # Example for using image as input (channel-first; channel-last also works):
        self.observation_space = spaces.Box(low=0., high=1.,
                                            shape=(3, 64, 64), dtype=np.float32)

    def step(self, action):
        obs, r, done, info = self.actual_env.step(action)

        info["success"] = r == 10

        r += ((obs[0, 0, 0] / 255) - 0.5) * 0.2
        if action in (2, 5, 8):
            r -= 0.1

        # if r == 0: r = -0.02  # slightly punish the agent for each time step
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
        self.buffer_size = self._calc_buffer_size() 
        self.rng = np.random.default_rng(seed=seed)

        self.states = torch.empty((self.buffer_size, 3, 64, 64), dtype=torch.float32)
        self.actions = torch.empty(self.buffer_size, dtype=torch.int8)

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
                    image = np.array(np.moveaxis(image, -1, -3) / 255, dtype=np.float32)
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
        return self.states[ind].to(self.device), self.actions[ind].to(self.device).to(torch.int64)

    def sample_trajectory(self) -> tuple:
        """
        Samples an entire expert trajectory
        """
        idx = self.rng.choice(np.arange(len(self.episode_start_idx)))
        next_idx = idx + 1 if idx != len(self.episode_start_idx) - 1 else None

        if next_idx:
            return self.states[self.episode_start_idx[idx]:self.episode_start_idx[next_idx]].to(self.device), \
                self.actions[self.episode_start_idx[idx]:self.episode_start_idx[next_idx]].to(self.device)
        else:
            return self.states[self.episode_start_idx[idx]:].to(self.device), \
                self.actions[self.episode_start_idx[idx]:].to(self.device)


if __name__ == '__main__':
    buffer = CoinRunReplayBuffer('cpu', 0, './coinrun/expert-dataset/easy')
    print(buffer.sample(batch_size=23)[0].shape)
    print(buffer.sample_trajectory()[0].shape)
    state = buffer.sample(batch_size=23)[0]
    print(f"Min: {state.min()}, max: {state.max()}")

    _env = VanillaEnv()
    _obs_arr = [_env.reset(), _env.step(0)[0], _env.step(0)[0], _env.step(0)[0], _env.step(1)[0]]
    for _obs in _obs_arr:
        assert _obs.dtype == np.float32, "Incorrect datatype"
        assert _obs.shape == (3, 64, 64), "Incorrect shape"
