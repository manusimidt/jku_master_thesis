"""
This module holds different modified jumping tasks environments
"""

import gym
import numpy as np
import torch
from gym import spaces
from typing import List

from torch.utils.data import Dataset, DataLoader

from gym_jumping_task.envs import JumpTaskEnv

TRAIN_CONFIGURATIONS = {
    "narrow_grid": {
        # (obstacle_pos, floor_height)
        (26, 12), (29, 12), (31, 12), (34, 12),
        (26, 20), (29, 20), (31, 20), (34, 20),
        (26, 28), (29, 28), (31, 28), (34, 28),
    },
    "wide_grid": {
        # (obstacle_pos, floor_height)
        (22, 8), (27, 8), (32, 8), (38, 8),
        (22, 20), (27, 20), (32, 20), (38, 20),
        (22, 32), (27, 32), (32, 32), (38, 32),
    }
}


class VanillaEnv(gym.Env):
    metadata = {"render.modes": ["human"]}
    min_obstacle_pos = 14
    max_obstacle_pos = 47
    min_floor_height = 0
    max_floor_height = 40

    def __init__(self, configurations: List[tuple] or None = None, rendering=False):
        """
        :param configurations: possible configurations, array of tuples consisting of
            the obstacle position and the floor height
        """
        super().__init__()
        # If no configuration was provided, use the default JumpingTask configuration
        if configurations is None: configurations = [(30, 10), ]
        self.configurations = configurations

        # Jumping env has 2 possible actions
        self.num_actions = 2
        self.action_space = spaces.Discrete(self.num_actions)
        self.observation_space = spaces.Box(low=0, high=1,
                                            shape=(1, 60, 60), dtype=np.float32)
        self.actualEnv = JumpTaskEnv(rendering=rendering)

    def _sample_conf(self):
        """
        :return: returns random configuration as a tuple consisting of the obstacle position and
            the floor height
        """
        idx = np.random.choice(len(self.configurations))
        return self.configurations[idx]

    def step(self, action) -> tuple:
        obs, r, done, info = self.actualEnv.step(action)
        return np.expand_dims(obs, axis=0), float(r), done, info

    def reset(self) -> np.ndarray:
        conf = self._sample_conf()
        obs = self.actualEnv._reset(obstacle_position=conf[0], floor_height=conf[1])
        return np.expand_dims(obs, axis=0)

    def render(self, mode="human"):
        pass

    def close(self):
        self.actualEnv.close()


class BCDataset(Dataset):
    def __init__(self, x, y):
        super().__init__()
        assert x.shape[0] == y.shape[0]
        self.x = x
        self.y = y

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]


def generate_expert_trajectory(env):
    states, actions = [], []
    done = False
    obs = env.reset()
    obstacle_position = env.actualEnv.obstacle_position
    jumping_pixel = obstacle_position - 14
    step = 0
    while not done:
        action = 1 if step == jumping_pixel else 0
        next_obs, reward, done, info = env.step(action)
        assert bool(info['collision']) is False, "Trajectory not optimal!"
        states.append(obs)
        actions.append(action)
        obs = next_obs
        env.render()
        step += 1
    return np.array(states), np.array(actions)


def generate_bc_dataset(envs, batch_size, batch_count):
    """
    :param envs: the envs from which to create the dataset
    :param batch_size: size of each mini batch
    :param batch_count: how many batches the dataloader should contain
    """
    total_states, total_actions = [], []
    while len(total_states) < batch_size * batch_count:
        states, actions = generate_expert_trajectory(env)
        total_states += states
        total_actions += actions
    total_states = total_states[0:batch_size * batch_count + 1]
    total_actions = total_actions[0:batch_size * batch_count + 1]

    total_states, total_actions = np.array(total_states), np.array(total_actions)

    data: BCDataset = BCDataset(torch.tensor(total_states), torch.tensor(total_actions))

    trainloader: DataLoader = DataLoader(data, batch_size=batch_size, shuffle=True)
    return trainloader


if __name__ == '__main__':

    _envs = [VanillaEnv()]
    for _env in _envs:
        _obs_arr = [_env.reset(), _env.step(0)[0], _env.step(0)[0], _env.step(0)[0], _env.step(1)[0]]
        for _obs in _obs_arr:
            assert _obs.dtype == np.float32, "Incorrect datatype"
            assert _obs.shape == (1, 60, 60), "Incorrect shape"
            assert 0. in _obs, "No white pixels present"
            assert .5 in _obs, "No grey pixels present"
            assert 1. in _obs, "No black pixels present"
