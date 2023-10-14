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

    def __init__(self, seed, semantic=False):
        super().__init__()
        self.semantic = semantic
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
        if self.semantic:
            local_semantic = create_local_semantic(
                info['semantic'], info['player_pos'][0], info['player_pos'][1],
                info['inventory']['health'], info['inventory']['food'],
                info['inventory']['drink'], info['inventory']['energy'],
                info['inventory']['sapling'], info['inventory']['wood'],
                info['inventory']['stone'], info['inventory']['coal'],
                info['inventory']['iron'], info['inventory']['diamond'],
                info['inventory']['wood_pickaxe'], info['inventory']['stone_pickaxe'],
                info['inventory']['iron_pickaxe'], info['inventory']['wood_sword'],
                info['inventory']['stone_sword'], info['inventory']['iron_sword'],
            )
            return local_semantic.astype(np.float32).reshape((1, 9, 9)), float(r), done, info
        else:
            return np.array(np.moveaxis(obs, -1, -3) / 255, dtype=np.float32), float(r), done, info

    def reset(self) -> np.ndarray:
        obs = self.actualEnv.reset()
        if self.semantic:
            return np.zeros((1, 9, 9), dtype=np.float32)
        else:
            return np.array(np.moveaxis(obs, -1, -3) / 255, dtype=np.float32)

    def render(self, mode="human"):
        pass

    def close(self):
        self.actualEnv.close()


def create_local_semantic(global_semantic, player_x, player_y, health, food, water, energy, sapling, wood, stone, coal,
                          iron, diamond, wood_pix, stone_pix, iron_pix, wood_swo, stone_swo, iron_swo):
    global_semantic = np.pad(global_semantic, (5, 5), constant_values=(0, 0))
    player_obs = global_semantic[player_x - 4 + 5:player_x + 5 + 5, player_y - 3 + 5:player_y + 4 + 5]
    inventory = np.array([health, food, water, energy,
                          sapling, wood, stone, coal,
                          iron, diamond, wood_pix, stone_pix,
                          iron_pix, wood_swo, stone_swo, iron_swo, 0, 0]).reshape(2, 9, 1)
    player_obs = np.append(player_obs, inventory[0], 1)
    player_obs = np.append(player_obs, inventory[1], 1)

    mean = 3.04432
    std_dev = 2.95620
    return (player_obs - mean) / std_dev

    # return player_obs / 18


class CrafterReplayBuffer:
    def __init__(self, device, seed, data_dir, sematic=False):
        self.device = device
        self.seed = seed
        self.data_dir = data_dir
        self.rng = np.random.default_rng(seed=seed)
        self.semantic = sematic
        self.buffer_size = 62002 - 100 if sematic else 62002  # Size of the imitation dataset i downloaded
        self.actions = torch.empty(self.buffer_size, device=device, dtype=torch.int64)
        self.episode_start_idx = []

        if not sematic:
            self.states = torch.empty((self.buffer_size, 3, 64, 64), device=device, dtype=torch.float32)
            self._populate_image()
        else:
            self.states = torch.empty((self.buffer_size, 1, 9, 9), device=device, dtype=torch.float32)
            self._populate_semantic()

    def _populate_image(self):
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

    def _populate_semantic(self):
        i = 0
        for file in os.listdir(self.data_dir):
            with np.load(self.data_dir + os.sep + file) as data:
                self.episode_start_idx.append(i)
                for semantic, action in zip(data['local_semantics'], data['actions']):
                    self.states[i] = torch.tensor(semantic, dtype=torch.float32, device=self.device).unsqueeze(dim=0)
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


def simplify_actions(actions: torch.tensor) -> torch.tensor:
    """
    Simplifies the action space for calculating the PSE
    0 Noop                  => 0 Nothing
    1 Move Left             => 1 Move
    2 Move Right            => 1 Move
    3 Move Up               => 1 Move
    4 Move Down             => 1 Move
    5 Do                    => 2 Do
    6 Sleep                 => 3 Sleep
    7 Place Stone           => 4 Place Block
    8 Place Table           => 4 Place Block
    9 Place Furnace         => 4 Place Block
    10 Place Plant          => 5 Place Plant
    11 Make Wood Pickaxe    => 6 Make Pickaxe
    12 Make Stone Pickaxe   => 6 Make Pickaxe
    13 Make Iron Pickaxe    => 6 Make Pickaxe
    14 Make Wood Sword      => 7 Make Weapon
    15 Make Stone Sword     => 7 Make Weapon
    16 Make Iron Sword      => 7 Make Weapon
    """
    remap_dict = {0: 0, 1: 1, 2: 1, 3: 1, 4: 1, 5: 2, 6: 3, 7: 4, 8: 4, 9: 4, 10: 5, 11: 6, 12: 6, 13: 6, 14: 7,
                  15: 7, 16: 7}
    remap_function = np.vectorize(lambda x: remap_dict.get(x, x))
    return torch.from_numpy(remap_function(actions.cpu())).to(actions.device)


if __name__ == '__main__':
    buffer = CrafterReplayBuffer('cuda', 5, './dataset')
    _states, _actions = buffer.sample(batch_size=23)
    print(f"States shape {_states.shape}, device: {_states.device}, actions shape {_actions.shape}")
    _states, _actions = buffer.sample_trajectory()
    print(f"States shape {_states.shape}, device: {_states.device}, actions shape {_actions.shape}")

    buffer = CrafterReplayBuffer('cuda', 5, './dataset-semantic', sematic=True)
    _states, _actions = buffer.sample(batch_size=23)
    print(f"States shape {_states.shape}, device: {_states.device}, actions shape {_actions.shape}")
    _states, _actions = buffer.sample_trajectory()
    print(f"States shape {_states.shape}, device: {_states.device}, actions shape {_actions.shape}")

    simplify_actions(_actions)

    _env = VanillaEnv(seed=2)
    _obs_arr = [_env.reset(), _env.step(0)[0], _env.step(0)[0], _env.step(0)[0], _env.step(1)[0]]
    for _obs in _obs_arr:
        assert _obs.dtype == np.float32, "Incorrect datatype"
        assert _obs.shape == (3, 64, 64), "Incorrect shape"

    _env = VanillaEnv(seed=2, semantic=True)
    _obs_arr = [_env.reset(), _env.step(0)[0], _env.step(0)[0], _env.step(0)[0], _env.step(1)[0]]
    for _obs in _obs_arr:
        assert _obs.dtype == np.float32, "Incorrect datatype"
        assert _obs.shape == (1, 9, 9), "Incorrect shape"
