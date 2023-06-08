"""
This module holds different modified jumping tasks environments
"""
import random

import gym
import numpy as np
import torch
from gym import spaces
from typing import List

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as fn
from torchvision.transforms.functional import InterpolationMode
import augmentations
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

POSSIBLE_AUGMENTATIONS = [
    {'name': 'trans64', 'func': augmentations.random_translate, 'params': {'size': 64}},
    {'name': 'trans68', 'func': augmentations.random_translate, 'params': {'size': 68}},
    {'name': 'trans72', 'func': augmentations.random_translate, 'params': {'size': 72}},
    {'name': 'crop59', 'func': augmentations.random_crop, 'params': {'out': 59}},
    {'name': 'crop58', 'func': augmentations.random_crop, 'params': {'out': 58}},
    {'name': 'crop57', 'func': augmentations.random_crop, 'params': {'out': 57}},
    {'name': 'cut5', 'func': augmentations.random_cutout, 'params': {'min_cut': 2, 'max_cut': 5}},
    {'name': 'cut15', 'func': augmentations.random_cutout, 'params': {'min_cut': 5, 'max_cut': 15}},
    {'name': 'cut20', 'func': augmentations.random_cutout, 'params': {'min_cut': 10, 'max_cut': 20}},
    {'name': 'blur1', 'func': augmentations.gaussian_blur, 'params': {'sigma': .6}},
    {'name': 'blur2', 'func': augmentations.gaussian_blur, 'params': {'sigma': 1.2}},
    {'name': 'noise1', 'func': augmentations.random_noise, 'params': {'strength': .02}},
    {'name': 'noise2', 'func': augmentations.random_noise, 'params': {'strength': .05}},
    {'name': 'flip', 'func': augmentations.random_flip, 'params': {}},
]

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


class AugmentingEnv(VanillaEnv):
    """Custom Environment that follows gym interface."""
    metadata = {"render.modes": ["human"]}

    def __init__(self, configurations: List[tuple] or None = None, rendering=False):
        """
        :param configurations: possible configurations, array of tuples consisting of
            the obstacle position and the floor height
        """
        super().__init__(configurations, rendering)
        self.current_augmentation = None

    def step(self, action):
        aug_obs, _, r, done, info = self.step_augmented(action)
        return aug_obs, r, done, info

    def step_augmented(self, action):
        # returns both the augmented state and the not augmented state
        obs, r, done, info = super().step(action)
        return self._augment(obs), obs, r, done, info

    def reset(self):
        aug_obs, _ = self.reset_augmented()
        return aug_obs

    def reset_augmented(self):
        obs = super().reset()
        # sample a new augmentation
        self._sample_augmentation()
        aug_obs = self._augment(obs)
        return aug_obs, obs

    def _sample_augmentation(self):
        idx = np.random.choice(range(len(POSSIBLE_AUGMENTATIONS)))
        self.current_augmentation = POSSIBLE_AUGMENTATIONS[idx]

    def _augment(self, obs):
        augmentation = self.current_augmentation
        # convert the observation in the needed format (B x C x H x W)
        aug_obs = np.expand_dims(obs, axis=1)
        # augment the observation
        aug_obs = augmentation['func'](aug_obs, **augmentation['params'])
        aug_obs = aug_obs.squeeze(axis=0)
        # The augmented observation can have a different width and height!!
        # compensate for that
        if not obs.shape == aug_obs.shape:
            aug_obs = fn.resize(torch.from_numpy(aug_obs), size=[60, 60],
                                interpolation=InterpolationMode.NEAREST).numpy()
        return aug_obs

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


def generate_expert_episode(env, numpy=True):
    """ Generates a single expert trajectory """
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
    if numpy:
        return np.array(states), np.array(actions)
    else:
        return states, actions


def generate_bc_data(envs, total_size, balanced=False):
    """
    Generates an array of states and their corresponding optimal action
    :param envs: the envs from which to create the dataset
    :param total_size: size of the dataset
    :param balanced: If true the resulting dataset will have 33% jump actions and 66% non-jump actions
    """
    total_states, total_actions = [], []
    while len(total_states) < total_size:
        env = random.sample(envs, 1)[0]
        states, actions = generate_expert_episode(env, numpy=False)

        if not balanced:
            total_states += states
            total_actions += actions
        else:
            # balance the dataset. Always include the state that has the jump action and two non-jumping
            jump_idx = np.argmax(actions)
            total_states.append(states[jump_idx])
            total_actions.append(actions[jump_idx])
            non_jump_indices = np.random.choice(np.delete(np.arange(0, len(actions)), jump_idx), 2)
            total_states += [states[non_jump_indices[0]], states[non_jump_indices[1]]]
            total_actions += [actions[non_jump_indices[0]], actions[non_jump_indices[1]]]

    total_states = total_states[0:total_size]
    total_actions = total_actions[0:total_size]

    return np.array(total_states), np.array(total_actions)


def generate_bc_dataset(envs, batch_size, batch_count, balanced=False):
    """
    Generates a dataset of length batch_size * batch_count
    containing states and their corresponding optimal action
    """
    states, actions = generate_bc_data(envs, batch_size * batch_count, balanced)
    states, actions = torch.tensor(states), torch.tensor(actions)
    data: BCDataset = BCDataset(states, actions)
    train_set_length = int(len(data) * 0.8)
    train_set, val_set = torch.utils.data.random_split(data, [train_set_length, len(data) - train_set_length])
    train_loader: DataLoader = DataLoader(train_set, batch_size=64, shuffle=True)
    test_loader: DataLoader = DataLoader(val_set, batch_size=64, shuffle=True)
    return train_loader, test_loader


def generate_positive_pairs(envs: [VanillaEnv]):
    env1, env2 = random.sample(envs, 2)
    assert len(env1.configurations) == len(env2.configurations) == 1, "Only works with deterministic Vanilla Envs!"
    obstacle_pos_1 = env1.configurations[0][0]
    obstacle_pos_2 = env2.configurations[0][0]
    diff = obstacle_pos_1 - obstacle_pos_2

    if diff > 0: return generate_positive_pairs([env2, env1])

    state1, action1 = generate_expert_episode(env1, numpy=True)
    state2, action2 = generate_expert_episode(env2, numpy=True)

    state2 = state2[-diff:]
    state1 = state1[:len(state2)]

    assert len(state1) == len(state2), "Some error in the code above"
    # sample a random index along the common frames
    idx = np.random.randint(0, len(state2), size=1)[0]
    return state1[idx], state2[idx]

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
