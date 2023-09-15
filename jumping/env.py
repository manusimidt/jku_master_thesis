"""
This module holds different modified jumping tasks environments
"""
import itertools
import random

import gym
import numpy as np
import torch
from gym import spaces
from typing import List
from jumping.gym_jumping_task.envs import JumpTaskEnv

n_obstacle_pos = 26  # how many obstacle position you want to try out (paper: 27, max: 30)
n_floor_heights = 11  # how many floor heights you want to try out (paper: 11, max: 40)
obstacle_pos = np.rint(np.linspace(20, 45, n_obstacle_pos)).astype(np.int8)
floor_height = np.rint(np.linspace(10, 20, n_floor_heights)).astype(np.int8)

TRAIN_CONFIGURATIONS = {
    "wide_grid": {
        # (obstacle_pos, floor_height)
        (20, 10), (25, 10), (30, 10), (35, 10), (40, 10), (45, 10),
        (20, 15), (25, 15), (30, 15), (35, 15), (40, 15), (45, 15),
        (20, 20), (25, 20), (30, 20), (35, 20), (40, 20), (45, 20),
    },
    "narrow_grid": {
        # (obstacle_pos, floor_height)
        (28, 13), (30, 13), (32, 13), (34, 13), (36, 13), (38, 13),
        (28, 15), (30, 15), (32, 15), (34, 15), (36, 15), (38, 15),
        (28, 17), (30, 17), (32, 17), (34, 17), (36, 17), (38, 17),
    }
}


def gen_rand_grid():
    return set(itertools.product(np.random.choice(obstacle_pos, size=18), np.random.choice(floor_height, size=18)))


POSSIBLE_CONFIGURATIONS = set(itertools.product(obstacle_pos, floor_height))

for _key in TRAIN_CONFIGURATIONS:
    for _conf in TRAIN_CONFIGURATIONS[_key]:
        assert _conf in POSSIBLE_CONFIGURATIONS, "Invalid Train configuration!"


class VanillaEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    # min_obstacle_pos = 14
    # max_obstacle_pos = 47
    # min_floor_height = 0
    # max_floor_height = 48

    def __init__(self, configurations: List[tuple] or None = None, rendering=False):
        """
        :param configurations: possible configurations, array of tuples consisting of
            the obstacle position and the floor height
        """
        super().__init__()
        # If no configuration was provided, use the default JumpingTask configuration
        if configurations is None:
            configurations = [(30, 10), ]
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

    def reset(self, **kwargs) -> np.ndarray:
        conf = self._sample_conf()
        obs = self.actualEnv._reset(obstacle_position=conf[0], floor_height=conf[1])
        return np.expand_dims(obs, axis=0)

    def render(self, mode="human"):
        pass

    def close(self):
        self.actualEnv.close()


class JumpingExpertBuffer:
    """ A simple BC replay buffer to store episodes (without storing next_state) """

    def __init__(self, training_pos: [tuple], device, seed):
        self.training_pos = training_pos
        self.device = device
        self.seed = seed
        self.buffer_size = len(training_pos) * 56  # There are 56 steps in an optimal episode
        self.rng = np.random.default_rng(seed=seed)

        self.states = torch.empty((self.buffer_size, 1, 60, 60), device=device, dtype=torch.float32)
        self.actions = torch.empty(self.buffer_size, device=device, dtype=torch.int64)

        # Stores the indices of the jumping states/actions. For faster sampling
        self.jump_idx = []
        self._populate()

    def _populate(self):
        i = 0
        env = JumpTaskEnv()

        for pos in self.training_pos:
            jumping_pixel = pos[0] - 14

            done = False
            obs = env._reset(obstacle_position=pos[0], floor_height=pos[1])
            step = 0

            while not done:
                action = 1 if step == jumping_pixel else 0
                next_obs, reward, done, info = env.step(action)
                assert bool(info['collision']) is False, "Trajectory not optimal!"

                self.states[i] = torch.from_numpy(obs).to(self.device).unsqueeze(0)
                self.actions[i] = action
                if action == 1: self.jump_idx.append(i)
                i += 1

                obs = next_obs
                step += 1

            assert step == 56, "Non expert episode!"
        assert i == self.buffer_size, "Buffer not correctly filled up"

    def sample(self, batch_size, replace=False, balanced=False) -> tuple:
        """
        Samples from the generated expert trajectories
        :param batch_size:
        :param replace: if replace is set to true, a batch can contain the same state twice
        :param balanced: if balanced is set to true, a batch will contain the same amount of jumping and non-jumping states
        """
        if not balanced:
            ind = self.rng.choice(range(self.buffer_size), size=batch_size, replace=replace)
        else:
            ind_jumping = self.rng.choice(self.jump_idx, size=int(batch_size / 2), replace=True)
            non_jump_idx = list(set(range(self.buffer_size)) - set(self.jump_idx))
            ind_non_jumping = self.rng.choice(non_jump_idx, size=batch_size - len(ind_jumping), replace=replace)
            ind = np.concatenate([ind_jumping, ind_non_jumping])
            self.rng.shuffle(ind)

        return self.states[ind], self.actions[ind]

    def sample_trajectory(self) -> tuple:
        """
        Samples an entire expert trajectory
        """
        idx = self.rng.choice(range(len(self.training_pos))) * 56
        return self.states[idx:idx + 56], self.actions[idx:idx + 56]


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


def generate_positive_pairs(envs: [VanillaEnv]):
    env1, env2 = random.sample(envs, 2)
    assert len(env1.configurations) == len(env2.configurations) == 1, "Only works with deterministic Vanilla Envs!"
    obstacle_pos_1 = env1.configurations[0][0]
    obstacle_pos_2 = env2.configurations[0][0]
    diff = obstacle_pos_1 - obstacle_pos_2

    if diff > 0:
        return generate_positive_pairs([env2, env1])

    state1, action1 = generate_expert_episode(env1, numpy=True)
    state2, action2 = generate_expert_episode(env2, numpy=True)

    state2 = state2[-diff:]
    state1 = state1[:len(state2)]

    assert len(state1) == len(state2), "Some error in the code above"
    # sample a random index along the common frames
    idx = np.random.randint(0, len(state2), size=1)[0]
    return state1[idx], state2[idx]


if __name__ == '__main__':
    buffer = JumpingExpertBuffer(TRAIN_CONFIGURATIONS['wide_grid'], 'cuda', 3)
    _states, _actions = buffer.sample(256, balance=True)
    buffer.sample_trajectory()
    _envs = [VanillaEnv()]
    for _env in _envs:
        _obs_arr = [_env.reset(), _env.step(0)[0], _env.step(0)[0], _env.step(0)[0], _env.step(1)[0]]
        for _obs in _obs_arr:
            assert _obs.dtype == np.float32, "Incorrect datatype"
            assert _obs.shape == (1, 60, 60), "Incorrect shape"
            assert 0. in _obs, "No white pixels present"
            assert .5 in _obs, "No grey pixels present"
            assert 1. in _obs, "No black pixels present"
