"""
This module holds different modified jumping tasks environments
"""

import gym
import numpy as np
from gym import spaces
from typing import List

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
