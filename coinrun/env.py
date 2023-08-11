import gym
import numpy as np


class VanillaEnv(gym.Env):
    def __init__(self, start_level=0xCAFEBEE, num_levels=1):
        self.actual_env = gym.make('procgen:procgen-coinrun-v0', start_level=start_level,
                                   num_levels=num_levels)

    def step(self, action):
        obs, r, done, info = self.actual_env.step(action)
        return np.array(np.moveaxis(obs, -1, -3) / 255, dtype=np.float32), r, done, info

    def reset(self, **kwargs):
        obs = self.actual_env.reset()
        return np.array(np.moveaxis(obs, -1, -3) / 255, dtype=np.float32)

    def close(self):
        self.actual_env.close()


if __name__ == '__main__':
    _env = VanillaEnv()

    _obs_arr = [_env.reset(), _env.step(0)[0], _env.step(0)[0], _env.step(0)[0], _env.step(1)[0]]
    for _obs in _obs_arr:
        assert _obs.dtype == np.float32, "Incorrect datatype"
        assert _obs.shape == (3, 64, 64), "Incorrect shape"



