import gym
import logging
from torch import optim
import matplotlib.pyplot as plt

from coinrun.env import VanillaEnv
from common.rl.ppo.ppo import PPO
from common.rl.buffer2 import RolloutBuffer
from common.rl.ppo.policies import ActorCriticNet
from rl.logger import ConsoleLogger, Tracker, WandBLogger

logging.basicConfig(level=logging.INFO)

policy: ActorCriticNet = ActorCriticNet(obs_space=(3, 64, 64), action_space=15, hidden_size=256)
optimizer = optim.Adam(policy.parameters(), lr=.001)
buffer = RolloutBuffer(capacity=2000, batch_size=256, min_transitions=2000)

logger1 = ConsoleLogger(log_every=100, average_over=100)
logger2 = WandBLogger(project='ppo-coin-run', info={'lr': 0.001})
tracker = Tracker(logger1, logger2)

env = VanillaEnv(start_level=0, num_levels=1000)

alg = PPO(policy, env, optimizer, seed=124, tracker=tracker, buffer=buffer, n_epochs=10)

logging.info("Training on " + alg.device)
alg.learn(30000)
alg.save('./runs/', 'ppo-1000')
alg.learn(30000)
alg.save('./runs/', 'ppo-1000')
