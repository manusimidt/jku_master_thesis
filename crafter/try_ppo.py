from torch import optim
from rl.common.logger import ConsoleLogger, TensorboardLogger, Tracker
from rl.ppo.policies import ActorCriticNet
from rl.ppo.ppo import PPO
from crafter.env import VanillaEnv

env = VanillaEnv()

run_name = 'test'

policy: ActorCriticNet = ActorCriticNet(obs_space=(3, 64, 64), action_space=env.num_actions, hidden_size=256)
optimizer = optim.Adam(policy.parameters(), lr=0.001)

logger1 = ConsoleLogger(log_every=100, average_over=100)
logger2 = TensorboardLogger('./tensorboard2', run_id=run_name)
tracker = Tracker(logger1, logger2)

ppo = PPO(policy, env, optimizer, seed=31, tracker=tracker)
print("Training on ", ppo.device)
ppo.load('./ckpts/BC test.pth')
ppo.learn(15_000)
ppo.save('./ckpts', run_name + '-15000')
ppo.learn(15_000)
ppo.save('./ckpts', run_name + '-30000')

