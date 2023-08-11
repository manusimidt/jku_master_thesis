import torch
import numpy as np
from coinrun.env import VanillaEnv
from common.rl.ppo.policies import ActorCriticNet


def validate(model, start_level=0xABC, num_levels=0, iterations=100, record_optimal=False):
    env = VanillaEnv(start_level=start_level, num_levels=num_levels)

    avg_reward = []
    avg_iterations = []

    for _ in range(iterations):
        cum_reward = 0  # cumulative reward
        num_iterations = 0
        states = []
        actions = []
        obs = env.reset()
        while True:
            action_logits = model.actor.forward(torch.FloatTensor(obs).unsqueeze(0))
            action = torch.argmax(action_logits)
            obs, rew, done, info = env.step(action)

            num_iterations += 1
            cum_reward += rew
            if done:
                if record_optimal and rew == 10.0:
                    np.savez(f'./dataset/{info["level_seed"]}.npz', state=np.array(states), action=np.array(actions))
                break
        avg_reward.append(cum_reward)
        avg_iterations.append(num_iterations)
    return np.mean(avg_reward), np.mean(avg_iterations)


if __name__ == '__main__':
    _model = ActorCriticNet(obs_space=(3, 64, 64), action_space=15, hidden_size=256)
    ckp = torch.load('./runs/ppo-test-2.pth')
    _model.load_state_dict(ckp['state_dict'])

    print(validate(_model, start_level=0xCAFE, num_levels=20, record_optimal=True))
