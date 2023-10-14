from collections import defaultdict

import numpy as np
import torch
from crafter_rl.env import VanillaEnv
from crafter_rl.policy import ActorNet


def validate(model: ActorNet, device, semantic=False, iterations=10):
    returns = []
    steps = []
    achievements = defaultdict(lambda: [])
    inventory = defaultdict(lambda: [])

    # for seed in np.random.randint(low=0, high=np.iinfo(np.int32).max, size=iterations):
    for seed in np.random.randint(low=0, high=4, size=iterations):
        # needed to average over the entire episode
        curr_inventory = defaultdict(lambda: [])
        env = VanillaEnv(seed=seed, semantic=semantic)

        state = env.reset()
        done = False
        rewards = []
        step = 0

        while not done:
            input_state = torch.tensor(state).unsqueeze(dim=0).to(device)
            action = torch.argmax(model.forward(input_state, contrastive=False)).item()
            next_state, r, done, info = env.step(action)
            rewards.append(r)
            state = next_state
            step += 1
            if done:
                break

            for item_name, value in info['inventory'].items():
                curr_inventory[item_name].append(value)
        returns.append(np.mean(rewards))
        steps.append(step)
        # Add achievements to global list
        for achievement_name, value in info['achievements'].items():
            achievements[achievement_name].append(value)
        # Add inventory to global list
        for item_name, value in curr_inventory.items():
            inventory[item_name] = np.mean(value)

    achievements = dict(achievements)
    inventory = dict(inventory)

    # average everything
    for achievement_name, value in achievements.items():
        achievements[achievement_name] = np.mean(achievements[achievement_name])

    for item_name, value in inventory.items():
        inventory[item_name] = np.mean(inventory[item_name])

    return np.mean(returns), np.mean(steps), achievements, inventory


if __name__ == '__main__':
    model = ActorNet()
    import time

    start_time = time.time()
    returns, steps, achievements, inventory = validate(model, 'cpu')
    print(time.time() - start_time)
    print("Avg. Return", returns)
    print("Avg. Steps", steps)
    print(achievements)
    print(inventory)
