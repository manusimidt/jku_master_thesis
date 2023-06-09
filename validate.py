import numpy as np
import torch
from matplotlib.backends.backend_agg import FigureCanvasAgg
from common import map_conf_to_index, plot_evaluation_grid
from env import VanillaEnv, obstacle_pos, floor_height
from policy import ActorNet


def validate(model: ActorNet, device, train_configurations):
    grid = np.zeros((len(obstacle_pos), len(floor_height)))

    solved_counter, failed_counter = 0, 0

    for obs_pos_idx in range(len(obstacle_pos)):
        for floor_height_idx in range(len(floor_height)):
            curr_obs_pos = obstacle_pos[obs_pos_idx]
            curr_floor_height = floor_height[floor_height_idx]

            env = VanillaEnv([(curr_obs_pos, curr_floor_height), ])

            done = False
            obs = env.reset()
            info = {}
            episode_return = 0
            while not done:
                action_logits = model.forward(torch.FloatTensor(obs).unsqueeze(0).to(device), contrastive=False)
                action = torch.argmax(action_logits)
                obs, rewards, done, info = env.step(action.item())
                episode_return += rewards
            is_solved = not info['collision']

            if is_solved:
                grid[obs_pos_idx][floor_height_idx] += 1
                solved_counter += 1
            else:
                failed_counter += 1

    generalization_performance = round(solved_counter * 100 / (solved_counter + failed_counter))
    fig = plot_evaluation_grid(grid, map_conf_to_index(obstacle_pos, floor_height, train_configurations))
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    width, height = fig.get_size_inches() * fig.get_dpi()
    image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape((int(height), int(width), 3))
    return image, generalization_performance
