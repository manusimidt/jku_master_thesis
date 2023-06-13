import random
import numpy as np
import torch
import matplotlib.pyplot as plt


def dict2mdtable(d, key='Name', val='Value'):
    rows = [f'| {key} | {val} |']
    rows += ['|--|--|']
    rows += [f'| {k} | {v} |' for k, v in d.items()]
    return "  \n".join(rows)


def plot_evaluation_grid(grid, training_positions):
    """Plots the evaluation grid."""
    fig, ax = plt.subplots(figsize=(8, 4))
    grid_x, grid_y = grid.shape
    extent = (0, grid_x, grid_y, 0)
    ax.imshow(grid.T, extent=extent, origin='lower', cmap='copper')

    x_ticks = np.arange(grid_x)
    y_ticks = np.arange(grid_y)
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)

    ax.set_ylabel("Floor height")
    ax.set_xlabel("Obstacle position")

    # Loop over data dimensions and create text annotations.
    for (obstacle_pos, floor_height) in training_positions:
        pos_index = obstacle_pos
        height_index = floor_height
        ax.text(
            pos_index + 0.5,
            height_index + 0.5,
            'T',
            ha='center',
            va='center',
            color='r',
            fontsize='large',
            fontweight='bold')

    ax.grid(color='w', linewidth=1)
    fig.tight_layout()
    return fig


def map_conf_to_index(obstacle_pos, floor_height, confs: list) -> list:
    """
    Takes a list of configurations (i.e.: [(14, 0), (25, 20)]) and returns the
    indices that correspond to the grid (i.e.: [(0, 0), (8,5)])
    """
    res = []
    for conf in confs:
        x = np.where(obstacle_pos == conf[0])[0]
        y = np.where(floor_height == conf[1])[0]
        assert len(x) == 1 and len(y) == 1, "Make sure the configurations are defined equally while training/evaluating"
        res.append((x[0], y[0]))
    return res


def set_seed(seed, env, force=False):
    random.seed(seed)
    np.random.seed(seed)
    if env: env.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        if force: torch.use_deterministic_algorithms(True)
