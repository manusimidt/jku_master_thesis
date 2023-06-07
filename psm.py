"""
Provides different PSMs
"""

import numpy as np
import torch


def _metric_fixed_point_fast(cost_matrix, gamma, eps=1e-7):
    """Dynamic programming for calculating PSM."""
    d = np.zeros_like(cost_matrix)

    def operator(d_cur):
        d_new = 1 * cost_matrix
        discounted_d_cur = gamma * d_cur
        d_new[:-1, :-1] += discounted_d_cur[1:, 1:]
        d_new[:-1, -1] += discounted_d_cur[1:, -1]
        d_new[-1, :-1] += discounted_d_cur[-1, 1:]
        return d_new

    while True:
        d_new = operator(d)
        if np.sum(np.abs(d - d_new)) < eps:
            break
        else:
            d = d_new[:]
    return d


def _calculate_action_cost_matrix(actions_1, actions_2):
    actions_1 = torch.tensor(actions_1)  # Assuming actions_1 is a NumPy array or a tensor
    actions_2 = torch.tensor(actions_2)  # Assuming actions_2 is a NumPy array or a tensor

    action_equality = torch.eq(
        actions_1.unsqueeze(1), actions_2.unsqueeze(0))
    return 1.0 - action_equality.float()


def psm_paper(actions1, actions2, gamma=0.99):
    """Taken from Agarwal et.al"""
    action_cost = _calculate_action_cost_matrix(actions1, actions2)
    return _metric_fixed_point_fast(np.array(action_cost), gamma=gamma)


def psm_default(x_arr, y_arr, gamma=0.99):
    storage = np.full(shape=(len(x_arr), len(y_arr)), fill_value=-1.0)

    def psm_dyn(x_idx, y_idx):
        tv = 0. if x_arr[x_idx] == y_arr[y_idx] else 1.
        if x_idx == len(x_arr) - 1 and y_idx == len(y_arr) - 1:
            return tv
        else:
            next_x_idx = min(x_idx + 1, len(x_arr) - 1)
            next_y_idx = min(y_idx + 1, len(y_arr) - 1)
            next_psm = psm_dyn(next_x_idx, next_y_idx) if storage[next_x_idx, next_y_idx] == -1 else storage[
                next_x_idx, next_y_idx]
            return tv + gamma * next_psm

    for i in range(len(x_arr)):
        for j in range(len(y_arr)):
            storage[i, j] = psm_dyn(i, j)
    return storage


def psm_fb(x_arr, y_arr, gamma_forward=0.99, gamma_backward=0.99):
    """ PSM Forward Backward"""
    storage_fwrd = np.full(shape=(len(x_arr), len(y_arr)), fill_value=-1.0)
    storage_bwrd = np.full(shape=(len(x_arr), len(y_arr)), fill_value=-1.0)

    def psm2_dyn_forward(x_idx, y_idx):
        tv = 0. if x_arr[x_idx] == y_arr[y_idx] else 1.
        if x_idx == len(x_arr) - 1 and y_idx == len(y_arr) - 1:
            return tv
        else:
            next_x_idx = min(x_idx + 1, len(x_arr) - 1)
            next_y_idx = min(y_idx + 1, len(y_arr) - 1)

            next_psm = psm2_dyn_forward(next_x_idx, next_y_idx) if storage_fwrd[next_x_idx, next_y_idx] == -1 else \
                storage_fwrd[next_x_idx, next_y_idx]
            return tv + gamma_forward * next_psm

    def psm2_dyn_backward(x_idx, y_idx):
        tv = 0. if x_arr[x_idx] == y_arr[y_idx] else 1.
        if x_idx == 0 and y_idx == 0:
            return tv
        else:
            past_x_idx = max(x_idx - 1, 0)
            past_y_idx = max(y_idx - 1, 0)

            past_psm = psm2_dyn_backward(past_x_idx, past_y_idx) if storage_bwrd[past_x_idx, past_y_idx] == -1 else \
                storage_bwrd[past_x_idx, past_y_idx]
            return tv + gamma_backward * past_psm

    for i in range(len(x_arr)):
        for j in range(len(y_arr)):
            storage_fwrd[i, j] = psm2_dyn_forward(i, j)
            storage_bwrd[i, j] = psm2_dyn_backward(i, j)
    return storage_fwrd + storage_bwrd
