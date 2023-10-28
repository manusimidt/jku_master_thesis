"""
Provides different PSMs
"""
import torch


def _calculate_action_cost_matrix(actions_1, actions_2):
    action_equality = torch.eq(actions_1.unsqueeze(1), actions_2.unsqueeze(0))
    return 1.0 - action_equality.float()


def psm_default(x_arr, y_arr, gamma=0.999, window=1e9):
    """
    Slow but intuitive implementation of PSM
    """
    storage = torch.full(size=(len(x_arr), len(y_arr)), fill_value=-1.0).to(x_arr.device)

    def psm_dyn(x_idx, y_idx, curr_window):
        tv = 0. if x_arr[x_idx] == y_arr[y_idx] else 1.
        if (x_idx == len(x_arr) - 1 and y_idx == len(y_arr) - 1) or curr_window == 0:
            return tv
        else:
            next_x_idx = min(x_idx + 1, len(x_arr) - 1)
            next_y_idx = min(y_idx + 1, len(y_arr) - 1)
            next_psm = psm_dyn(next_x_idx, next_y_idx, curr_window - 1) if storage[next_x_idx, next_y_idx] == -1 else \
                storage[next_x_idx, next_y_idx]
            return tv + gamma * next_psm

    for i in range(len(x_arr)):
        for j in range(len(y_arr)):
            storage[i, j] = psm_dyn(i, j, window)
    return storage


def psm_fb(x_arr, y_arr, gamma=0.99, window=1e9):
    """
    Slow but intuitive implementation of the PSM-FB
    """
    storage_fwrd = torch.full(size=(len(x_arr), len(y_arr)), fill_value=-1.0).to(x_arr.device)
    storage_bwrd = torch.full(size=(len(x_arr), len(y_arr)), fill_value=-1.0).to(x_arr.device)

    def psm2_dyn_forward(x_idx, y_idx, curr_window):
        tv = 0. if x_arr[x_idx] == y_arr[y_idx] else 1.
        if (x_idx == len(x_arr) - 1 and y_idx == len(y_arr) - 1) or curr_window == 0:
            return tv
        else:
            next_x_idx = min(x_idx + 1, len(x_arr) - 1)
            next_y_idx = min(y_idx + 1, len(y_arr) - 1)

            next_psm = psm2_dyn_forward(next_x_idx, next_y_idx, curr_window - 1) \
                if storage_fwrd[next_x_idx, next_y_idx] == -1 else storage_fwrd[next_x_idx, next_y_idx]
            return tv + gamma * next_psm

    def psm2_dyn_backward(x_idx, y_idx, curr_window):
        tv = 0. if x_arr[x_idx] == y_arr[y_idx] else 1.
        if (x_idx == 0 and y_idx == 0) or curr_window == 0:
            return tv
        else:
            past_x_idx = max(x_idx - 1, 0)
            past_y_idx = max(y_idx - 1, 0)

            past_psm = psm2_dyn_backward(past_x_idx, past_y_idx, curr_window - 1) \
                if storage_bwrd[past_x_idx, past_y_idx] == -1 else storage_bwrd[past_x_idx, past_y_idx]
            return tv + gamma * past_psm

    for i in range(len(x_arr)):
        for j in range(len(y_arr)):
            storage_fwrd[i, j] = psm2_dyn_forward(i, j, window)
            storage_bwrd[i, j] = psm2_dyn_backward(i, j, window)
    return storage_fwrd + storage_bwrd


def psm_f_fast(actions1, actions2, gamma=0.9, eps=1e-7):
    """
    Taken from the PAPER Agarwal et al. 2021
    """
    # matrix that holds the TV for each element of the two arrays
    # the entry i,j is 1 if the i-th entry of actions1 does NOT equal to the j-th entry of action 2
    action_cost = _calculate_action_cost_matrix(actions1, actions2)

    d = torch.zeros_like(action_cost)

    def operator(d_cur):
        d_new = 1 * action_cost
        discounted_d_cur = gamma * d_cur
        d_new[:-1, :-1] += discounted_d_cur[1:, 1:]
        d_new[:-1, -1] += discounted_d_cur[1:, -1]
        d_new[-1, :-1] += discounted_d_cur[-1, 1:]
        return d_new

    while True:
        d_new = operator(d)
        if torch.sum(torch.abs(d - d_new)) < eps:
            break
        else:
            d = d_new[:]
    return d


def psm_f_fast_repo(actions1, actions2, gamma=0.999, eps=1e-7):
    """
    Taken from https://github.com/google-research/google-research/tree/master/pse/jumping_task
    """
    # matrix that holds the TV for each element of the two arrays
    # the entry i,j is 1 if the i-th entry of actions1 does NOT equal to the j-th entry of action 2
    action_cost = _calculate_action_cost_matrix(actions1, actions2)

    n, m = action_cost.shape
    d_metric = torch.zeros_like(action_cost)

    def fixed_point_operator(d_metric):
        d_metric_new = torch.empty_like(d_metric)
        for i in range(n):
            for j in range(m):
                d_metric_new[i, j] = action_cost[i, j] + \
                                     gamma * d_metric[min(i + 1, n - 1), min(j + 1, m - 1)]
        return d_metric_new

    while True:
        d_metric_new = fixed_point_operator(d_metric)
        if torch.sum(torch.abs(d_metric - d_metric_new)) < eps:
            break
        else:
            d_metric = d_metric_new
    return d_metric


def psm_fb_fast(actions1, actions2, gamma=0.9, eps=1e-7):
    """
    Efficient implementation of PSE-FB (inspired by Agarwal et al. 2021)
    """
    action_cost = _calculate_action_cost_matrix(actions1, actions2)
    d_bwrd = torch.zeros_like(action_cost)
    d_fwrd = torch.zeros_like(action_cost)

    def operator_bwrd(d_cur):
        d_new = 1 * action_cost
        discounted_d_cur = gamma * d_cur
        d_new[:-1, :-1] += discounted_d_cur[1:, 1:]
        d_new[:-1, -1] += discounted_d_cur[1:, -1]
        d_new[-1, :-1] += discounted_d_cur[-1, 1:]
        return d_new

    def operator_fwrd(d_cur):
        d_new = 1 * action_cost
        discounted_d_cur = gamma * d_cur
        d_new[1:, 1:] += discounted_d_cur[:-1, :-1]
        d_new[1:, 0] += discounted_d_cur[:-1, 0]
        d_new[0, 1:] += discounted_d_cur[0, :-1]
        return d_new

    while True:
        d_new_b = operator_bwrd(d_bwrd)
        d_new_f = operator_fwrd(d_fwrd)
        if torch.sum(torch.abs(d_bwrd - d_new_b)) < eps and torch.sum(torch.abs(d_fwrd - d_new_f)) < eps:
            break
        else:
            d_bwrd = d_new_b[:]
            d_fwrd = d_new_f[:]
    return d_bwrd + d_fwrd


if __name__ == '__main__':
    import time

    Mx = torch.tensor([0, 0, 0, 1, 0])
    My = torch.tensor([0, 1, 0, 0, 0])

    Mx = torch.randint(low=0, high=17, size=(200,))
    My = torch.randint(low=0, high=17, size=(200,))

    start_time = time.time()
    result1 = psm_default(Mx, My, gamma=0.8)
    result1_time = time.time() - start_time

    start_time = time.time()
    result2 = psm_f_fast_repo(Mx, My, gamma=0.8)
    result2_time = time.time() - start_time

    # assert torch.allclose(result1, result2)
    print(result1)

    start_time = time.time()
    result3 = psm_fb(Mx, My, gamma=0.8)
    result3_time = time.time() - start_time

    start_time = time.time()
    result4 = psm_fb_fast(Mx, My, gamma=0.8)
    result4_time = time.time() - start_time

    assert torch.allclose(result3, result4)
    print(result2)

    print("Absolute Timings")
    print(f"PSE Forward Timings: {result1_time:.4f} sec,  {result2_time:.4f} sec")
    print(f"PSE FB Timings: {result3_time:.4f} sec,  {result4_time:.4f} sec")

    print(f"Relative Timings:")
    print(f"PSE Forward Timings: {result1_time / Mx.size()[0]:.4f},  {result2_time / Mx.size()[0]:.4f}")
    print(f"PSE FB Timings: {result3_time / Mx.size()[0]:.4f},  {result4_time / Mx.size()[0]:.4f}")
