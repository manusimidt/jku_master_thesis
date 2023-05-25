import torch
import numpy as np


class Transition(object):
    def __init__(self, state, action, reward, log_probs):
        self.state = state
        self.action = action
        self.reward = reward
        self.g_return = 0.0
        self.log_probs = log_probs

class AugmentedTransition(Transition):
    def __init__(self, state, augmented_state, action, reward, log_probs):
        super().__init__(state, action, reward, log_probs)
        self.augmented_state = augmented_state

class Episode(object):
    def __init__(self, discount):
        self.discount = discount
        self._empty()
        self.total_reward = 0.0

    def _empty(self):
        self.n = 0
        self.transitions = []

    def reset(self):
        self._empty()

    def size(self):
        return self.n

    def append(self, transition):
        self.transitions.append(transition)
        self.n += 1

    def states(self):
        return [s.state for s in self.transitions]

    def actions(self):
        return [a.action for a in self.transitions]

    def rewards(self):
        return [r.reward for r in self.transitions]

    def returns(self):
        return [r.g_return for r in self.transitions]

    def calculate_return(self):
        # turn rewards into return
        rewards = self.rewards()
        trajectory_len = len(rewards)
        return_array = torch.zeros((trajectory_len,))
        g_return = 0.
        for i in range(trajectory_len - 1, -1, -1):
            g_return = rewards[i] + self.discount * g_return
            return_array[i] = g_return
            self.transitions[i].g_return = g_return
        return return_array


class RolloutBuffer(object):
    def __init__(self, capacity=2000, batch_size=1000, min_transitions=2000):
        self.capacity = capacity
        self.batch_size = batch_size
        self.min_transitions = min_transitions
        self.buffer = []
        self._empty()
        self.mean_returns = []
        self.all_returns = []

    def _empty(self):
        del self.buffer[:]
        self.position = 0

    def add(self, episode):
        """Saves a transition."""
        episode.calculate_return()
        for t in episode.transitions:
            if len(self.buffer) < self.capacity:
                self.buffer.append(None)
            self.buffer[self.position] = t
            self.position = (self.position + 1) % self.capacity

    def update_stats(self):
        returns = [t.g_return for t in self.buffer]
        self.all_returns += returns
        mean_return = np.mean(np.array(returns))
        self.mean_returns += ([mean_return] * len(returns))

    def reset(self):
        self._empty()

    def sample(self):
        prob = [1 / len(self.buffer) for _ in range(0, len(self.buffer))]
        return np.random.choice(self.buffer, size=self.batch_size, p=prob, replace=False)

    def __len__(self):
        return len(self.buffer)
