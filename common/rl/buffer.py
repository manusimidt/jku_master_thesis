import torch
import numpy as np
import random


class ReplayBuffer():
    def __init__(self, num_actions=2, memory_len=10000):
        self.memory_len = memory_len
        self.transition = []
        self.num_actions = num_actions

    def add(self, state, action, reward, next_state, done):
        if self.length() > self.memory_len:
            self.remove()
        self.transition.append([state, action, reward, next_state, done])

    def sample_batch(self, batch_size=32):
        minibatch = random.sample(self.transition, batch_size)
        states_mb, a_, reward_mb, next_states_mb, done_mb = map(np.array, zip(*minibatch))

        mb_reward = torch.from_numpy(reward_mb).type(torch.float32)
        mb_done = torch.from_numpy(done_mb.astype(int))
        a_ = a_.astype(int)
        a_mb = np.zeros((a_.size, self.num_actions), dtype=np.float32)
        a_mb[np.arange(a_.size), a_] = 1
        mb_a = torch.from_numpy(a_mb)
        return states_mb, mb_a, mb_reward, next_states_mb, mb_done  # states will be converted to tensors in forward pass

    def length(self):
        return len(self.transition)

    def remove(self):
        self.transition.pop(0)


if __name__ == '__main__':
    buffer = ReplayBuffer(2, memory_len=1000)
    for _ in range(1000):
        buffer.add(torch.rand((60, 60)), 1, 1, torch.rand((60, 60)), False)
    sample = buffer.sample_batch()
    print(len(sample))
    print("Buffer length: ", buffer.length())
