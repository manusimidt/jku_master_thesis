import gym
import torch
import numpy as np
from copy import copy
from rl.dqn.policies import DQNPolicy
from rl.common.buffer import ReplayBuffer
from rl.common.utils import soft_update


class DQN():

    def __init__(self,
                 policy: DQNPolicy,
                 env: gym.Env,
                 optimizer: torch.optim.Optimizer,

                 metric=torch.nn.MSELoss(),
                 # todo initialize Replay Buffer like in PPO
                 buffer: ReplayBuffer = ReplayBuffer(),
                 minibatch_size=128,
                 update_after=2000,
                 gamma=0.99,
                 epsilon: float = 0.99,
                 tau=0.999,
                 device='cuda' if torch.cuda.is_available() else 'cpu'
                 ) -> None:

        self.policy = policy.to(device)
        self.dqn_target = copy(policy).to(device)
        self.env = env
        self.optimizer = optimizer
        self.metric = metric
        self.buffer = buffer
        self.minibatch_size = minibatch_size
        self.update_after = update_after
        self.gamma = gamma
        self.epsilon = epsilon
        self.tau = tau
        self.device = device

    def train(self) -> None:
        """
        Update the policy using randomly sampled rollouts
        """
        self.policy.train() # Switch to train mode (as apposed to eval)
        self.optimizer.zero_grad()

        # Sample minibatch from replay buffer
        mini_batch = self.buffer.sample_batch(self.minibatch_size)
        states, actions, rewards, nextStates, dones = mini_batch

        # Compute q values for states
        target_q_value = self.dqn_target.forward(nextStates)

        # Compute the targets for training
        targets = rewards.to(self.device) + (self.gamma * torch.max(target_q_value, 1).values * (1 - dones.to(self.device)))

        # compute the predictions for training
        online_q_values = self.policy.forward(states)
        action_idx = actions.to(self.device).argmax(axis=1)
        predictions = online_q_values.gather(1, action_idx.unsqueeze(1)).flatten()

        # Update the loss
        loss = self.metric(predictions, targets)
        loss.backward(retain_graph=False)
        self.optimizer.step()

        soft_update(self.policy, self.dqn_target, self.tau)

    def learn(self, n_episodes: int) -> None:
        """
        Collect rollouts
        :param n_episodes: number of full episodes the agent should interact with the environment
        """
        state = self.env.reset()

        for i in range(n_episodes):
            episode_return = 0
            done = False
            while not done:

                # epsilon decay
                epsilon = self.epsilon

                # epsilon greedy action selection
                if np.random.choice([True, False], p=[epsilon, 1 - epsilon]):
                    action = np.random.randint(low=0, high=self.policy.num_actions)
                else:
                    logits = self.policy.forward(state).detach().cpu().numpy()
                    action = np.argmax(logits)

                # interact with the environment
                next_state, r, done, _ = self.env.step(action)
                episode_return += r

                self.buffer.add(state, action, r, next_state, done)
                state = next_state

                # update policy using temporal difference
                if self.buffer.length() > self.minibatch_size and self.buffer.length() > self.update_after:
                    self.train()

                if done:
                    state = self.env.reset()
                    print(f"Episode: \t{i}\t{episode_return}")
                    break
