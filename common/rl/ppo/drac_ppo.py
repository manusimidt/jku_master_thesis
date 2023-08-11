import gym
import matplotlib.pyplot as plt
import numpy as np
import torch

import env
from env import AugmentingEnv
from rl.ppo.ppo import PPO
from rl.common.buffer2 import RolloutBuffer, AugmentedTransition, Episode
from rl.common.logger import Tracker
from rl.ppo.policies import ActorCriticNet


class DrACPPO(PPO):

    def __init__(self, policy: ActorCriticNet, env: gym.Env, optimizer: torch.optim.Optimizer,
                 metric=torch.nn.MSELoss(),
                 buffer: RolloutBuffer = RolloutBuffer(), seed: int or None = None, n_epochs: int = 4, gamma=0.99,
                 eps_clip=0.2, reward_scale: float = 0.01, value_loss_scale: float = 0.5,
                 policy_loss_scale: float = 1.0, entropy_loss_scale: float = 0.01, use_buffer_reset=True,
                 tracker: Tracker = Tracker(),
                 device='cuda' if torch.cuda.is_available() else 'cpu',
                 alpha_policy: float = 0.1, alpha_value: float = 0.01) -> None:
        super().__init__(policy, env, optimizer, metric, buffer, seed, n_epochs, gamma, eps_clip, reward_scale,
                         value_loss_scale, policy_loss_scale, entropy_loss_scale, use_buffer_reset, tracker, device)
        # Check if the environment is an instance of AugmentingEnv
        # DrAC only works for augmenting environments!
        assert isinstance(env, AugmentingEnv), "DrAC only works for augmenting envs!"
        self.env: AugmentingEnv = env
        # weight of the DrAC regularization term
        self.alpha_policy = alpha_policy
        self.alpha_value = alpha_value

    def train(self) -> None:
        """
        Update the policy using the currently gathered rollout buffer
        """
        self.policy.train()  # Switch to train mode (as apposed to eval)
        self.optimizer.zero_grad()

        for _ in range(self.n_epochs):
            # sample batch from buffer
            # NOTE: for DrAC these are not normal Transitions but augmented Transitions!
            samples = self.buffer.sample()
            episode_returns = []
            ori_states = []
            aug_states = []
            actions = []
            old_log_probs = []
            for s in samples:
                ori_states += [torch.FloatTensor(s.state).to(self.device), ]
                aug_states += [torch.FloatTensor(s.augmented_state).to(self.device), ]
                episode_returns += [s.g_return, ]
                actions += [s.action, ]
                old_log_probs += [s.log_probs, ]

            # create tensors
            ori_state_t = torch.stack(ori_states, dim=0).to(self.device).detach()
            aug_state_t = torch.stack(aug_states, dim=0).to(self.device).detach()
            action_t = torch.LongTensor(actions).to(self.device).detach()
            return_t = torch.FloatTensor(episode_returns).view(-1, 1).to(self.device)
            old_log_probs_t = torch.stack(old_log_probs, dim=0).to(self.device).detach()

            # normalize returns
            return_t = (return_t - return_t.mean()) / (return_t.std() + 1e-5)

            # get value function estimate
            v_s, log_probs, entropy = self.policy.evaluate(ori_state_t, action_t)

            # use importance sampling
            ratios = torch.exp(log_probs.view(-1, 1) - old_log_probs_t.detach())

            # compute advantage
            advantages = return_t - v_s.detach()

            # compute ppo trust region loss
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            policy_loss = -torch.min(surr1, surr2)

            # compute critic loss
            value_loss = self.metric(v_s, return_t)

            # combine losses
            loss = self.value_loss_scale * value_loss + \
                   self.policy_loss_scale * policy_loss - \
                   self.entropy_loss_scale * entropy.view(-1, 1)

            # ============ DRAC CHANGES ================ #
            ori_action_logits = self.policy.actor.forward(ori_state_t)
            aug_action_logits = self.policy.actor.forward(aug_state_t)

            ori_values = self.policy.critic.forward(ori_state_t)
            aug_values = self.policy.critic.forward(aug_state_t)

            ori_action_probs = torch.distributions.Categorical(logits=ori_action_logits)
            aug_action_probs = torch.distributions.Categorical(logits=aug_action_logits)

            G_pi = torch.distributions.kl_divergence(ori_action_probs, aug_action_probs).mean()
            mse = torch.nn.MSELoss()
            G_V = mse(ori_values, aug_values)

            # loss -= self.alpha * (G_pi + G_V)
            loss += (self.alpha_policy * G_pi + self.alpha_value * G_V)
            # ============ END DRAC CHANGES ============ #

            # perform training step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            self.tracker.end_epoch({
                "value_loss": value_loss.mean(),
                "policy_loss": policy_loss.mean(),
                "entropy_loss": entropy.view(-1, 1).mean(),
                "policy_reg_loss": G_pi.mean(),
                "value_reg_loss": G_V.mean()
            })

    def learn(self, n_episodes) -> None:
        """
        Collect rollouts
        :param n_episodes: number of full episodes the agent should interact with the environment
        """
        for i in range(n_episodes):
            state_aug, state_ori = self.env.reset_augmented()
            # In DrAC we train using the ORIGINAL state
            state = state_ori
            # create a new episode
            episode = Episode(discount=self.gamma)
            done = False
            while not done:
                # select an action from the agent's policy
                state_t = torch.FloatTensor(state).to(self.device)
                state_t = state_t if len(state_t.shape) == 2 else state_t.unsqueeze(0)
                action, log_probs = self.policy.act(state_t)

                # enter action into the env
                next_state_aug, next_state_ori, reward, done, _ = self.env.step_augmented(action.item())
                self.tracker.step(action, reward)
                episode.total_reward += reward

                # store agent trajectory
                # todo what is reward scale
                transition = AugmentedTransition(state=state, augmented_state=state_aug, action=action,
                                                 reward=(reward * self.reward_scale), log_probs=log_probs)
                episode.append(transition)

                # update agent if done
                if done:
                    if isinstance(self.env, env.UCBAugmentingEnv):
                        aug_names = [a['name'] for a in env.POSSIBLE_AUGMENTATIONS]
                        aug_counts = dict(zip(aug_names, self.env.N))
                        self.tracker.end_episode(aug_counts=aug_counts)
                    else:
                        self.tracker.end_episode()
                    # add current episode to the replay buffer
                    self.buffer.add(episode)

                    # skip if stored episodes are less than the batch size
                    if len(self.buffer) < self.buffer.min_transitions: break

                    # update the network
                    self.train()
                    self.buffer.update_stats()
                    if self.use_buffer_reset: self.buffer.reset()

                state = next_state_ori
                state_aug = next_state_aug
