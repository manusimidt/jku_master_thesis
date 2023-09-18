import os
import gym
import torch
import numpy as np
from common.rl.logger import Tracker
from common.rl.utils import set_seed
from common.rl.buffer2 import Episode, Transition, RolloutBuffer
from common.rl.ppo.policies import ActorCriticNet


class PPO:
    def __init__(self,
                 policy: ActorCriticNet,
                 env: gym.Env,
                 optimizer: torch.optim.Optimizer,
                 lr_decay: torch.optim.lr_scheduler.LRScheduler,
                 metric=torch.nn.MSELoss(),
                 buffer: RolloutBuffer = RolloutBuffer(),
                 seed: int or None = None,
                 n_epochs: int = 4,
                 gamma=0.99,
                 eps_clip=0.2,
                 reward_scale: float = 0.01,
                 value_loss_scale: float = 0.5,
                 policy_loss_scale: float = 1.0,
                 entropy_loss_scale: float = 0.01,
                 use_buffer_reset=True,
                 tracker: Tracker = Tracker(),  # by default initialize it as an empty tracker
                 device='cuda' if torch.cuda.is_available() else 'cpu',
                 ) -> None:
        """
        :param gamma: discount rate
        """
        if seed: set_seed(env, seed, force=False)
        self.policy = policy.to(device)
        self.env = env
        self.optimizer = optimizer
        self.lr_decay = lr_decay
        self.metric = metric
        self.buffer = buffer

        self.n_epochs = n_epochs
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.reward_scale = reward_scale
        self.value_loss_scale = value_loss_scale
        self.policy_loss_scale = policy_loss_scale
        self.entropy_loss_scale = entropy_loss_scale
        self.use_buffer_reset = use_buffer_reset
        self.tracker = tracker
        self.device = device

    def train(self) -> None:
        """
        Update the policy using the currently gathered rollout buffer
        """
        self.policy.train()  # Switch to train mode (as apposed to eval)
        self.optimizer.zero_grad()

        value_losses, policy_losses, entropy_losses, total_losses = [], [], [], []

        for _ in range(self.n_epochs):
            # sample batch from buffer
            samples = self.buffer.sample()
            episode_returns = []
            states = []
            actions = []
            old_log_probs = []
            for s in samples:
                states += [torch.FloatTensor(s.state).to(self.device), ]
                episode_returns += [s.g_return, ]
                actions += [s.action, ]
                old_log_probs += [s.log_probs, ]

            # create tensors
            state_t = torch.stack(states, dim=0).to(self.device).detach()
            action_t = torch.LongTensor(actions).to(self.device).detach()
            return_t = torch.FloatTensor(episode_returns).view(-1, 1).to(self.device)
            old_log_probs_t = torch.stack(old_log_probs, dim=0).to(self.device).detach()

            # normalize returns
            return_t = (return_t - return_t.mean()) / (return_t.std() + 1e-5)

            # get value function estimate
            v_s, log_probs, entropy = self.policy.evaluate(state_t, action_t)

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

            # perform training step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            self.lr_decay.step()

            value_losses.append(value_loss.mean().item())
            policy_losses.append(policy_loss.mean().item())
            entropy_losses.append(entropy.view(-1, 1).mean().item())
            total_losses.append(loss.mean().item())

        self.tracker.end_epoch({
            "value_loss": np.mean(value_losses),
            "policy_loss": np.mean(policy_losses),
            "entropy_loss": np.mean(entropy_losses),
            "total_loss": np.mean(total_losses),
            "learning_rate": self.lr_decay.get_last_lr()[0]
        })

    def learn(self, n_episodes) -> None:
        """
        Collect rollouts
        :param n_episodes: number of full episodes the agent should interact with the environment
        """
        for _ in range(n_episodes):
            state = self.env.reset()
            # create a new episode
            episode = Episode(discount=self.gamma)
            done = False
            while not done:
                # select an action from the agent's policy
                state_t = torch.FloatTensor(state).to(self.device)
                state_t = state_t if len(state_t.shape) == 2 else state_t.unsqueeze(0)
                action, log_probs = self.policy.act(state_t)

                # enter action into the env
                next_state, reward, done, _ = self.env.step(action.item())
                self.tracker.step(action, reward)
                episode.total_reward += reward

                # store agent trajectory
                transition = Transition(state=state, action=action, reward=(reward * self.reward_scale),
                                        log_probs=log_probs)
                episode.append(transition)

                # update agent if done
                if done:
                    self.tracker.end_episode()
                    # add current episode to the replay buffer
                    self.buffer.add(episode)

                    # skip if stored episodes are less than the batch size
                    if len(self.buffer) < self.buffer.min_transitions: break

                    # update the network
                    self.train()
                    self.buffer.update_stats()
                    if self.use_buffer_reset: self.buffer.reset()

                state = next_state

    def save(self, checkpoint_dir: str, name: str = 'PPO', info: dict or None = None) -> str:
        """
        Saves the model
        :param checkpoint_dir: folder to which the model should be saved to
        :param name: name of the checkpoint file
        :param info: any additional info you want to store into the model file
        :return: returns the path to which the model was saved to
        """
        state = {
            'state_dict': self.policy.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'info': info
        }
        cpt_path = f"{checkpoint_dir}{os.sep}{name}.pth"
        os.makedirs(checkpoint_dir, exist_ok=True)
        torch.save(state, cpt_path)
        return cpt_path

    def load(self, path: str) -> dict:
        """
        Loads the weights from the checkpoint file
        :param path: path to the checkpoint file
        :return: returns the info
        """
        ckp: dict = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(ckp['state_dict'])
        self.optimizer.load_state_dict(ckp['optimizer'])
        return ckp['info'] if 'info' in ckp else None
