import abc
import os
from collections import deque
from pathlib import Path
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib
from torch.utils.tensorboard import SummaryWriter


class Logger(abc.ABC):
    """ Extracts and/or persists tracker information. """
    def __init__(self):
        pass

    @abc.abstractmethod
    def on_step(self, step: int, **kwargs):
        pass

    @abc.abstractmethod
    def on_epoch_end(self, epoch: int, **kwargs):
        """Actions to take on the end of an epoch."""
        pass

    @abc.abstractmethod
    def on_episode_end(self, episode: int, **kwargs):
        pass


class ConsoleLogger(Logger):
    def __init__(self, log_every: int = 1, average_over: int or None = None):
        """
        :param log_every: log every nth episode
        :param average_over: number of episodes the metrics should be averaged over
        """
        super().__init__()
        self.log_every = log_every
        self.average_over = average_over if average_over else log_every
        self.return_queue = deque(maxlen=average_over)

    def on_step(self, step: int, **kwargs):
        pass

    def on_epoch_end(self, epoch: int, **kwargs):
        pass

    def on_episode_end(self, episode: int, **kwargs):
        self.return_queue.append(kwargs['episode_return'])
        if not episode % self.log_every == 0: return
        msg = f"Episode: {str(episode).rjust(6)}, avg.ret.: {np.mean(self.return_queue):.4f} " \
              f"(over last {self.average_over} episodes)"
        print(msg)


class TensorboardLogger(Logger):
    def __init__(self, log_dir: str = './tensorboard', run_id: str = datetime.today().strftime('%Y-%m-%d-%H%M%S')):
        super().__init__()
        self.writer = SummaryWriter(log_dir=log_dir + os.sep + run_id)
        print(f"Tensor board logging active. Start tensorboard with 'tensorboard --logdir {log_dir}'")

    def on_step(self, step: int, **kwargs):
        pass

    def on_epoch_end(self, epoch: int, **kwargs):
        if 'losses' in kwargs and kwargs['losses']:
            for loss_name in kwargs['losses']:
                self.writer.add_scalar(f'loss/{loss_name}', kwargs['losses'][loss_name], epoch)

    def on_episode_end(self, episode: int, **kwargs):
        self.writer.add_scalar('rollout/ep_return', kwargs['episode_return'], episode)
        self.writer.add_scalar('rollout/ep_length', kwargs['episode_length'], episode)

        if 'aug_counts' in kwargs and kwargs['aug_counts']:
            self.writer.add_scalars('aug/ucb_counts', kwargs['aug_counts'], episode)


class FigureLogger(Logger):
    """
    Stores all necessary information to be able to return a nice figure at the end
    """

    def __init__(self):
        super().__init__()
        self.episode_returns = []
        self.episode_lengths = []

    def on_step(self, step: int, **kwargs):
        pass

    def on_epoch_end(self, epoch: int, **kwargs):
        pass

    def on_episode_end(self, episode: int, **kwargs):
        self.episode_returns.append(kwargs['episode_return'])
        self.episode_lengths.append(kwargs['episode_length'])
        pass

    def get_figure(self, fig_size=(16, 8)) -> matplotlib.figure:
        def get_running_stat(stat, stat_len):
            # evaluate stats
            cum_sum = np.cumsum(np.insert(stat, 0, 0))
            return (cum_sum[stat_len:] - cum_sum[:-stat_len]) / stat_len

        episode_nr = np.array(range(1, len(self.episode_returns) + 1))
        cum_r = get_running_stat(self.episode_returns, 10)
        cum_l = get_running_stat(self.episode_lengths, 10)
        fig: plt.Figure = plt.figure(figsize=fig_size)

        plot1 = fig.add_subplot(121)

        # plot rewards
        plot1.plot(episode_nr[-len(cum_r):], cum_r)
        plot1.plot(episode_nr, self.episode_returns, alpha=0.5)
        plot1.set_xlabel('Episode')
        plot1.set_ylabel('Episode Reward')

        plot2 = fig.add_subplot(122)

        # plot episode lengths
        plot2.plot(episode_nr[-len(cum_l):], cum_l)
        plot2.plot(episode_nr, self.episode_lengths, alpha=0.5)
        plot2.set_xlabel('Episode')
        plot2.set_ylabel('Episode Length')
        return fig


class WeightsAndBiasesLogger:
    # todo
    pass


class Tracker:
    """ Collects loggers """

    def __init__(self, *loggers: Logger):
        self.episode = 0
        self.epoch = 0
        self.steps = 0

        self.step_pointer = 0  # points to the step where the last episode ended
        self.episode_return = 0
        self.loggers = list(loggers)

    def step(self, action: int, reward: float):
        self.steps += 1
        self.episode_return += reward
        for logger in self.loggers:
            logger.on_step(self.epoch, action=action, reward=reward)

    def end_epoch(self, losses: dict or None = None):
        self.epoch += 1
        for logger in self.loggers:
            logger.on_epoch_end(self.epoch, losses=losses)

    def end_episode(self, aug_counts: list or None = None):
        """
        Ends the episode.
        :param aug_counts: optionally log how often a certain augmentation was chosen
        :return:
        """
        self.episode += 1
        episode_length = self.steps - self.step_pointer

        for logger in self.loggers:
            logger.on_episode_end(self.episode, episode_return=self.episode_return, episode_length=episode_length,
                                  aug_counts=aug_counts)

        # reset rolling stats
        self.step_pointer = self.steps
        self.episode_return = 0
