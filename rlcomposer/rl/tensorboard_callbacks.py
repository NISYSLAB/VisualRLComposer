import stable_baselines3.sac
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import TensorBoardOutputFormat
from stable_baselines3.common.logger import Figure
import tensorflow as tf

from stable_baselines3.common.logger import configure
import numpy as np
import matplotlib.pyplot as plt

class Callback(BaseCallback):
    def __init__(self, tensorboard_log):
        super(Callback, self).__init__()
        self.states = []
        self.actions = []
        self.tensorboard_log = tensorboard_log

    def _on_step(self):
        if "SAC" in str(self.model):
            print("SAC")
            for i in range(0, self.locals["new_obs"].shape[1]):
                self.logger.record(f'state/State {i+1}', self.locals["new_obs"][0, i])
            self.logger.record('train/actions', self.locals["action"][0][0])
            self.logger.record('train/rewards', self.locals["reward"][0])
            self.logger.record('train/buffer_action', self.locals["buffer_action"][0][0])

        if "A2C" in str(self.model):
            print("A2C")
            for i in range(0, self.locals["new_obs"].shape[0]):
                for j in range(0, self.locals["new_obs"].shape[1]):
                    self.logger.record(f'state/Environment {i+1}, State {j+1}', self.locals["new_obs"][i, j])
                self.logger.record(f'train/Environment {i+1}, action', self.locals["actions"][i][0])
                self.logger.record(f'train/Environment {i+1}, reward', self.locals["rewards"][i])
                self.logger.record(f'train/Environment {i+1}, values', self.locals["values"][i][0])
                self.logger.record(f'train/Environment {i+1}, clipped_actions', self.locals["clipped_actions"][i][0])
                self.logger.record(f'train/Environment {i+1}, log_probs', self.locals["log_probs"][i])

        if "TD3" in str(self.model):
            print("TD3")
            for i in range(0, self.locals["new_obs"].shape[1]):
                self.logger.record(f'state/State {i+1}', self.locals["new_obs"][0, i])
            self.logger.record('train/actions', self.locals["action"][0][0])
            self.logger.record('train/rewards', self.locals["reward"][0])
            self.logger.record('train/buffer_action', self.locals["buffer_action"][0][0])

        if "PPO" in str(self.model):
            print("PPO")
            for i in range(0, self.locals["new_obs"].shape[0]):
                for j in range(0, self.locals["new_obs"].shape[1]):
                    self.logger.record(f'state/Environment {i + 1}, State {j + 1}', self.locals["new_obs"][i, j])
                self.logger.record(f'train/Environment {i + 1}, action', self.locals["actions"][i][0])
                self.logger.record(f'train/Environment {i + 1}, reward', self.locals["rewards"][i])
                self.logger.record(f'train/Environment {i + 1}, values', self.locals["values"][i][0])
                self.logger.record(f'train/Environment {i + 1}, clipped_actions', self.locals["clipped_actions"][i][0])
                self.logger.record(f'train/Environment {i + 1}, log_probs', self.locals["log_probs"][i])

        if "DQN" in str(self.model):
            print("DQN")
            for i in range(0, self.locals["new_obs"].shape[1]):
                self.logger.record(f'state/State {i+1}', self.locals["new_obs"][0, i])
            self.logger.record('train/actions', self.locals["action"][0])
            self.logger.record('train/rewards', self.locals["reward"][0])

        if "DDPG" in str(self.model):
            print("DDPG")
            for i in range(0, self.locals["new_obs"].shape[1]):
                self.logger.record(f'state/State {i+1}', self.locals["new_obs"][0, i])
            self.logger.record('train/actions', self.locals["action"][0][0])
            self.logger.record('train/rewards', self.locals["reward"][0])
            self.logger.record('train/buffer_action', self.locals["buffer_action"][0][0])

        self.logger.dump(step=self.num_timesteps)
        for i in self.locals.keys():
            print(i, self.locals[i])
        return True

    def _on_training_end(self):
        pass
