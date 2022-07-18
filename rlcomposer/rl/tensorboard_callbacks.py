from stable_baselines3.common.callbacks import BaseCallback
#from stable_baselines3.common.logger import TensorBoardOutputFormat
#from stable_baselines3.common.logger import Figure
#import tensorflow as tf


class Callback(BaseCallback):
    def __init__(self, tensorboard_log, signal):
        super(Callback, self).__init__()
        self.states = []
        self.actions = []
        self.signal = signal
        self.signal.progress.emit(0)
        self.tensorboard_log = tensorboard_log

    def _on_step(self):
        self.signal.progress.emit(100 - int(100*self.model._current_progress_remaining))

        if self.locals['tb_log_name'] in ['A2C', 'PPO']:
            for i in range(0, self.locals["new_obs"].shape[0]):
                for j in range(0, self.locals["new_obs"].shape[1]):
                    self.logger.record(f'state/Environment {i+1}, State {j+1}', self.locals["new_obs"][i, j])
                self.logger.record(f'train/Environment {i+1}, reward', self.locals["rewards"][i])
                self.logger.record(f'train/Environment {i+1}, values', self.locals["values"][i][0])
                self.logger.record(f'train/Environment {i+1}, log_probs', self.locals["log_probs"][i])
                if len(self.locals["actions"].shape) > 1:
                    self.logger.record(f'train/Environment {i + 1}, action', self.locals["actions"][i][0])
                else:
                    self.logger.record(f'train/Environment {i + 1}, action', self.locals["actions"][i])

        if self.locals['tb_log_name'] in ['SAC', 'DQN', 'TD3', 'DDPG']:
            for i in range(0, self.locals["new_obs"].shape[1]):
                self.logger.record(f'state/State {i+1}', self.locals["new_obs"][0, i])
            self.logger.record('train/rewards', self.locals["reward"][0])
            self.logger.record('train/actions', self.locals["action"][0][0])
            if len(self.locals["action"].shape) > 1:
                self.logger.record('train/actions', self.locals["action"][0][0])
            else:
                self.logger.record('train/actions', self.locals["action"][0])

        self.logger.dump(step=self.num_timesteps)
        #for i in self.locals.keys():
        #    print(i, self.locals[i])
        return self.signal.finished_value

    def _on_training_end(self):
        self.signal.progress.emit(100 - int(100*self.model._current_progress_remaining))
        pass
