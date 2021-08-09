from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import TensorBoardOutputFormat

class Callback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.states = []
        self.actions = []

    def _on_step(self):
        self.states.append(self.locals["obs_tensor"].detach().cpu().numpy())
        self.actions.append(self.locals["actions"])
        self.logger.record('actions', self.locals["actions"][0])
        for i in range (self.states[0].shape[1]):
            self.logger.record('states_' + str(i), self.locals["obs_tensor"][0,i])
        return True

    def _on_training_end(self):
        pass
