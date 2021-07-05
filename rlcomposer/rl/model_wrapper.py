from stable_baselines3 import *
import sys

class ModelWrapper():
  def __init__(self, model_name=""):
    self.model = None
    self.env = None
    self.policy_name = "MlpPolicy"
    self.model_name = model_name
    self.param = {"policy_name": "MlpPolicy",
                  "total_timesteps": 2000,
    }
    self.total_timesteps = 10

  def setModel(self):
    print(sys.modules[__name__])
    self.model = getattr(sys.modules[__name__], self.model_name)(self.policy_name, self.env)


  def setParameters(self, param):
    for key,value in param.items():
      setattr(self, key, value)
    if not self.env is None: self.model = getattr(sys.modules[__name__], self.model_name)(self.policy_name, self.env, verbose=1)