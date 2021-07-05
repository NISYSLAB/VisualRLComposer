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
    if self.model is None:
      self.model = getattr(sys.modules[__name__], self.model_name)(self.policy_name, self.env)

  def loadModel(self, name):
    model_str = name.split('/')[-1].split('_')[0]
    self.model = getattr(sys.modules[__name__], model_str).load(name)

  def setParameters(self, param):
    for key,value in param.items():
      setattr(self, key, value)
    if not self.env is None: self.model = getattr(sys.modules[__name__], self.model_name)(self.policy_name, self.env, verbose=1)

  def setEnv(self, env):
    self.env = env