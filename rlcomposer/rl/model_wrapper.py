from stable_baselines import *
import sys

class ModelWrapper():
  def __init__(self, model_name=""):
    self.model = None
    self.env = None
    self.model_name = model_name
    self.param = {"policy_name": "MlpPolicy",
                  "total_timesteps": 2000,
    }
    self.total_timesteps = 10

  def setModel(self,env):
    self.env = env
    if self.model is None:
      if self.model_name=="DQN":
        self.model = getattr(sys.modules[__name__], self.model_name)(
        env=self.env,
        policy="MlpPolicy",
        learning_rate=1e-3,
        buffer_size=50000,
        exploration_fraction=0.1,
        exploration_final_eps=0.02,
    )

      else:
        self.model = getattr(sys.modules[__name__], self.model_name)(self.param["policy_name"], self.env)

  def loadModel(self, name):
    model_str = name.split('/')[-1].split('_')[0]
    self.model = getattr(sys.modules[__name__], model_str).load(name)

  def setParameters(self, param):
    for key,value in param.items():
      setattr(self, key, value)
    if not self.env is None: self.model = getattr(sys.modules[__name__], self.model_name)(self.param["policy_name"],
                                                                                          self.env, verbose=1)

