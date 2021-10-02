from .components.rewards import *
import sys

class RewardWrapper():
  def __init__(self, reward_name=""):
    self.reward_name = reward_name
    self.setReward()
    self.setParameters(self.param)


  def setReward(self):
    print(sys.modules[__name__])
    self.reward = getattr(sys.modules[__name__], self.reward_name)()
    self.param = {}

  def setParameters(self, param):
    for key,value in param.items():
      setattr(self.reward, key, value)



