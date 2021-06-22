
import numpy as np

class RewardWrapper():
  def __init__(self, reward_name=""):
    self.reward = None
    self.reward_name = reward_name
    self.param = {}
    self.setReward()
    self.setParameters(self.param)


  def setReward(self):
    if self.reward_name == "Pendulum Reward":
      self.reward = PendulumReward()
      self.param = {}

  def setParameters(self, param):
    for key,value in param.items():
      setattr(self.reward, key, value)




def angle_normalize(x):
  return (((x + np.pi) % (2 * np.pi)) - np.pi)

class PendulumReward():
    def __init__(self):
        pass

    def calculateReward(self, th, thdot, u):
        return angle_normalize(th) ** 2 + .1 * thdot ** 2 + .001 * (u ** 2)