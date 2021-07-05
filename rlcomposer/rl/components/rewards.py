import numpy as np
import sys

def return_classes():
  current_module = sys.modules[__name__]
  class_names = []
  for key in dir(current_module):
    if isinstance(getattr(current_module, key), type):
      class_names.append(key)
  return class_names

### Reward Function for Pendulum environment ###
def angle_normalize(x):
  return (((x + np.pi) % (2 * np.pi)) - np.pi)

class PendulumReward():
    def __init__(self):
        pass

    def calculateReward(self, th, thdot, u):
        return angle_normalize(th) ** 2 + .1 * thdot ** 2 + .001 * (u ** 2)

###############################################################################

class MountainCarReward():
    def __init__(self):
        pass

    def calculateReward(self):
        return -1