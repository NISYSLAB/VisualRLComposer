import numpy as np
import sys
import math

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


class MountainCarContinuousReward():
    def __init__(self):
        pass

    def calculateReward(self, done, act):
        reward = 0
        if done:
            reward = 100.0
        reward -= math.pow(act, 2) * 0.1
        return reward

class CartPoleReward():
    def __init__(self):
        pass

    def calculateReward(self, done, obj, logger):
        if not done:
            reward = 1.0
        elif obj.steps_beyond_done is None:
            # Pole just fell!
            obj.steps_beyond_done = 0
            reward = 1.0
        else:
            if obj.steps_beyond_done == 0:
                logger.warn(
                    "You are calling 'step()' even though this "
                    "environment has already returned done = True. You "
                    "should always call 'reset()' once you receive 'done = "
                    "True' -- any further steps are undefined behavior."
                )
            obj.steps_beyond_done += 1
            reward = 0.0
        return reward



