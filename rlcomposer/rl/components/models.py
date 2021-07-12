import sys
from dataclasses import dataclass, field

def return_classes():
  current_module = sys.modules[__name__]
  class_names = []
  for key in dir(current_module):
    if isinstance(getattr(current_module, key), type):
      class_names.append(key)
  return class_names


class DQN():
    pass

class PPO():
  pass

class SAC():
  pass

class A2C():
  pass

class DDPG():
  pass


class TD3():
  pass
