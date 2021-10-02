import sys
from stable_baselines3 import *

class ModelWrapper():
  def __init__(self, model_name=""):
    self.model = None
    self.env = None
    self.model_name = model_name
    self.initParam()


  def initParam(self):
    self.param = {}
    if self.model_name == "DQN":
      self.param = {"total_timesteps": 20000,
                    "policy": "MlpPolicy",
                    "learning_rate": 0.0001,
                    "buffer_size": 1000000,
                    "learning_starts": 50000,
                    "batch_size": 32,
                    "tau": 1.0,
                    "gamma": 0.99,
                    "train_freq": 4,
                    "gradient_steps": 1,
                    "target_update_interval": 10000,
                    "exploration_fraction": 0.1,
                    "exploration_initial_eps": 1.0,
                    "exploration_final_eps": 0.05,
                    "max_grad_norm": 10,
                    }

    elif self.model_name == "SAC":
      self.param = {"total_timesteps": 20000,
                    "policy": "MlpPolicy",
                    "learning_rate": 1e-3,
                    "buffer_size": 1000000,
                    "batch_size": 256,
                    "tau": 0.005,
                    "gamma": 0.99,
                    "learning_starts": 100,
                    "train_freq": 1,
                    "gradient_steps": 1
                    }

    elif self.model_name == "A2C":
      self.param = {"total_timesteps": 20000,
                    "policy": "MlpPolicy",
                    "learning_rate": 0.0007,
                    "n_steps": 5,
                    "gamma": 0.99,
                    "gae_lambda": 1.0,
                    "ent_coef": 0.0,
                    "vf_coef": 0.5,
                    "max_grad_norm": 0.5,
                    "rms_prop_eps": 1e-05
                    }

    elif self.model_name == "DDPG":
        self.param = {"total_timesteps": 10000,
                      "policy": "MlpPolicy",
                      "learning_rate":0.001,
                      "buffer_size": 1000000,
                      "learning_starts": 100,
                      "batch_size": 100,
                      "tau": 0.005,
                      "gamma": 0.99,
                      }
    elif self.model_name == "PPO":
        self.param = {"total_timesteps": 20000,
                      "policy": "MlpPolicy",
                      "learning_rate": 0.0003,
                      "n_steps": 2048,
                      "batch_size": 64,
                      "n_epochs": 10,
                      "gae_lambda": 0.95,
                      "gamma": 0.99,
                      "clip_range": 0.2,
                      "ent_coef": 0.0,
                      "vf_coef": 0.5,
                      "max_grad_norm": 0.5,
                      }

    elif self.model_name == "TD3":
        self.param = {"total_timesteps": 10000,
                      "policy": "MlpPolicy",
                      "learning_rate": 0.001,
                      "buffer_size": 1000000,
                      "learning_starts": 100,
                      "batch_size": 100,
                      "tau": 0.005,
                      "gamma": 0.99,
                      "policy_delay": 2,
                      "target_policy_noise": 0.2,
                      "target_noise_clip": 0.5
                      }
    try:
        self.total_timesteps = self.param["total_timesteps"]
    except:
        pass

  def setModel(self,env):
    self.env = env
    if self.model is None:
      self.model = getattr(sys.modules[__name__], self.model_name)(
      env=self.env,
      **without(self.param, "total_timesteps"))

  def loadModel(self, dir):
    model_str = dir.split('/')[-1].split('_')[0]
    self.model = getattr(sys.modules[__name__], model_str).load(dir.split("/")[-1].split(".")[0])

  def setParameters(self, param):
    for key,value in param.items():
      if key == "total_timesteps":
        continue
      self.param[key] = value
    self.total_timesteps = param["total_timesteps"]
    if not self.env is None:
      self.model = getattr(sys.modules[__name__], self.model_name)(
        env=self.env,
        **without(self.param, "total_timesteps"))

def without(d, key):
  new_d = d.copy()
  new_d.pop(key)
  return new_d