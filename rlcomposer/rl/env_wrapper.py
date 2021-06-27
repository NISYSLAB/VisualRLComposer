

DEBUG = True


def decorator(env_name, **kwargs):
  """
  Decorator for gym environments
  """

  def _init():
    if env_name == "Pendulum":
      from environments import Pendulum
      env = Pendulum()
      return env

  return _init

# class EnvWrapper(DummyVecEnv):
#   def __init__(self, name="", config=None, n_workers=1, dec_fn=decorator, **kwargs):
#     self.name = name
#     self.config = config
#     self.workers = n_workers
#     self.env_type = ''
#     vectorized = [dec_fn(name, config=config, **kwargs) for x in range(n_workers)]
#     super(EnvWrapper, self).__init__(vectorized)



class EnvWrapper():
  def __init__(self, env_name=""):
    self.env = None
    self.env_name = env_name
    self.param = {}
    self.setEnv()
    self.setParameters(self.param)


  def setEnv(self):
    if self.env_name == "Pendulum":
      from environments import Pendulum
      self.env = Pendulum()
      self.param = {"max_speed":8,
                    "max_torque":2.,
                    "dt":.05,
                    "g":10.0,
                    "m": 1.,
                    "l":1.,}

  def setReward(self, reward):
    setattr(self.env, "reward_obj", reward)

  def setParameters(self, param):
    for key,value in param.items():
      setattr(self.env, key, value)


#