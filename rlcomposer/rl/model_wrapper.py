class ModelWrapper():
  def __init__(self, model_name=""):
    self.env = None
    self.env_name = env_name
    self.param = {}

  def setEnv(self):
    if self.env_name == "Pendulum":
      self.env = PendulumEnv()
