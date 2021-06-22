import gym
from gym.utils import seeding
import numpy as np
from gym import spaces
from os import path

DEBUG = True

class EnvWrapper():
  def __init__(self, env_name=""):
    self.env = None
    self.env_name = env_name
    self.param = {}
    self.setEnv(None)
    self.setParameters(self.param)


  def setEnv(self, reward):
    if self.env_name == "Pendulum":
      self.env = Pendulum(reward)
      self.param = {"max_speed":8,
                    "max_torque":2.,
                    "dt":.05,
                    "g":10.0,
                    "m": 1.,
                    "l":1.,}

  def setParameters(self, param):
    for key,value in param.items():
      setattr(self.env, key, value)

class Pendulum(gym.Env):
  metadata = {
    'render.modes': ['human', 'rgb_array'],
    'video.frames_per_second': 30
  }

  def __init__(self, reward=None):
    self.max_speed = 8
    self.max_torque = 2.
    self.dt = .05
    self.g = 10.0
    self.m = 1.
    self.l = 1.
    self.reward_obj = reward

    self.viewer = None

    high = np.array([1., 1., self.max_speed], dtype=np.float32)
    self.action_space = spaces.Box(
      low=-self.max_torque,
      high=self.max_torque, shape=(1,),
      dtype=np.float32
    )
    self.observation_space = spaces.Box(
      low=-high,
      high=high,
      dtype=np.float32
    )

    self.seed()

  def seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]

  def step(self, u):
    th, thdot = self.state  # th := theta

    g = self.g
    m = self.m
    l = self.l
    dt = self.dt

    u = np.clip(u, -self.max_torque, self.max_torque)[0]
    self.last_u = u  # for rendering


    costs = self.reward_obj.calculateReward(th, thdot, u)

    newthdot = thdot + (-3 * g / (2 * l) * np.sin(th + np.pi) + 3. / (m * l ** 2) * u) * dt
    newth = th + newthdot * dt
    newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)
    if DEBUG: print("Inside step 1")
    self.state = np.array([newth, newthdot])
    return self._get_obs(), -costs, False, {}

  def reset(self):
    high = np.array([np.pi, 1])
    print("Reset 1")
    self.state = self.np_random.uniform(low=-high, high=high)
    print("Reset 2")
    self.last_u = None
    print(self.state)
    return self._get_obs()

  def _get_obs(self):
    theta, thetadot = self.state
    print("Get obs 1")
    return np.array([np.cos(theta), np.sin(theta), thetadot])

  def render(self, mode='human'):
    if self.viewer is None:
      from gym.envs.classic_control import rendering
      self.viewer = rendering.Viewer(500, 500)
      self.viewer.set_bounds(-2.2, 2.2, -2.2, 2.2)
      rod = rendering.make_capsule(1, .2)
      rod.set_color(.8, .3, .3)
      self.pole_transform = rendering.Transform()
      rod.add_attr(self.pole_transform)
      self.viewer.add_geom(rod)
      axle = rendering.make_circle(.05)
      axle.set_color(0, 0, 0)
      self.viewer.add_geom(axle)
      if DEBUG: print("Inside render 1")
      try:
        fname = path.join(path.dirname(__file__), "assets\\clockwise.png")

        if DEBUG: print("Inside render 2")
        self.img = rendering.Image(fname, 1., 1.)
      except Exception as e:
        print(e)
      if DEBUG: print("Inside render 3")
      self.imgtrans = rendering.Transform()
      if DEBUG: print("Inside render 4")
      self.img.add_attr(self.imgtrans)
    if DEBUG: print("Inside render 5")
    self.viewer.add_onetime(self.img)
    if DEBUG: print("Inside render 6")
    self.pole_transform.set_rotation(self.state[0] + np.pi / 2)
    if DEBUG: print("Inside render 7")
    if self.last_u:
      if DEBUG: print("Inside render 8")
      self.imgtrans.scale = (-self.last_u / 2, np.abs(self.last_u) / 2)
    if DEBUG: print("Inside render 9")
    return self.viewer.render(return_rgb_array=mode == 'rgb_array')

  def close(self):
    if self.viewer:
      self.viewer.close()
      self.viewer = None


#