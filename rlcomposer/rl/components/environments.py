import gym
from gym.utils import seeding
import numpy as np
from gym import spaces
from os import path
import sys

def return_classes():
  current_module = sys.modules[__name__]
  class_names = []
  for key in dir(current_module):
    if isinstance(getattr(current_module, key), type):
      class_names.append(key)
  return class_names

DEBUG = False

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
    self.reward_fn = reward

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


    costs = self.reward_fn.calculateReward(th, thdot, u)

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
        fname = path.join(path.dirname(__file__), "../assets/clockwise.png")

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



"""
http://incompleteideas.net/sutton/MountainCar/MountainCar1.cp
permalink: https://perma.cc/6Z2N-PFWC
"""
import math

import numpy as np

import gym
from gym import spaces
from gym.utils import seeding


class MountainCarEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self, goal_velocity=0, reward=None):
        self.min_position = -1.2
        self.max_position = 0.6
        self.max_speed = 0.07
        self.goal_position = 0.5
        self.goal_velocity = goal_velocity

        self.force = 0.001
        self.gravity = 0.0025

        self.reward_fn = reward

        self.low = np.array(
            [self.min_position, -self.max_speed], dtype=np.float32
        )
        self.high = np.array(
            [self.max_position, self.max_speed], dtype=np.float32
        )

        self.viewer = None

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            self.low, self.high, dtype=np.float32
        )

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        position, velocity = self.state
        velocity += (action - 1) * self.force + math.cos(3 * position) * (-self.gravity)
        velocity = np.clip(velocity, -self.max_speed, self.max_speed)
        position += velocity
        position = np.clip(position, self.min_position, self.max_position)
        if (position == self.min_position and velocity < 0):
            velocity = 0

        done = bool(
            position >= self.goal_position and velocity >= self.goal_velocity
        )
        reward = self.reward_fn.calculateReward()

        self.state = (position, velocity)
        return np.array(self.state), reward, done, {}

    def reset(self):
        self.state = np.array([self.np_random.uniform(low=-0.6, high=-0.4), 0])
        return np.array(self.state)

    def _height(self, xs):
        return np.sin(3 * xs) * .45 + .55

    def render(self, mode='human'):
        screen_width = 600
        screen_height = 400

        world_width = self.max_position - self.min_position
        scale = screen_width / world_width
        carwidth = 40
        carheight = 20

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            xs = np.linspace(self.min_position, self.max_position, 100)
            ys = self._height(xs)
            xys = list(zip((xs - self.min_position) * scale, ys * scale))

            self.track = rendering.make_polyline(xys)
            self.track.set_linewidth(4)
            self.viewer.add_geom(self.track)

            clearance = 10

            l, r, t, b = -carwidth / 2, carwidth / 2, carheight, 0
            car = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            car.add_attr(rendering.Transform(translation=(0, clearance)))
            self.cartrans = rendering.Transform()
            car.add_attr(self.cartrans)
            self.viewer.add_geom(car)
            frontwheel = rendering.make_circle(carheight / 2.5)
            frontwheel.set_color(.5, .5, .5)
            frontwheel.add_attr(
                rendering.Transform(translation=(carwidth / 4, clearance))
            )
            frontwheel.add_attr(self.cartrans)
            self.viewer.add_geom(frontwheel)
            backwheel = rendering.make_circle(carheight / 2.5)
            backwheel.add_attr(
                rendering.Transform(translation=(-carwidth / 4, clearance))
            )
            backwheel.add_attr(self.cartrans)
            backwheel.set_color(.5, .5, .5)
            self.viewer.add_geom(backwheel)
            flagx = (self.goal_position-self.min_position) * scale
            flagy1 = self._height(self.goal_position) * scale
            flagy2 = flagy1 + 50
            flagpole = rendering.Line((flagx, flagy1), (flagx, flagy2))
            self.viewer.add_geom(flagpole)
            flag = rendering.FilledPolygon(
                [(flagx, flagy2), (flagx, flagy2 - 10), (flagx + 25, flagy2 - 5)]
            )
            flag.set_color(.8, .8, 0)
            self.viewer.add_geom(flag)

        pos = self.state[0]
        self.cartrans.set_translation(
            (pos-self.min_position) * scale, self._height(pos) * scale
        )
        self.cartrans.set_rotation(math.cos(3 * pos))

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def get_keys_to_action(self):
        # Control with left and right arrow keys.
        return {(): 1, (276,): 0, (275,): 2, (275, 276): 1}

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None