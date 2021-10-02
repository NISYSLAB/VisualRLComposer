from stable_baselines3 import *
from tensorboard_callbacks import Callback
import sys, subprocess
from tensorboard import program

DEBUG = False

def disable_view_window():
  from gym.envs.classic_control import rendering
  org_constructor = rendering.Viewer.__init__

  def constructor(self, *args, **kwargs):
    org_constructor(self, *args, **kwargs)
    self.window.set_visible(visible=False)

  rendering.Viewer.__init__ = constructor



class Instance():
    def __init__(self, scene):
        self.scene = scene
        self.env_wrapper, self.reward_wrapper, self.model_wrapper = None, None, None
        self.env = None
        self.model = None
        self.tensorboard_log = None
        self.buildInstance()


    def buildInstance(self):
        disable_view_window()
        current_env = None
        for item in self.scene.nodes:
            if item.title == "Environment":
                current_env = item.wrapper.env
        for item in self.scene.nodes:
            if item.title == "Environment":
                self.env_wrapper = item.wrapper
            elif item.title == "Reward":
                self.reward_wrapper = item.wrapper
            elif item.title == "Models":
                self.model_wrapper = item.wrapper
                self.model_wrapper.setModel(current_env)
        self.reward_func = self.reward_wrapper.reward
        self.env_wrapper.setReward(self.reward_func)
        self.env = self.env_wrapper.env
        self.model = self.model_wrapper.model
        self.tensorboard_log = self.env_wrapper.env_name + "_" +  self.model_wrapper.model_name
        print(self.model)


    def train_model(self):
        self.model = self.model_wrapper.model
        setattr(self.model, "tensorboard_log", self.tensorboard_log)
        self.model.learn(self.model_wrapper.total_timesteps, callback=Callback())


    def step(self):
        action, _ = self.model.predict(self.state)
        action_probabilities = 0
        self.state, reward, done, _ = self.env.step(action)
        img = self.env.render(mode="rgb_array")

        return img, reward, done, action_probabilities, self.state, action

    def prep(self):
        if DEBUG: print(self.env)
        self.state = self.env.reset()
        if DEBUG: print("resetted")
        img = self.env.render(mode="rgb_array")
        if DEBUG: print(type(img))
        return img

    def save(self, filename):
        self.model.save(filename)

    def removeInstance(self):
        pass

    def tensorboard(self, browser=True):
        # Kill current session
        self._tensorboard_kill()
        # Open the dir of the current env
        if sys.platform == 'win32':
            try:
                tb = program.TensorBoard()
                tb.configure(argv=[None, '--logdir', self.tensorboard_log])
                url = tb.launch()
                cmd = ''
            except:
                pass
        else:
            cmd = 'tensorboard --logdir {} --port 6006'.format(self.tensorboard_log) #--reload_interval 1

        try:
            DEVNULL = open(os.devnull, 'wb')
            subprocess.Popen(cmd, shell=True, stdout=DEVNULL, stderr=DEVNULL)
        except:
            pass


    def _tensorboard_kill(self):
        """
        Destroy all running instances of tensorboard
        """
        print('Closing current session of tensorboard.')
        if sys.platform in ['win32', 'Win32']:
            try:
                os.system("taskkill /f /im tensorboard.exe")
                os.system('taskkill /IM "tensorboard.exe" /F')
            except:
                pass
        elif sys.platform in ['linux', 'linux', 'Darwin', 'darwin']:
            try:
                os.system('pkill tensorboard')
                os.system('killall tensorboard')
            except:
                pass

        else:
            print('No running instances of tensorboard.')