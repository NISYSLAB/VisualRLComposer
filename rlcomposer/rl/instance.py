from stable_baselines3 import *
import stable_baselines3.common.logger as logger
from stable_baselines3.common.utils import get_latest_run_id
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
import stable_baselines3.common.callbacks
from .tensorboard_callbacks import Callback
import sys, subprocess, webbrowser, os
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
        self.env_wrapper_list, self.reward_wrapper, self.model_wrapper = [], None, None
        self.env = None
        self.model = None
        self.tensorboard_log = None
        self.logger = logger
        self.buildInstance()


    def buildInstance(self):
        disable_view_window()
        for item in self.scene.nodes:
            if item.title == "Environment":
                self.env_wrapper_list.append(item.wrapper)
            elif item.title == "Reward":
                self.reward_wrapper = item.wrapper
            elif item.title == "Models":
                self.model_wrapper = item.wrapper

        self.reward_func = self.reward_wrapper.reward
        for env_wrapper in self.env_wrapper_list:
            env_wrapper.setReward(self.reward_func)
        self.env = SubprocVecEnv([env_wrapper.callable_env() for env_wrapper in self.env_wrapper_list])
        self.model_wrapper.setModel(self.env)
        self.tensorboard_log = self.env_wrapper_list[0].env_name + "_" + self.model_wrapper.model_name
        setattr(self.model_wrapper.model, "tensorboard_log", self.tensorboard_log)
        self.model = self.model_wrapper.model
        if self.scene.model_archive is not None:
            self.model = self.scene.model_archive


    def train_model(self, network, signal):
        self.model = self.model_wrapper.model
        setattr(self.model, "policy_kwargs", network)
        print(getattr(self.model, "policy_kwargs"))
        self.tensorboard(browser=False, folder=str(self.model_wrapper.model_name + "_" + str(get_latest_run_id(self.tensorboard_log, self.model_wrapper.model_name)+1)))
        signal.url.emit(self.url)
        self.model.learn(total_timesteps=self.model_wrapper.total_timesteps, callback=Callback(self.tensorboard_log, signal))
        if not signal.finished_value:
            signal.progress.emit(0)
        self.scene.model_archive = self.model

    def step(self):
        action, _ = self.model.predict(self.state)
        action_probabilities = 0
        self.state, reward, done, _ = self.env.step(action)
        img = self.env.render(mode="rgb_array")

        return img, reward, done, action_probabilities, self.state, action

    def prep(self):
        if DEBUG: print(self.env)
        print(len(self.env_wrapper_list))
        self.state = self.env.reset()
        if DEBUG: print("resetted")
        self.env.env_method('set_render', len(self.env_wrapper_list))
        img = self.env.render(mode="rgb_array")
        if DEBUG: print(type(img))
        return img

    def save(self, filename):
        self.model.save(filename)

    def removeInstance(self):
        pass

    def tensorboard(self, browser=True, folder = None):
        # Kill current session
        self._tensorboard_kill()
        print('New session of tensorboard.')
        # Open the dir of the current env
        print(self.tensorboard_log)
        if sys.platform == 'win32':
            try:
                tb = program.TensorBoard()
                tb.configure(argv=[None, '--logdir', self.tensorboard_log+"/"+folder])
                self.url = tb.launch()
                cmd = ''
            except Exception as e:
                print(e)
                pass
        else:
            cmd = 'tensorboard --logdir {} --reload_multifile true --reload_interval 2'.format(self.tensorboard_log+"/"+folder)  # --reload_interval 1

        try:
            DEVNULL = open(os.devnull, 'wb')
            subprocess.Popen(cmd, shell=True, stdout=DEVNULL, stderr=DEVNULL)
        except:
            print('Tensorboard Error')
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