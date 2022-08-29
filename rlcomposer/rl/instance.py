from stable_baselines3 import *
import stable_baselines3.common.logger as logger
from stable_baselines3.common.utils import get_latest_run_id
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env import VecFrameStack
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
    def __init__(self, window_widget):
        self.scene = window_widget.scene
        self.component_wrapper_list = []
        from qfdfg.graph import Graph
        self.graph = Graph(window_widget.get_current_tab_name)
        self.runtime = None
        self.model = None
        self.tensorboard_log = None
        self.logger = logger
        self.buildInstance()

    def buildInstance(self):
        for item in self.scene.nodes:
            self.component_wrapper_list.append(item.wrapper)
        # bind properties
        for wrapper in self.component_wrapper_list:
            wrapper.bindProperties()
        # print component info
        for wrapper in self.component_wrapper_list:
            wrapper.print_info()
        # add components
        for wrapper in self.component_wrapper_list:
            self.graph.add_component(wrapper.component)
        # add flows
        self.parse_flows()
        from runtime.runtime import Runtime
        self.runtime = Runtime(self.graph)
        self.runtime.initialize()
        self.runtime.execute()

    def parse_flows(self):
        id_graph = []
        serialized = self.scene.serialize()
        for edge in self.scene.edges:
            start_node, end_node = None, None
            print(edge.start_socket.id, edge.end_socket.id)
            for node in self.scene.nodes:
                for output_node in node.outputs:
                    if output_node.id == edge.start_socket.id:
                        start_node = node
                        print(start_node)
                for input_node in node.inputs:
                    if input_node.id == edge.end_socket.id:
                        end_node = node
                        print(end_node)

            self.graph.add_flow(*start_node.wrapper.param['Output Name'], start_node.wrapper.component,
                                *end_node.wrapper.param['Input Name'], end_node.wrapper.component)


    def train_model(self, network, signal, plots):
        self.model_wrapper.add_parameters(network, self.tensorboard_log)
        self.model = self.model_wrapper.model
        print(getattr(self.model, "policy_kwargs"))

        self.tensorboard(browser=False, folder=str(self.model_wrapper.model_name + "_" + str(get_latest_run_id(self.tensorboard_log, self.model_wrapper.model_name)+1)))
        signal.url.emit(self.url)
        self.model.learn(total_timesteps=self.model_wrapper.total_timesteps, callback=Callback(self.tensorboard_log, signal, plots))
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
        self.url = 'Null'
        if False:    # sys.platform == 'win32':  tensorboard.program cannot close manually
            try:
                tb = program.TensorBoard()
                tb.configure(argv=[None, '--logdir', self.tensorboard_log+"/"+folder])
                self.url = tb.launch()
                cmd = ''
            except Exception as e:
                print("Tensorboard Error:", e)
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