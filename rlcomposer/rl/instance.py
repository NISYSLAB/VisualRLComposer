import sys, subprocess, os
from tensorboard import program
import posix_ipc
from runtime.process_runtime import POSIXMsgQueue, PPRuntime as Runtime
from qfdfg.graph import Graph
from rlcomposer.rl.component_wrapper import get_shape, to_bool
import numpy as np

DEBUG = False


class Instance():
    def __init__(self, window_widget):
        self.scene = window_widget.scene
        self.component_wrapper_list = []

        self.graph = Graph(window_widget.get_current_tab_name)
        self.runtime = None
        self.model = None
        self.tensorboard_log = None
        self.buildInstance()

    def buildInstance(self):
        for item in self.scene.nodes:
            self.component_wrapper_list.append(item.wrapper)
        # bind properties
        for wrapper in self.component_wrapper_list:
            wrapper.bindParameters()
        # print component info
        for wrapper in self.component_wrapper_list:
            wrapper.print_info()

        # add components
        for wrapper in self.component_wrapper_list:
            self.graph.add_component(wrapper.component)
        # add flows
        self.parse_flows()

    def parse_flows(self):
        for edge in self.scene.edges:
            start_node, end_node = None, None
            if DEBUG: print(edge.start_socket.id, edge.end_socket.id, edge.options)
            for node in self.scene.nodes:
                for output_node in node.outputs:
                    if output_node.id == edge.start_socket.id:
                        start_node = node
                        if DEBUG: print(start_node)
                for input_node in node.inputs:
                    if input_node.id == edge.end_socket.id:
                        end_node = node
                        if DEBUG: print(end_node)

            self.graph.add_flow(edge.value, start_node.wrapper.component, edge.value, end_node.wrapper.component)
            print(edge.value, start_node.wrapper.component, edge.value, end_node.wrapper.component)

    def start_runtime(self, runtime_param):
        print(runtime_param)
        self.runtime = Runtime(self.graph)
        self.runtime._iterations = int(runtime_param['Iterations'])
        self.runtime._use_futures = to_bool(runtime_param['Use Futures'])
        self.runtime.initialize(max_msg_count=self.runtime._iterations + 1,
                                max_msg_size=int(runtime_param['Max MSG Size']))

        for i in runtime_param['Samples']:
            tempdict = i.get_data()
            if DEBUG:  print(tempdict)
            qid = self.runtime._qid_table.get(tempdict['Name'])
            queue = POSIXMsgQueue(tempdict['Name'], tempdict['Shape'], tempdict['Type'], qid)
            mq = posix_ipc.MessageQueue(tempdict['Name'], posix_ipc.O_CREAT)
            queue.init_queue(mq)
            if len(tempdict['Push']) > 0:
                if DEBUG: print(tempdict['Push'])
                for k in range(self.runtime._iterations):
                    push = np.array(tempdict['Push'])
                    queue.push(push, tempdict['Shape'])

        self.runtime.execute()

    def removeInstance(self):
        pass

    def tensorboard(self, browser=True, folder=None):
        # Kill current session
        self._tensorboard_kill()
        print('New session of tensorboard.')
        # Open the dir of the current env
        print(self.tensorboard_log)
        self.url = 'Null'
        if False:  # sys.platform == 'win32':  tensorboard.program cannot close manually
            try:
                tb = program.TensorBoard()
                tb.configure(argv=[None, '--logdir', self.tensorboard_log + "/" + folder])
                self.url = tb.launch()
                cmd = ''
            except Exception as e:
                print("Tensorboard Error:", e)
                pass
        else:
            cmd = 'tensorboard --logdir {} --reload_multifile true --reload_interval 2'.format(
                self.tensorboard_log + "/" + folder)  # --reload_interval 1

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