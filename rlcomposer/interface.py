from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *

from .window_widget import RLComposerWindow
from .tensorboard_widget import Tensorboard
from .plot_widget import TestingWidgetPlot, TrainingWidgetPlot
from .custom_network_widget import NetConfigWidget
from .treeview_widget import FunctionTree
from .rl.instance import Instance
from .plot_button import PlotButton
import os
import numpy as np
DEBUG = True


class InstanceWorkerSignals(QObject):
    finished = pyqtSignal()


class InstanceWorker(QRunnable):
    def __init__(self, fn, start_fn, stop_fn):
        super(InstanceWorker, self).__init__()
        self.continue_run = True  # provide a bool run condition for the class
        self.start_run = True
        self.pause_f = False
        self.fn = fn
        self.start_fn = start_fn
        self.stop_fn = stop_fn
        self.signals = InstanceWorkerSignals()

    def run(self):
        i = 0
        self.start_fn()
        #try:
        #    self.start_fn()
        #except Exception as e:
        #    print(e)
        #    self.stop_fn()
        while self.start_run:
            QThread.msleep(0)
        while self.continue_run:  # give the loop a stoppable condition
            self.fn(i)
            i = i + 1
            while self.pause_f:
                QThread.msleep(0)

        self.stop_fn()
        self.signals.finished.emit()  # emit the finished signal when the loop is done

    def start(self):
        self.start_run = False

    def pause(self):
        self.pause_f = True

    def cont(self):
        self.pause_f = False

    def stop(self):
        self.continue_run = False  # set the run condition to false on stop
        print("Finish signal emitted")


class WorkerSignals(QObject):
    progress = pyqtSignal(int)
    url = pyqtSignal(str)
    finished_value = True


class Worker(QRunnable):
    def __init__(self, fn):
        super(Worker, self).__init__()
        self.signals = WorkerSignals()
        self.fn = fn

    @pyqtSlot()
    def run(self):
        self.fn(self.signals)

    def stop(self):
        self.signals.finished_value = False


class Plot(QRunnable):
    def __init__(self, fn):
        super(Plot, self).__init__()
        self.fn = fn
        self.start_run = True

    @pyqtSlot()
    def run(self):
        while self.start_run:
            self.fn.refresh()

    def stop(self):
        self.start_run = False


class Interface(QWidget):

    def __init__(self, parent):
        super(Interface, self).__init__(parent=parent)
        self.worker = None
        self.threadpool = QThreadPool()
        self.threadpool.setExpiryTimeout(-1)
        self.fname = None
        self.instance = None
        self.p = True
        # self.display = ImageDisplay(self)
        self.initUI()
        self.createLayout()

    def initUI(self):
        self.window_widget = RLComposerWindow(self)
        self.window_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        # self.window_widget.scene.addIsModifiedListener(self.createTitle)

        self.tensorboard = Tensorboard()

        self.testing_reward_widget = TestingWidgetPlot(name="Reward")
        self.testing_state_widget = TestingWidgetPlot(name="State")
        self.testing_action_widget = TestingWidgetPlot(name="Action")
        self.training_reward_widget = TrainingWidgetPlot(name="Reward")
        self.training_action_widget = TrainingWidgetPlot(name="Action")

        self.tree = FunctionTree(self.window_widget)

        self.netconf = NetConfigWidget(self, '')

        self.plot_tab = QTabWidget(self)
        # self.test_plot_widgets = TestPlots(self.reward_plot_widget, self.action_plot_widget, self.state_plot_widget)
        self.plot_tab.addTab(self.tensorboard, 'Tensorboard')
        self.plot_button_widgets = PlotButton(self.testing_reward_widget, self.testing_action_widget, self.testing_state_widget,
                                                       self.training_reward_widget, self.training_action_widget)
        self.plot_button_widgets.set_training_buttons(False)
        self.plot_button_widgets.set_testing_buttons(False)
        self.plot_tab.addTab(self.plot_button_widgets, "Plots")
        self.plot_tab.addTab(self.netconf, "Custom Network")
        self.plot_tab.currentChanged.connect(self.onTabChange)

        self.plot_tab_reload = QToolButton(self)
        self.plot_tab_reload.setIcon(QIcon('assets/refresh.svg'))
        self.plot_tab.setCornerWidget(self.plot_tab_reload)
        self.plot_tab_reload.clicked.connect(self.tensorboard.reload)

        self.img_view = QLabel(self)
        self.data = np.random.rand(256, 256)
        qimage = QImage(self.data, self.data.shape[0], self.data.shape[1], QImage.Format_RGB32)
        self.pixmap = QPixmap(qimage)
        self.img_view.setPixmap(self.pixmap)
        self.tree.status.setText("Status:  Create an Instance")

        self.pauseButton = QPushButton("Pause/Continue", self)
        self.pauseButton.clicked.connect(self.pauseContinue)
        self.pauseButton.setEnabled(False)

        self.saveModelButton = QPushButton("Save Model", self)
        self.saveModelButton.clicked.connect(self.saveModel)
        self.saveModelButton.setEnabled(False)

        self.createButton = QPushButton("Create Instance", self)
        self.createButton.clicked.connect(self.createInstance)

        self.trainButton = QPushButton("Train Instance", self)
        self.trainButton.clicked.connect(self.trainThread)
        self.trainButton.setEnabled(False)

        self.testButton = QPushButton("Test Instance", self)
        self.testButton.clicked.connect(self.testThread)
        self.testButton.setEnabled(False)

        self.closeButton = QPushButton("Close Instance", self)
        self.closeButton.clicked.connect(self.closeInstanceButton)
        self.closeButton.setEnabled(False)



        # self.createTitle()

    def createLayout(self):
        layout = QGridLayout(self)
        layout.setRowStretch(0, 6)
        layout.setRowStretch(1, 4)
        layout.setRowStretch(2, 1)
        layout.setColumnStretch(0, 70)
        layout.setColumnStretch(1, 5)
        layout.setColumnStretch(2, 5)
        layout.setColumnStretch(3, 5)
        layout.setColumnStretch(4, 5)
        layout.setColumnStretch(5, 5)
        layout.setColumnStretch(6, 5)

        layout.addWidget(self.window_widget, 0, 0)
        layout.addWidget(self.plot_tab, 1, 0, 2, 1)
        layout.addWidget(self.tree, 0, 1, 1, 6)
        layout.addWidget(self.img_view, 1, 1, 1, 6)
        layout.addWidget(self.createButton, 2, 1)
        layout.addWidget(self.trainButton, 2, 2)
        layout.addWidget(self.saveModelButton, 2, 3)
        layout.addWidget(self.testButton, 2, 4)
        layout.addWidget(self.pauseButton, 2, 5)
        layout.addWidget(self.closeButton, 2, 6)


    def threadComplete(self):
        print("Thread finished")
        self.threadpool.clear()

    def initInstance(self):
        self.tree.status.setText("Status:  Creating Instance")
        self.instance = Instance(self.window_widget.scene)
        self.n_envs = len(self.instance.env_wrapper_list)
        img = self.instance.prep()
        self.img_view.setPixmap(self.convertToPixmap(img))
        self.tree.status.setText("Status:  Instance Created")
        self.createButton.setEnabled(False)
        if not self.checkLoaded():
            self.trainButton.setEnabled(True)
        self.testButton.setEnabled(True)
        self.closeButton.setEnabled(True)
        self.pauseButton.setEnabled(False)

        if type(self.getSpaceNames(self.instance.env_wrapper_list[0].env_name)[2]) is tuple:
            self.netconf.signal.signal.emit('Cnn')
        else:
            self.netconf.signal.signal.emit('Mlp')

    def pauseContinue(self):
        if self.p:
            self.p = False
            self.tree.status.setText("Status:  Paused")
            self.test_worker.pause()
        else:
            self.p=True
            self.tree.status.setText("Status:  Testing in Progress")
            self.test_worker.cont()


    def saveModel(self):
        fname, filt = QFileDialog.getSaveFileName(self, "Save Model")
        if fname == "":
            return False
        self.instance.save(fname)

    def checkLoaded(self):
        scene = self.window_widget.scene
        for node in scene.nodes:
            if node.nodeType == "Load PreTrained Model":
                return True
        return False

    def createInstance(self):
        self.test_worker = InstanceWorker(self.testInstance, self.initInstance, self.closeInstance)
        self.test_worker.setAutoDelete(True)
        self.test_worker.signals.finished.connect(self.threadComplete)
        self.threadpool.start(self.test_worker)
        
        self.testing_reward_thread = Plot(self.testing_reward_widget)
        self.testing_state_thread = Plot(self.testing_state_widget)
        self.testing_action_thread = Plot(self.testing_action_widget)
        self.testing_reward_thread.setAutoDelete(True)
        self.testing_state_thread.setAutoDelete(True)
        self.testing_action_thread.setAutoDelete(True)

        self.training_reward_thread = Plot(self.training_reward_widget)
        self.training_action_thread = Plot(self.training_action_widget)
        self.training_reward_thread.setAutoDelete(True)
        self.training_action_thread.setAutoDelete(True)

    def closeInstanceButton(self):
        self.saveModelButton.setEnabled(False)
        self.closeButton.setEnabled(False)
        self.testButton.setEnabled(False)
        self.trainButton.setEnabled(False)
        self.pauseButton.setEnabled(False)
        self.createButton.setEnabled(True)
        self.plot_button_widgets.set_training_buttons(False)
        self.plot_button_widgets.set_testing_buttons(False)

        self.tree.status.setText("Status:  Create an Instance")
        self.img_view.setPixmap(self.pixmap)
        self.tensorboard.load(QUrl())

        try:
            self.training_reward_widget.clear_canvas()
            self.training_action_widget.clear_canvas()
            self.training_reward_thread.stop()
            self.training_action_thread.stop()
        except Exception as e:
            print("Training Widget clearing error", e)

        try:
            self.testing_reward_widget.clear_canvas()
            self.testing_action_widget.clear_canvas()
            self.testing_state_widget.clear_canvas()
            self.testing_reward_thread.stop()
            self.testing_state_thread.stop()
            self.testing_action_thread.stop()
        except Exception as e:
            print("Testing Widget clearing error", e)

        self.test_worker.pause()
        self.test_worker.stop()
        del self.test_worker
        try:
            self.worker.stop()
            self.worker.signals.progress.emit(0)
            del self.worker
        except Exception as e:
            print(e)

        try:
            self.instance._tensorboard_kill()
            del self.instance
        except Exception as e:
            print(e)

    def closeInstance(self):
        try:
            self.instance.env.close()
            self.initUI()
        except Exception as e:
            print(e)
            print("Create a Scene first!")
            self.createButton.setEnabled(True)
            self.testButton.setEnabled(False)
            self.closeButton.setEnabled(False)
            self.pauseButton.setEnabled(False)
            self.trainButton.setEnabled(False)

    def trainInstance(self, signal):
        self.tree.status.setText("Status:  Training in Progress")
        plots = [self.training_reward_widget, self.training_action_widget]
        self.instance.train_model(self.netconf.create_conf(), signal, plots)

        self.trainButton.setEnabled(False)
        self.saveModelButton.setEnabled(True)
        if not signal.finished_value:
            signal.progress.emit(0)
            self.saveModelButton.setEnabled(False)
        self.tree.status.setText("Status:  Training Finished")

    def testThread(self):
        self.threadpool.start(self.testing_reward_thread)
        self.threadpool.start(self.testing_state_thread)
        self.threadpool.start(self.testing_action_thread)
        self.testing_reward_widget.set_canvas(self.n_envs, ["Reward"])
        self.testing_state_widget.set_canvas(self.n_envs, self.getSpaceNames(self.instance.env_wrapper_list[0].env_name)[0])
        self.testing_action_widget.set_canvas(self.n_envs, self.getSpaceNames(self.instance.env_wrapper_list[0].env_name)[1])
        self.plot_button_widgets.set_testing_buttons(True)
        self.pauseButton.setEnabled(True)
        self.testButton.setEnabled(False)
        self.tree.status.setText("Status:  Testing in Progress")
        self.test_worker.start()

    def trainThread(self):
        #self.tensorboard.initial_load()
        self.threadpool.start(self.training_reward_thread)
        self.threadpool.start(self.training_action_thread)
        self.training_reward_widget.set_canvas(self.n_envs, ["Reward"])
        self.training_action_widget.set_canvas(self.n_envs, self.getSpaceNames(self.instance.env_wrapper_list[0].env_name)[1])
        self.plot_button_widgets.set_training_buttons(True)

        self.worker = Worker(self.trainInstance)
        self.worker.signals.progress.connect(self.tree.progress_bar_handler)
        self.worker.signals.url.connect(self.tensorboard.setURL)
        self.threadpool.start(self.worker)

    def testInstance(self, step):
        img, reward, done, action_probabilities, self.state, action = self.instance.step()
        self.img_view.setPixmap(self.convertToPixmap(img))
        print(f"Step {step}")
        self.testing_reward_widget.update_data(step, reward, ["Reward"])
        if self.getSpaceNames(self.instance.env_wrapper_list[0].env_name)[0] != "Invalid":
            self.testing_state_widget.update_data(step, self.state, self.getSpaceNames(self.instance.env_wrapper_list[0].env_name)[0])
        self.testing_action_widget.update_data(step, action, self.getSpaceNames(self.instance.env_wrapper_list[0].env_name)[1])

    def getSpaceNames(self, env_name):
        state_label, action_label, action_shape, observation_shape = [], [], None, None
        if env_name == "Pendulum":
            state_label = ["sin(theta)", "cos(theta)", "Velocity"]
            action_label = ["Torque"]
            observation_shape, action_shape = 3, 1

        elif env_name == "CartPoleEnv":
            state_label = ["Cart Position", "Cart Velocity", "Pole Angle", "Pole Angular Velocity"]
            action_label = ["0: Left, 1: Right"]
            observation_shape, action_shape = 4, 2

        elif env_name == "AcrobotEnv":
            state_label = ["cos(theta1)", "sin(theta1)", "cos(theta2)", "sin(theta2)", "Velocity of 1", "Velocity of 2"]
            action_label = ["Torque"]
            observation_shape, action_shape = 6, 3

        elif env_name == "Continuous_MountainCarEnv":
            state_label = ["Position", "Velocity"]
            action_label = ["Action"]
            observation_shape, action_shape = 2, 1

        elif env_name == "MountainCarEnv":
            state_label = ["Position", "Velocity"]
            action_label = ["Action"]
            observation_shape, action_shape = 2, 3

        elif env_name == "LunarLander":
            state_label = ["Coord-X", "Coord-Y", "Velocity-X", "Velocity-Y", "Angle", "Angular Velocity", "Left Leg Contact", "Right Leg Contact"]
            action_label = ["0:Nothing, 1:Fire Left, 2:Fire Main, 3:Fire Right"]
            observation_shape, action_shape = 8, 4

        elif env_name == "SokobanEnv":
            state_label = ["Invalid"]
            action_label = ["Action"]
            observation_shape, action_shape = (160, 160, 3), 9

        else:
            pass

        return state_label, action_label, observation_shape, action_shape

    def convertToPixmap(self, img):
        im = np.transpose(img, (0, 1, 2)).copy()
        im = QImage(im, im.shape[1], im.shape[0], im.shape[1] * 3, QImage.Format_RGB888)
        pixmap = QPixmap(im)
        return pixmap

    def onTabChange(self, i):  # changed!
        pass

    # def createTitle(self):
    #     title = "Visual RL Composer - "
    #     if self.fname is None:
    #         title += "New"
    #     else:
    #         title += os.path.basename(self.fname)
    #
    #     if self.window_widget.scene.is_modified:
    #         title += "*"
    #
    #     self.setWindowTitle(title)

    def kill(self):
        try:
            self.instance.env.close()
            del self.instance.model
        except:
            pass