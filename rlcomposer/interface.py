from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *

from .window_widget import RLComposerWindow
from .tensorboard_widget import Tensorboard
from .treeview_widget import FunctionTree
from .rl.instance import Instance
from .runtime_settings import RuntimeSettingsWindow
import os
import numpy as np

DEBUG = True


class InstanceWorkerSignals(QObject):
    finished = pyqtSignal()


class InstanceWorker(QRunnable):
    def __init__(self, fn, start_fn, stop_fn):
        super(InstanceWorker, self).__init__()
        self.start_run = True
        self.fn = fn
        self.start_fn = start_fn
        self.stop_fn = stop_fn
        self.signals = InstanceWorkerSignals()

    def run(self):
        self.start_fn()
        # try:
        #    self.start_fn()
        # except Exception as e:
        #    print(e)
        #    self.stop_fn()
        while self.start_run:
            QThread.msleep(0)

        self.fn()
        self.signals.finished.emit()
        self.stop_fn()

    def start(self):
        self.start_run = False

    def stop(self):
        self.continue_run = False  # set the run condition to false on stop
        print("Finish signal emitted")


class Interface(QWidget):

    def __init__(self, parent):
        super(Interface, self).__init__(parent=parent)
        self.worker = None
        self.threadpool = QThreadPool()
        self.threadpool.setExpiryTimeout(-1)
        self.fname = None
        self.instance = None
        self.initUI()
        self.createLayout()

    def initUI(self):
        self.window_widget = RLComposerWindow(self)
        self.window_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        # self.window_widget.scene.addIsModifiedListener(self.createTitle)

        self.tensorboard = Tensorboard()

        self.tree = FunctionTree(self.window_widget)

        self.runtime_param = {"Iterations": 2, "Use Futures": True, "Max MSG Size": 3000000, "Samples": []}
        self.settings_window = RuntimeSettingsWindow(self)

        self.plot_tab = QTabWidget(self)
        self.plot_tab.addTab(self.tensorboard, 'Tensorboard')
        self.plot_tab.currentChanged.connect(self.onTabChange)

        self.plot_tab_reload = QToolButton(self)
        self.plot_tab_reload.setIcon(QIcon('rlcomposer/rl/assets/refresh.svg'))
        self.plot_tab.setCornerWidget(self.plot_tab_reload)
        self.plot_tab_reload.clicked.connect(self.tensorboard.reload)

        self.tree.status.setText("Status:  Create an Instance")

        self.createButton = QPushButton("Create Instance", self)
        self.createButton.clicked.connect(self.createInstance)

        self.runtimeSettingsButton = QPushButton("Runtime Settings", self)
        self.runtimeSettingsButton.clicked.connect(self.runtimeSettings)
        self.runtimeSettingsButton.setEnabled(False)

        self.startRuntimeButton = QPushButton("Start Runtime", self)
        self.startRuntimeButton.clicked.connect(self.trainThread)
        self.startRuntimeButton.setEnabled(False)

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
        layout.setColumnStretch(1, 8)
        layout.setColumnStretch(2, 8)
        layout.setColumnStretch(3, 8)
        layout.setColumnStretch(4, 8)

        layout.addWidget(self.window_widget, 0, 0)
        layout.addWidget(self.plot_tab, 1, 0, 2, 1)
        layout.addWidget(self.tree, 0, 1, 2, 4)
        # layout.addWidget(self.line, 1, 1, 1, 4)
        layout.addWidget(self.createButton, 2, 1)
        layout.addWidget(self.runtimeSettingsButton, 2, 2)
        layout.addWidget(self.startRuntimeButton, 2, 3)
        layout.addWidget(self.closeButton, 2, 4)

    def threadComplete(self):
        print("Thread finished")
        self.threadpool.clear()

    def initInstance(self):
        self.tree.status.setText("Status:  Creating Instance")
        self.instance = Instance(self.window_widget)

        self.tree.status.setText("Status:  Instance Created")
        self.createButton.setEnabled(False)
        self.startRuntimeButton.setEnabled(True)
        self.runtimeSettingsButton.setEnabled(True)
        self.closeButton.setEnabled(True)

    def runtimeSettings(self):
        # self.settings_window = RuntimeSettingsWindow(self)
        self.settings_window.show()

    def removeWindow(self, r_dict):
        # self.settings_window = None
        self.runtime_param = r_dict

    def createInstance(self):
        self.test_worker = InstanceWorker(self.trainInstance, self.initInstance, self.closeInstance)
        self.test_worker.setAutoDelete(True)
        self.test_worker.signals.finished.connect(self.threadComplete)
        self.threadpool.start(self.test_worker)

    def closeInstanceButton(self):
        self.runtimeSettingsButton.setEnabled(False)
        self.closeButton.setEnabled(False)
        self.startRuntimeButton.setEnabled(False)
        self.createButton.setEnabled(True)

        self.tree.status.setText("Status:  Create an Instance")
        self.tensorboard.load(QUrl())

        try:
            del self.test_worker
        except:
            pass

        try:
            self.instance._tensorboard_kill()
            del self.instance
        except Exception as e:
            pass
            # print(e)

    def closeInstance(self):
        try:
            del self.instance
        except:
            pass

    def trainInstance(self):
        self.tree.status.setText("Status:  Starting Runtime")
        self.instance.start_runtime(self.runtime_param)

        self.startRuntimeButton.setEnabled(False)
        self.runtimeSettingsButton.setEnabled(False)
        self.tree.status.setText("Status:  Finished")

    def trainThread(self):
        self.test_worker.start()

    def onTabChange(self, i):  # changed!
        pass

    def kill(self):
        try:
            self.instance.env.close()
            del self.instance.model
        except:
            pass