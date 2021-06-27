from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *

from window_widget import RLComposerWindow
from tensorboard_widget import Tensorboard
from plot_widget import MplCanvas
from custom_network_widget import NetConfigWidget
from treeview_widget import FunctionTree
import numpy as np
from rl.instance import Instance
import os


DEBUG = True

class Worker(QRunnable):

    def __init__(self, fn):
        super(Worker, self).__init__()
        self.continue_run = True  # provide a bool run condition for the class
        self.fn = fn
        self.signals = WorkerSignals()

    def run(self):
        i = 1
        while self.continue_run:  # give the loop a stoppable condition
            # print(i)
            self.fn()
            QThread.msleep(10)
            # i = i + 1
        self.signals.finished.emit()  # emit the finished signal when the loop is done

    def stop(self):
        self.continue_run = False  # set the run condition to false on stop
        print("Finish signal emitted")

class StepWorker(QRunnable):

    def __init__(self, fn):
        super(StepWorker, self).__init__()
        self.continue_run = True  # provide a bool run condition for the class
        self.fn = fn
        self.signals = WorkerSignals()

    def run(self):
        self.fn()
        self.signals.finished.emit()  # emit the finished signal when the loop is done

    def stop(self):
        print("Stop function")


class WorkerSignals(QObject):
    finished = pyqtSignal()
    error = pyqtSignal(tuple)
    result = pyqtSignal(object)
    progress = pyqtSignal(int)


class Interface(QWidget):
    stop_signal = pyqtSignal()
    def __init__(self, parent):
        super(Interface, self).__init__(parent=parent)
        self.worker = None
        self.threadpool = QThreadPool()
        self.fname = None
        # self.display = ImageDisplay(self)
        self.initUI()
        self.createLayout()

    def initUI(self):
        self.window_widget = RLComposerWindow(self)
        self.window_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.window_widget.scene.addIsModifiedListener(self.createTitle)

        self.tensorboard = Tensorboard()

        self.canvas = MplCanvas(self, self.window_widget.scene, width=5, height=4, dpi=100)

        self.netconf = NetConfigWidget(self, '')

        self.plot_tab = QTabWidget(self)
        self.plot_tab.addTab(self.tensorboard, 'Tensorboard')
        self.plot_tab.addTab(self.canvas, "Plots")
        self.plot_tab.addTab(self.netconf, "Custom Network")
        self.plot_tab.currentChanged.connect(self.onTabChange)

        self.tree = FunctionTree(self.window_widget.scene)

        self.img_view = QLabel(self)
        self.data = np.random.rand(256,256)
        qimage = QImage(self.data, self.data.shape[0], self.data.shape[1], QImage.Format_RGB32)
        pixmap = QPixmap(qimage)
        self.img_view.setPixmap(pixmap)

        # self.create = QPushButton("Kill Thread", self)
        # self.create.clicked.connect(self.stepThread)

        self.step = QPushButton("Step Instance", self)
        self.step.clicked.connect(self.stepThread)

        self.createTitle()

    def createLayout(self):
        layout = QGridLayout(self)
        layout.setRowStretch(0, 6)
        layout.setRowStretch(1, 4)
        layout.setRowStretch(2, 1)
        layout.setColumnStretch(0, 70)
        layout.setColumnStretch(1, 10)
        layout.setColumnStretch(2, 10)
        layout.setColumnStretch(3, 10)

        layout.addWidget(self.window_widget, 0, 0)
        layout.addWidget(self.plot_tab, 1, 0,2,1)
        layout.addWidget(self.tree, 0,1,1,3)
        layout.addWidget(self.img_view, 1, 1, 1,3)
        # layout.addWidget(self.create, 2, 2)
        layout.addWidget(self.step, 2, 3)

    # def killStepThread(self):
    #     self.

    def createInstance(self):
        if DEBUG: print("Inside createInstance")
        self.instance = Instance(self.window_widget.scene)
        img = self.instance.prep()
        im = np.transpose(img, (1, 0, 2)).copy()
        im = QImage(im, im.shape[1], im.shape[0], im.shape[1]*3, QImage.Format_RGB888)
        pixmap = QPixmap(im)
        self.img_view.setPixmap(pixmap)
        print("Create Instance array size:", img.shape)
        print(img)

    def stepThread(self):
        self.worker = StepWorker(self.stepInstance)
        self.stop_signal.connect(self.worker.stop)
        self.threadpool.start(self.worker)

    # def createThread(self):
    #     self.worker = StepWorker(self.createInstance)
    #     self.stop_signal.connect(self.worker.stop)
    #     self.threadpool.start(self.worker)


    def stepInstance(self):
        if self.window_widget.scene._parameter_updated:
            self.window_widget.scene._parameter_updated = False
            self.instance = Instance(self.window_widget.scene)
            img = self.instance.prep()
            im = np.transpose(img, (0, 1, 2)).copy()
            im = QImage(im, im.shape[1], im.shape[0], im.shape[1] * 3, QImage.Format_RGB888)
            pixmap = QPixmap(im)
            self.img_view.setPixmap(pixmap)
        else:
            if DEBUG: print("Inside stepInstance 1")
            img, reward, done, action_probabilities = self.instance.step()
            (width, height, channel) = img.shape
            if DEBUG: print("Inside stepInstance 3", img.shape)
            im = np.transpose(img, (0, 1, 2)).copy()
            im = QImage(im, im.shape[1], im.shape[0], im.shape[1] * 3, QImage.Format_RGB888)
            pixmap = QPixmap(im)
            self.img_view.setPixmap(pixmap)
            print(reward)

    def onTabChange(self,i): #changed!
        if i==1:
            self.setPlotThreads(self.canvas.update_plot)
        else:
            self.stop_signal.emit()
        QMessageBox.information(self,
                  "Tab Index Changed!",
                  "Current Tab Index: %d" % i ) #changed!

    def setPlotThreads(self, fn):
        self.worker = Worker(fn)
        self.stop_signal.connect(self.worker.stop)
        self.threadpool.start(self.worker)


    def createTitle(self):
        title = "Visual RL Composer - "
        if self.fname is None:
            title += "New"
        else:
            title += os.path.basename(self.fname)

        if self.window_widget.scene.is_modified:
            title += "*"

        self.setWindowTitle(title)