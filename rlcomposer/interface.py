from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *

from PyQt5 import QtCore

from window_widget import RLComposerWindow
from tensorboard_widget import Tensorboard
from plot_widget import WidgetPlot
from custom_network_widget import NetConfigWidget
from treeview_widget import FunctionTree
import numpy as np
from rl.instance import Instance
import os

DEBUG = True


class InstanceWorker(QRunnable):

    def __init__(self, fn, start_fn, stop_fn):
        super(InstanceWorker, self).__init__()
        self.continue_run = True  # provide a bool run condition for the class
        self.start_run = True
        self.pause_f = False
        self.fn = fn
        self.start_fn = start_fn
        self.stop_fn = stop_fn
        self.signals = WorkerSignals()



    def run(self):
        i = 0

        self.start_fn()
        while self.start_run:
            QThread.msleep(0)
        while self.continue_run:  # give the loop a stoppable condition
            self.fn(i)
            QThread.msleep(100)
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
    finished = pyqtSignal()


class Interface(QWidget):

    def __init__(self, parent):
        super(Interface, self).__init__(parent=parent)
        self.worker = None
        self.threadpool = QThreadPool.globalInstance()
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

        self.plot_widget = WidgetPlot(self)

        self.netconf = NetConfigWidget(self, '')

        self.plot_tab = QTabWidget(self)
        self.plot_tab.addTab(self.tensorboard, 'Tensorboard')
        self.plot_tab.addTab(self.plot_widget, "Plots")
        self.plot_tab.addTab(self.netconf, "Custom Network")
        self.plot_tab.currentChanged.connect(self.onTabChange)

        self.tree = FunctionTree(self.window_widget.scene)

        self.img_view = QLabel(self)
        self.data = np.random.rand(256, 256)
        qimage = QImage(self.data, self.data.shape[0], self.data.shape[1], QImage.Format_RGB32)
        self.pixmap = QPixmap(qimage)
        self.img_view.setPixmap(self.pixmap)

        self.pauseButton = QPushButton("Pause/Continue", self)
        self.pauseButton.clicked.connect(self.pauseContinue)

        self.saveModelButton = QPushButton("Save Model", self)
        self.saveModelButton.clicked.connect(self.saveModel)

        self.createButton = QPushButton("Create Instance", self)
        self.createButton.clicked.connect(self.createInstance)

        self.trainButton = QPushButton("Train Instance", self)
        self.trainButton.clicked.connect(self.trainInstance)

        self.testButton = QPushButton("Test Instance", self)
        self.testButton.clicked.connect(self.testThread)

        self.closeButton = QPushButton("Close Instance", self)
        self.closeButton.clicked.connect(self.closeInstanceButton)

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
        layout.addWidget(self.saveModelButton, 2,3)
        layout.addWidget(self.testButton, 2, 4)
        layout.addWidget(self.pauseButton, 2, 5)
        layout.addWidget(self.closeButton, 2, 6)


    def threadComplete(self):
        print("Thread finished")
        self.threadpool.clear()

    def initInstance(self):
        self.instance = Instance(self.window_widget.scene)
        img = self.instance.prep()
        self.img_view.setPixmap(self.convertToPixmap(img))

    def pauseContinue(self):
        if self.p:
            self.p = False
            self.test_worker.pause()
        else:
            self.p=True
            self.test_worker.cont()


    def saveModel(self):
        fname, filt = QFileDialog.getSaveFileName(self, "Save Model")
        if fname == "":
            return False
        self.instance.save(fname)

    def createInstance(self):
        self.plot_widget.canvas.set_data()
        self.test_worker = InstanceWorker(self.testInstance, self.initInstance, self.closeInstance)
        self.test_worker.setAutoDelete(True)
        self.test_worker.signals.finished.connect(self.threadComplete)

        self.threadpool.start(self.test_worker)

    def closeInstanceButton(self):
        self.test_worker.cont()
        self.test_worker.stop()

    def closeInstance(self):
        self.test_worker.stop()
        self.instance.env.close()
        del self.instance
        del self.test_worker
        self.img_view.setPixmap(self.pixmap)


    def trainInstance(self):
        self.instance.train_model()
        pass

    def testThread(self):
        self.test_worker.start()

    def testInstance(self, step):
        img, reward, done, action_probabilities = self.instance.step()
        self.img_view.setPixmap(self.convertToPixmap(img))
        print("step")
        self.plot_widget.canvas.update_plot(step, reward)

    def stepThread(self):
        pass

    def convertToPixmap(self, img):
        im = np.transpose(img, (0, 1, 2)).copy()
        im = QImage(im, im.shape[1], im.shape[0], im.shape[1] * 3, QImage.Format_RGB888)
        pixmap = QPixmap(im)
        return pixmap

    def onTabChange(self, i):  # changed!
        QMessageBox.information(self,
                                "Tab Index Changed!",
                                "Current Tab Index: %d" % i)  # changed!


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