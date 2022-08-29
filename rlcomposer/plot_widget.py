from PyQt5.QtWidgets import QWidget, QVBoxLayout, QSizePolicy, QGridLayout, QTabWidget
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib
import pyqtgraph as pg
#matplotlib.use('Qt5Agg')


from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class TestingMplCanvas(FigureCanvas):
    def __init__(self, name, label, width=10, height=5, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi, constrained_layout=True)
        self.name = name
        self.axes = fig.add_subplot(111)
        super(TestingMplCanvas, self).__init__(fig)
        FigureCanvas.setSizePolicy(self, QSizePolicy.Expanding, QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        self.set_data(label)

    def set_data(self, label):
        self.xdata = []
        self.ydata = []
        self.label = label
        self.axes.cla()
        self.axes.set_title(self.name)
        self.axes.set_xlabel("Steps")

    def update_list_data(self, step, value, label):
        self.xdata.append(step)
        self.ydata.append(value)
        self.label = label

    def refresh_plot(self):
        self.axes.clear()
        self.axes.set_title(self.name)
        self.axes.set_xlabel("Steps")
        if len(self.label) == 1:
            self.label = self.label[0]
        self.axes.plot(self.xdata, self.ydata, label=self.label, marker="*")
        self.axes.legend(loc='lower right')
        self.axes.grid()
        if len(self.ydata) > 100:
            self.axes.set_xlim(len(self.ydata) - 100, len(self.ydata))
        else:
            self.axes.set_xlim(0, len(self.ydata))
        self.draw()


class TestingWidgetPlot(QWidget):
    def __init__(self, name):
        QWidget.__init__(self)
        self.name = name
        self.count = 0

        self.layout = QGridLayout()
        self.setLayout(self.layout)
        self.setWindowTitle(self.name + " Plot")

    def set_canvas(self, n_envs, label):
        self.plot_tab = QTabWidget()
        self.canvas = []
        self.layout.addWidget(self.plot_tab, 0, 0)
        for i in range(n_envs):
            self.canvas.append(TestingMplCanvas(name=self.name, label=label))
            self.count += 1
            self.plot_tab.addTab(self.canvas[-1], f"Environment {self.count}")

    def clear_canvas(self):
        self.canvas = []
        self.count = 0
        self.plot_tab.clear()

    def update_data(self, step, value, label):
        for i in range(self.count):
            self.canvas[i].update_list_data(step, value[i], label)

    def refresh(self):
        for i in range(self.count):
            self.canvas[i].refresh_plot()




class TrainingMplCanvas(FigureCanvas):
    def __init__(self, name, label=None, width=10, height=5, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi, constrained_layout=True)
        self.name = name
        self.axes = fig.add_subplot(111)
        super(TrainingMplCanvas, self).__init__(fig)
        FigureCanvas.setSizePolicy(self, QSizePolicy.Expanding, QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        self.set_data(label)

    def set_data(self, label):
        self.xdata = []
        self.ydata = []
        self.label = label
        self.axes.cla()
        self.axes.set_title(self.name)
        self.axes.set_xlabel("Steps")

    def update_list_data(self, step, value):
        self.xdata.append(step)
        self.ydata.append(value)

    def refresh_plot(self):
        self.axes.clear()
        self.axes.set_title(self.name)
        self.axes.set_xlabel("Steps")
        if len(self.label) == 1:
            self.label = self.label[0]
        self.axes.plot(self.xdata, self.ydata, label=self.label)
        self.axes.legend(loc='lower right')
        self.axes.grid()
        self.draw()


class TrainingWidgetPlot(QWidget):
    def __init__(self, name):
        QWidget.__init__(self)
        self.name = name
        self.count = 0

        self.layout = QGridLayout()
        self.setLayout(self.layout)
        self.setWindowTitle(self.name + " Plot")

    def set_canvas(self, n_envs, label):
        self.plot_tab = QTabWidget()
        self.canvas = []
        self.layout.addWidget(self.plot_tab, 0, 0)
        for i in range(n_envs):
            self.canvas.append(TrainingMplCanvas(name=self.name, label=label))
            self.count += 1
            self.plot_tab.addTab(self.canvas[-1], f"Environment {self.count}")

    def clear_canvas(self):
        self.canvas = []
        self.count = 0
        self.plot_tab.clear()

    def update_data(self, step, value):
        for i in range(self.count):
            self.canvas[i].update_list_data(step, value[i])

    def refresh(self):
        for i in range(self.count):
            self.canvas[i].refresh_plot()