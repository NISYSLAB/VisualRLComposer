
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QSizePolicy

from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib
matplotlib.use('Qt5Agg')


from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class MplCanvas(FigureCanvas):

    def __init__(self, parent=None, width=5, height=4, dpi=100):

        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)
        FigureCanvas.setSizePolicy(self, QSizePolicy.Expanding, QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        self.set_data()

    def set_data(self):

        self.xdata = []
        self.ydata = []
        self.axes.cla()
        self.update_names()

    def update_plot(self, step, reward):
        self.xdata.append(step)
        self.ydata.append(reward)
        self.axes.cla()
        self.update_names()
        self.axes.plot(self.xdata, self.ydata, 'r')
        self.draw()

    def update_names(self):
        self.axes.set_title("Reward Values")
        self.axes.set_xlabel("Steps")

class WidgetPlot(QWidget):
    def __init__(self, *args, **kwargs):
        QWidget.__init__(self, *args, **kwargs)
        self.setLayout(QVBoxLayout())
        self.canvas = MplCanvas(self)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.layout().addWidget(self.toolbar)
        self.layout().addWidget(self.canvas)