
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QSizePolicy

from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib
matplotlib.use('Qt5Agg')


from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure




class MplCanvas(FigureCanvas):

    def __init__(self, name, width=5, height=4, dpi=100):

        fig = Figure(figsize=(width, height), dpi=dpi, constrained_layout=True)
        self.name = name
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

    def update_plot(self, step, value, label):
        self.xdata.append(step)
        self.ydata.append(value)
        self.axes.cla()
        self.update_names()
        if len(label) == 1:
            label = label[0]
        print(self.xdata, self.ydata)
        #self.axes.plot(self.xdata, self.ydata, label=label, marker="*")
        self.axes.legend(loc='lower right')
        self.axes.grid()
        if len(self.ydata) > 100:
            self.axes.set_xlim(len(self.ydata) - 100, len(self.ydata))
        else:
            self.axes.set_xlim(0, len(self.ydata))
        self.draw()

    def update_names(self):
        self.axes.set_title(self.name)
        self.axes.set_xlabel("Steps")

class WidgetPlot(QWidget):
    def __init__(self, name):
        QWidget.__init__(self)
        self.setLayout(QVBoxLayout())
        self.canvas = MplCanvas(name=name)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.layout().addWidget(self.toolbar)
        self.layout().addWidget(self.canvas)