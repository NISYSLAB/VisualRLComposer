from PyQt5 import QtWidgets
import pyqtgraph as pg
import numpy as np

def make_grid_layout(widgets, x, y):
    layout = QtWidgets.QGridLayout()
    for row in range(x):

        for col in range(y):
            widget = widgets[row * y +col]
            layout.addWidget(widget, row, col, 1, 1)

    return layout

def make_hv_layout(widgets, orientation):
    layouts = dict(h=QtWidgets.QHBoxLayout, v=QtWidgets.QVBoxLayout)
    layout = layouts[orientation]()
    for widget in widgets:
        layout.addWidget(widget)

    return layout

class GroupBox(QtWidgets.QGroupBox):
    def __init__(self, name, parent, widgets=None, orientation='h', grid=[]):
        super(GroupBox, self).__init__(name, parent=parent)
        self.par = parent
        self.setFlat(True)
        self.setStyleSheet("QGroupBox {" \
                           "text-align: center;" \
                           "font-weight: bold;" \
                           "font-size: 13px;}" \
                           "QGroupBox::title {" \
                           "background-color: transparent;" \
                           "color: rgb(0, 0, 0);" \
                           "subcontrol-position: top center;"
                           "} ")

        self.lay = None
        if widgets is not None:
            self.add_widgets(widgets, orientation, grid)

    def add_widgets(self, widgets, orientation, grid):
        if len(grid) > 1:
            x, y = grid
            self.setLayout(make_grid_layout(widgets, x, y))
        else:
            self.setLayout(make_hv_layout(widgets, orientation))



import cv2
import numpy as np

class ImageDisplay(GroupBox):
    def __init__(self, parent, size=(200, 200), clean=True):
        self.display = pg.ImageView()
        self.display.getView().getViewWidget().setMinimumSize(*size)
        self.display.getImageItem()
        self.display.setImage(np.random.randn(256,256,3))
        super(ImageDisplay, self).__init__('Environment Display', parent, [self.display])
        if clean:

            for element in ['menuBtn', 'roiBtn', 'histogram']:
                getattr(self.display.ui, element).hide()

    def update_image(self, image=None):
        if image is not None:
            self.display.setImage(image)
