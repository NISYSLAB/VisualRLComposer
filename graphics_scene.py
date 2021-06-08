import math

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *


class QDMGraphicsScene(QGraphicsScene):
    def __init__(self, scene, parent=None):
        super().__init__(parent)

        self.scene = scene

        self.grid_size = 20
        self.grid_Square = 5

        self._bckgr_color = QColor("#393939")
        self._grid_color = QColor("#2f2f2f")
        self._in_grid_color = QColor("#292929")

        self._pen_grid = QPen(self._grid_color)
        self._pen_grid.setWidth(1)

        self._pen_in_grid = QPen(self._in_grid_color)
        self._pen_in_grid.setWidth(2)

        self.setBackgroundBrush(self._bckgr_color)

    def setGrScene(self, width, height):
        self.setSceneRect(-width // 2, -height // 2, width, height)

    def drawBackground(self, painter, rect):
        super().drawBackground(painter, rect)

        # create the grid
        left = int(math.floor(rect.left()))
        right = int(math.ceil(rect.right()))
        top = int(math.floor(rect.top()))
        bottom = int(math.ceil(rect.bottom()))

        first_left = left - (left % self.grid_size)
        first_top = top - (top % self.grid_size)

        # compute the lines
        lines_grid, lines_in_grid = [], []
        for x in range(first_left, right, self.grid_size):
            if (x % (self.grid_size * self.grid_Square) != 0):
                lines_grid.append(QLine(x, top, x, bottom))
            else:
                lines_in_grid.append(QLine(x, top, x, bottom))

        for y in range(first_top, bottom, self.grid_size):
            if (y % (self.grid_size * self.grid_Square) != 0):
                lines_grid.append(QLine(left, y, right, y))
            else:
                lines_in_grid.append(QLine(left, y, right, y))

        # draw the lines
        painter.setPen(self._pen_grid)
        painter.drawLines(*lines_grid)

        painter.setPen(self._pen_in_grid)
        painter.drawLines(*lines_in_grid)
