from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *


class QDMGraphicsSocket(QGraphicsItem):
    def __init__(self, socket):
        self.socket = socket
        super().__init__(socket.node.grNode)

        self.radius = 6.0
        self.outline_width = 1.0

        self._colors = [QColor("#FFFF7700"), QColor("#FAA7700"), QColor("#FDD7700"), QColor("#EEEF7700")]

        self._color_circle = self._colors[0]
        self._color_outline = QColor("#FF000000")

        self._pen = QPen(self._color_outline)
        self._pen.setWidthF(self.outline_width)
        self._brush = QBrush(self._color_circle)

    def paint(self, painter, QStyleOptionGraphicsItem, widget=None):
        painter.setBrush(self._brush)
        painter.setPen(self._pen)
        painter.drawEllipse(-self.radius, -self.radius, 2 * self.radius, 2 * self.radius)

    def boundingRect(self):
        return QRectF(-self.radius - self.outline_width, -self.radius - self.outline_width,
                      2 * (self.radius + self.outline_width), 2 * (self.radius + self.outline_width))

    def mousePressEvent(self, QGraphicsSceneMouseEvent):
        pass
