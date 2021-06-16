from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *


class QDMGraphicsEdge(QGraphicsPathItem):
    """
    Class for the graphical features of an edge object

    Attributes
    ----------
    _color: QColor class
        the color of the edge when not selected
    _color_sel: QColor class
        the color of the edge when selected
    _pen: QPen class
        the pen object that draw the edge line when not selected
    _pen_sel: QPen class
        the pen object that draw the edge line when selected
    edge: Edge class
        the edge object
    posEnd: list
        the x,y positions of the end point of the edge
    posSource: list
        the x,y positions of the starting point of the edge
    Methods
    -------
    setSource(x,y)
        Set the x,y positions of the starting point of the edge
    setEnd(x,y)
        Set the x,y positions of the end point of the edge
    paint(painter, QStyleOptionGraphicsItem, widget=None)
        Visually draw and paints the edge
    updatePath()
        Draw and update the edge path
    """


    def __init__(self, edge, parent=None):
        super().__init__(parent)

        self.edge = edge
        self._color = QColor("#001000")
        self._color_sel = Qt.magenta

        self._pen = QPen(self._color)
        self._pen.setWidthF(2.0)
        self._pen_sel = QPen(self._color_sel)

        self.setFlag(QGraphicsItem.ItemIsSelectable)
        self.setZValue(-1)

        self.posSource = [0, 0]
        self.posEnd = [100, 100]

    def setSource(self, x, y):
        self.posSource = [x, y]

    def setEnd(self, x, y):
        self.posEnd = [x, y]

    def paint(self, painter, QStyleOptionGraphicsItem, widget=None):
        self.updatePath()

        painter.setPen(self._pen if not self.isSelected() else self._pen_sel)
        painter.setBrush(Qt.NoBrush)
        painter.drawPath(self.path())

    def updatePath(self):
        raise NotImplemented("This method has to be overriden in a child class")


class QDMGraphicsEdgeShaped(QDMGraphicsEdge):
    def updatePath(self):
        s = self.posSource
        d = self.posEnd
        dist = (d[0] - s[0]) * 0.5
        if s[0] > d[0]: dist *= -1

        path = QPainterPath(QPointF(self.posSource[0], self.posSource[1]))
        path.cubicTo(s[0] + dist, s[1], d[0] - dist, d[1], self.posEnd[0], self.posEnd[1])
        self.setPath(path)
