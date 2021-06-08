from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *


class QDMGraphicsNode(QGraphicsItem):
    def __init__(self, node, parent=None):
        super().__init__(parent)

        self.node = node
        self.content = self.node.content
        self.content.push.clicked.connect(self.printMe)

        self._title_color = Qt.white


        self.width = 250
        self.height = 300

        self.edge_size = 5
        self.title_height = 22
        self._padding = 10.0

        self._pen_default = QPen(QColor("#7F000000"))
        self._pen_selected = QPen(QColor("#FFFFA637"))

        # initializign title of the node
        self.initTitle()
        self.title = self.node.title

        # initializing content of the node
        self.initContent()

        # initializing sockets
        self.initSockets()

        self.initUI()

        self.moved = False

        self._title_brush = QBrush(QColor("#FF313131"))
        self._background_brush = QBrush(QColor("#E3313131"))

    def printMe(self):
        print("Input Nodes for ", self.node.title, "are: ", self.node.inputNodes)
        print("Output Nodes for ", self.node.title, "are: ", self.node.outputNodes)

    @property
    def title(self):
        return self._title

    @title.setter
    def title(self, value):
        self._title = value
        self.title_item.setPlainText(self._title)

    def boundingRect(self):
        return QRectF(0, 0, 2 * self.edge_size + self.width, 2 * self.edge_size + self.height).normalized()

    def initUI(self):
        self.setFlag(QGraphicsItem.ItemIsSelectable)
        self.setFlag(QGraphicsItem.ItemIsMovable)

    def initContent(self):
        self.grContent = QGraphicsProxyWidget(self)
        self.content.setGeometry(self.edge_size, self.title_height + self.edge_size, self.width - 2 * self.edge_size,
                                 self.height - 2 * self.edge_size - self.title_height)
        self.grContent.setWidget(self.content)

    def initSockets(self):
        pass

    def initTitle(self):
        self.title_item = QGraphicsTextItem(self)
        self.title_item.node = self.node
        self.title_item.setDefaultTextColor(self._title_color)
        self.title_item.setPos(self._padding, 0)
        self.title_item.setTextWidth(self.width - 2 * self._padding)

    def mouseMoveEvent(self, event):
        super().mouseMoveEvent(event)
        for node in self.scene().scene.nodes:
            if node.grNode.isSelected():
                node.updateConnectedEdges()
        self.moved = True

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)

        if self.moved:
            self.moved = False
            self.node.scene.history.storeHistory(str(self.title) + " moved")

    def paint(self, painter, QStyleOptionGraphicsItem, widget=None):
        title = QPainterPath()
        title.setFillRule(Qt.WindingFill)
        title.addRoundedRect(0, 0, self.width, self.title_height, self.edge_size, self.edge_size)
        title.addRect(0, self.title_height - self.edge_size, self.edge_size, self.edge_size)
        title.addRect(self.width - self.edge_size, self.title_height - self.edge_size, self.edge_size, self.edge_size)
        painter.setPen(Qt.NoPen)
        painter.setBrush(self._title_brush)
        painter.drawPath(title.simplified())

        back = QPainterPath()
        back.setFillRule(Qt.WindingFill)
        back.addRoundedRect(0, self.title_height, self.width, self.height - self.title_height, self.edge_size,
                            self.edge_size)
        back.addRect(0, self.title_height, self.edge_size, self.edge_size)
        back.addRect(self.width - self.edge_size, self.title_height, self.edge_size, self.edge_size)
        painter.setPen(Qt.NoPen)
        painter.setBrush(self._background_brush)
        painter.drawPath(back.simplified())

        outline = QPainterPath()
        outline.addRoundedRect(0, 0, self.width, self.height, self.edge_size, self.edge_size)
        painter.setPen(self._pen_default if not self.isSelected() else self._pen_selected)
        painter.setBrush(Qt.NoBrush)
        painter.drawPath(outline.simplified())
