from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

from node import Node
from scene import Scene
from graphics_view import QDMGraphicsView


class RLComposerWindow(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.initUI()

    def initUI(self):
        self.setGeometry(200, 200, 800, 600)
        self.layout = QVBoxLayout()
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(self.layout)

        # create graphics scene
        self.scene = Scene()

        self.addNodes()

        # create graphic view
        self.view = QDMGraphicsView(self.scene.grScene, self)
        self.view.setScene(self.scene.grScene)
        self.layout.addWidget(self.view)

        # self.addSome()

    def addSome(self):
        greenBrush = QBrush(Qt.green)
        outlinePen = QPen(Qt.black)
        rect = self.scene.grScene.addRect(-100, -100, 80, 100, outlinePen, greenBrush)
        rect.setFlag(QGraphicsItem.ItemIsMovable)

    def addNodes(self):
        node1 = Node(self.scene, "Node 1", inputs=[0, 0], outputs=[1, 1])
        node1.setPos(-300, -200)
        node2 = Node(self.scene, "Node 2", inputs=[3, 3, 3], outputs=[1, 3])
        node2.setPos(0, -200)
        node3 = Node(self.scene, "Node 3", inputs=[2, 2, 2], outputs=[1, 2])
        node3.setPos(400, -200)
