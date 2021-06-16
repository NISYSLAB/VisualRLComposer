import random

from PyQt5.QtCore import *
from PyQt5.QtWidgets import *

from node import Node


class QDMDockWidget(QDockWidget):
    """
    Class for the dock widget. This class contains both the visual features
    and events inside.

    Attributes
    ----------
    functionEdit: QComboBox class
        the combobox option where user can assign the function of the node
    inpsocketEdit: QLineEdit class
        the editable text where user can assign the number of input sockets to
        node
    layout: QGridLayout class
        the layout of the dock widget
    mainScene: Scene class
        the scene object where the design can be done visually
    outsocketEdit: QLineEdit class
        the editable text where user can assign the number of output sockets to
        node
    push:

    titleEdit:

    widget:

    window:Title

    Methods
    -------
    setPos(x,y)
        Set the positions of nodes on the scene
    getSocketPos(index, pos)
        Return the positions of the sockets
    updateConnectedEdges()
        Update the edge positions for each socket on a node
    remove()
        Remove the node from the scene
    serialize()
        Convert the object and its attributes to an ordered dictionary for serialization
    deserialize(data, hashmap)
        Initialize the object from a serialized data
    """

    def __init__(self, title, mainScene, parent=None):
        super().__init__(parent)
        self.windowTitle = title
        self.mainScene = mainScene
        # self.label_input_socket = self.addQlabel("input socket", None)
        self.initUI()

    def initUI(self):
        # self.listWidget = QDMDragListbox(self.scene)
        # self.setWidget(self.label_input_socket)
        self.setFloating(True)
        self.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self.setFeatures(QDockWidget.DockWidgetFloatable |
                         QDockWidget.DockWidgetMovable)
        self.setWindowTitle(self.windowTitle)
        self.layout = QGridLayout()
        self.addWidgets()

    def addWidgets(self):
        self.widget = QWidget(self)

        function = QLabel('Function')
        title = QLabel('Title')
        inpsocket = QLabel('Inputs')
        outsocket = QLabel('Outputs')

        self.functionEdit = QComboBox(self)
        self.inpsocketEdit = QLineEdit()
        self.outsocketEdit = QLineEdit()
        self.titleEdit = QLineEdit()

        self.push = QPushButton("Create Node", self)
        self.push.clicked.connect(self.onButtonClick)

        self.layout.addWidget(function, 1, 0)
        self.layout.addWidget(self.functionEdit, 1, 1)

        self.layout.addWidget(title, 2, 0)
        self.layout.addWidget(self.titleEdit, 2, 1)

        self.layout.addWidget(inpsocket, 3, 0)
        self.layout.addWidget(self.inpsocketEdit, 3, 1)

        self.layout.addWidget(outsocket, 4, 0)
        self.layout.addWidget(self.outsocketEdit, 4, 1)

        self.layout.addWidget(self.push, 5, 0)
        self.layout.setRowStretch(10, 1)
        self.widget.setLayout(self.layout)
        self.setWidget(self.widget)

    @pyqtSlot()
    def onButtonClick(self):
        inpNum = int(self.inpsocketEdit.text())
        outNum = int(self.outsocketEdit.text())
        title = self.titleEdit.text()
        node1 = Node(self.mainScene, title, inputs=[0 for x in range(inpNum)], outputs=[0 for x in range(outNum)])
        node1.setPos(random.randint(-300, 300), random.randint(-300, 300))
        self.mainScene.history.storeHistory("Created " + title + " by dock widget", setModified=True)

    def mousePressEvent(self, event):
        if event.button() == Qt.MiddleButton:
            self.middleMouseButtonPress(event)
        elif event.button() == Qt.LeftButton:
            self.leftMouseButtonPress(event)
        elif event.button() == Qt.RightButton:
            self.rightMouseButtonPress(event)
        else:
            super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MiddleButton:
            self.middleMouseButtonRelease(event)
        elif event.button() == Qt.LeftButton:
            self.leftMouseButtonRelease(event)
        elif event.button() == Qt.RightButton:
            self.rightMouseButtonRelease(event)
        else:
            super().mouseReleaseEvent(event)

    def rightMouseButtonPress(self, event):

        pass

    def rightMouseButtonRelease(self, event):
        pass

    def leftMouseButtonPress(self, event):
        super().mousePressEvent(event)
        print("clicked")
        pass

    def leftMouseButtonRelease(self, event):
        pass

    def middleMouseButtonPress(self, event):
        pass

    def middleMouseButtonRelease(self, event):
        pass

    def mouseMoveEvent(self, e):
        x = e.x()
        y = e.y()

        text = f'x: {x},  y: {y}'
        print(text)

    def getItemClicked(self, event):
        pos = event.pos()
        obj = self.itemAt(pos)
        return obj
