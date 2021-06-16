from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

import random
from node import Node

class StandardItem(QStandardItem):
    def __init__(self, txt='', font_size=12, set_bold=False, color=QColor(0, 0, 0)):
        super().__init__()

        fnt = QFont('Open Sans', font_size)
        fnt.setBold(set_bold)

        self.setEditable(False)
        self.setForeground(color)
        self.setFont(fnt)
        self.setText(txt)


class FunctionTree(QWidget):
    def __init__(self, scene):
        super().__init__()
        self.setWindowTitle('Node Function')
        self.layout = QGridLayout()


        self.mainScene = scene

        treeView = QTreeView()

        treeView.setHeaderHidden(True)

        treeModel = QStandardItemModel()
        rootNode = treeModel.invisibleRootItem()


        models = StandardItem('Models', 16, set_bold=True)

        model_1 = StandardItem('Model 1', 14)
        model_2 = StandardItem('Model 2', 14)
        models.appendRow(model_1)
        models.appendRow(model_2)


        rewards = StandardItem('Reward Functions', 16, set_bold=True)

        reward_1 = StandardItem('Reward 1', 14)
        reward_2 = StandardItem('Reward 2', 14)
        rewards.appendRows([reward_1, reward_2])


        rootNode.appendRow(models)
        rootNode.appendRow(rewards)

        treeView.setModel(treeModel)
        treeView.expandAll()
        treeView.doubleClicked.connect(self.getValue)
        self.layout.addWidget(treeView, 0,0,1,2)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.addWidgets()
        self.setLayout(self.layout)

    def addWidgets(self):
        self.widget = QWidget(self)

        title = QLabel('Title')
        inpsocket = QLabel('Inputs')
        outsocket = QLabel('Outputs')

        self.inpsocketEdit = QLineEdit()
        self.outsocketEdit = QLineEdit()
        self.titleEdit = QLineEdit()

        self.push = QPushButton("Create Node", self)
        self.push.clicked.connect(self.onButtonClick)

        self.layout.addWidget(title, 1, 0)
        self.layout.addWidget(self.titleEdit, 1, 1)

        self.layout.addWidget(inpsocket, 2, 0)
        self.layout.addWidget(self.inpsocketEdit, 2, 1)

        self.layout.addWidget(outsocket, 3, 0)
        self.layout.addWidget(self.outsocketEdit, 3, 1)

        self.layout.addWidget(self.push, 4, 0)

    def getValue(self, val):
        print(val.data())
        print(val.row())
        print(val.column())

    @pyqtSlot()
    def onButtonClick(self):
        inpNum = int(self.inpsocketEdit.text())
        outNum = int(self.outsocketEdit.text())
        title = self.titleEdit.text()
        node1 = Node(self.mainScene, title, inputs=[0 for x in range(inpNum)], outputs=[0 for x in range(outNum)])
        node1.setPos(random.randint(-300, 300), random.randint(-300, 300))
        self.mainScene.history.storeHistory("Created " + title + " by dock widget", setModified=True)
