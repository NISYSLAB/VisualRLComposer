from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

import random

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
        self.treeView = QTreeView()
        self.treeView.setHeaderHidden(True)

        self.env_names = ["Pendulum"]
        self.reward_names = ["Reward 1", "Reward 2"]
        self.initTreeModel()

    def initTreeModel(self):

        self.treeModel = QStandardItemModel()
        self.rootNode = self.treeModel.invisibleRootItem()



        self.envs = StandardItem('Environment', 12, set_bold=True)
        for env_name in self.env_names:
            self.envs.appendRow(self.createItem(env_name))

        self.rewards = StandardItem('Reward Function', 12, set_bold=True)
        for rew_name in self.reward_names:
            self.rewards.appendRow(self.createItem(rew_name))


        self.rootNode.appendRow(self.envs)
        self.rootNode.appendRow(self.rewards)
        self.treeView.setModel(self.treeModel)
        self.treeView.expandAll()
        self.treeView.doubleClicked.connect(self.getValue)
        self.layout.addWidget(self.treeView, 0,0,1,2)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.addWidgets()
        self.setLayout(self.layout)

    def createItem(self, name):
        return StandardItem(name, 11)

    def addWidgets(self):
        self.widget = QWidget(self)

        inpsocket = QLabel('Inputs')
        outsocket = QLabel('Outputs')

        self.inpsocketEdit = QLineEdit()
        self.outsocketEdit = QLineEdit()

        self.push = QPushButton("Create Node", self)
        self.push.clicked.connect(self.onButtonClick)


        self.layout.addWidget(inpsocket, 1, 0)
        self.layout.addWidget(self.inpsocketEdit, 1, 1)

        self.layout.addWidget(outsocket, 2, 0)
        self.layout.addWidget(self.outsocketEdit, 2, 1)

        self.layout.addWidget(self.push, 3, 0)

    def getValue(self, val):
        print(val.data())
        print(val.row())
        print(val.column())

    @pyqtSlot()
    def onButtonClick(self):
        try:
            inpNum = int(self.inpsocketEdit.text())
            outNum = int(self.outsocketEdit.text())
        except Exception as e:
            print("Fill the editable textboxes!")
            return
        index = self.treeView.selectedIndexes()[0]
        nodeType = index.model().itemFromIndex(index).text()
        parentTitle = index.model().itemFromIndex(index.parent()).text()
        self.mainScene.generateNode(parentTitle, inpNum, outNum, nodeType=nodeType)
        self.mainScene.history.storeHistory("Created " + parentTitle + " by dock widget", setModified=True)
