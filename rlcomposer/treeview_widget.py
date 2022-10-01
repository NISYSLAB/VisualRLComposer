from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import sys
import os
sys.path.append(os.getcwd() + "/rl")

import random
import os


def get_classes(current_module):
    class_names = []
    to_be_dropped = ['Component', 'PPO', 'uidarray']
    for key in dir(current_module):
        if isinstance(getattr(current_module, key), type) and key not in to_be_dropped:
            class_names.append(key)
    return class_names


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
    def __init__(self, window_widget):
        super().__init__()
        self.setWindowTitle('Node Function')
        self.layout = QGridLayout()
        self.window_widget = window_widget
        self.treeView = QTreeView()
        self.treeView.setHeaderHidden(True)

        import ppo.ppo_components as ppo_components
        import ppo.sac_components as sac_components
        self.ppo_component_names = get_classes(ppo_components)
        self.sac_component_names = get_classes(ppo_components)
        self.initTreeModel()

    def initTreeModel(self):

        self.treeModel = QStandardItemModel()
        self.rootNode = self.treeModel.invisibleRootItem()

        self.ppo_components = StandardItem('PPO_Components', 12, set_bold=True)
        for component_name in self.ppo_component_names:
            self.ppo_components.appendRow(self.createItem(component_name))

        self.sac_components = StandardItem('SAC_Components', 12, set_bold=True)
        for component_name in self.sac_component_names:
            self.sac_components.appendRow(self.createItem(component_name))

        self.rootNode.appendRow(self.ppo_components)
        self.rootNode.appendRow(self.sac_components)
        self.treeView.setModel(self.treeModel)
        self.treeView.expandAll()
        self.treeView.doubleClicked.connect(self.getValue)
        self.layout.addWidget(self.treeView, 0, 0, 1, 10)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.addWidgets()
        self.setLayout(self.layout)

    def createItem(self, name):
        return StandardItem(name, 11)

    def addWidgets(self):
        self.widget = QWidget(self)

        inpsocket = QLabel('Inputs:')
        outsocket = QLabel('Outputs:')
        self.status = QLabel('Status:')

        self.inpsocketEdit = QSpinBox()
        self.inpsocketEdit.setMinimum(0)
        self.inpsocketEdit.setMaximum(4)
        self.inpsocketEdit.setProperty('value', 0)
        self.outsocketEdit = QSpinBox()
        self.outsocketEdit.setMinimum(0)
        self.outsocketEdit.setMaximum(4)
        self.outsocketEdit.setProperty('value', 0)

        self.push = QPushButton("Create Node", self)
        self.push.clicked.connect(self.onButtonClick)

        self.layout.addWidget(inpsocket, 1, 0)
        self.layout.addWidget(self.inpsocketEdit, 1, 2)

        self.layout.addWidget(outsocket, 2, 0)
        self.layout.addWidget(self.outsocketEdit, 2, 2)

        self.layout.addWidget(self.status, 1, 6, 1, 1)
        self.layout.addWidget(self.push, 2, 6, 1, 1)

    def getValue(self, val):
        print(val.data())
        print(val.row())
        print(val.column())

    @pyqtSlot()
    def onButtonClick(self):
        try:
            inpNum = int(self.inpsocketEdit.text())
            outNum = int(self.outsocketEdit.text())
            index = self.treeView.selectedIndexes()[0]
        except IndexError:
            print("Select a Module!")
            return
        except Exception as e:
            print("Fill the editable textboxes!")
            return
        nodeType = index.model().itemFromIndex(index).text()

        parentTitle = index.model().itemFromIndex(index.parent()).text()
        model_name = None
        if nodeType == "Load PreTrained Model":
            fname, filt = QFileDialog.getOpenFileName(self, "Load Pretrained Model")
            if fname == "":
                return
            if os.path.isfile(fname):
                model_name = fname

        self.mainScene = self.window_widget.get_scene()
        self.mainScene.generateNode(parentTitle, inpNum, outNum, nodeType=nodeType, model_name=model_name)
        self.mainScene.history.storeHistory("Created " + parentTitle + " by dock widget", setModified=True)
