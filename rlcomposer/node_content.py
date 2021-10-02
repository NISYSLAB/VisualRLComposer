from collections import OrderedDict

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

from .serializer import Serialize

DEBUG = True

class QDMNodeContentWidget(QWidget, Serialize):
    """
    Class for representing a node content

    Attributes
    ----------
    content: QDMTextEdit class
        the editable text part of the node's content
    layout: QVBoxLayout class
        the layout of the content orientation of the node
    push: QPushButton class
        the push button object on the node
    wdg_label: QLabel class
        the subtitle positioned below the main title

    Methods
    -------
    serialize()
        Convert the object and its attributes to an ordered dictionary for serialization
    deserialize(data, hashmap)
        Initialize the object from a serialized data
    """


    def __init__(self, node=None):
        super().__init__()
        self.parent = node
        self.subtitle = node.nodeType
        self.param_dict = node.param
        self.param_window = None
        self.initUI()

    def initUI(self):
        self.layout = QVBoxLayout()
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(self.layout)

        self.wdg_label = QLabel(self.subtitle)
        self.layout.addWidget(self.wdg_label)
        print("initUI node content 1")
        self.push = QPushButton("Parameters", self)
        self.push.clicked.connect(self.openWindow)
        print("initUI node content 2")
        self.layout.addWidget(self.push)


    def openWindow(self):
        if DEBUG: print("openWindow node content 1")
        self.param_window = ParameterWindow(self)
        if DEBUG: print("openWindow node content 2")
        self.param_window.show()
        if DEBUG: print("openWindow node content 3")

    def removeWindow(self, r_dict):
        self.param_window = None
        self.param_dict = r_dict
        self.parent.param = r_dict
        self.parent.updateWrapper()

    def keyPressEvent(self, event):
        print(self.parent.__dict__)
        super().keyPressEvent(event)

    def serialize(self):
        return OrderedDict([
        ])

    def deserialize(self, data, hashmap={}):
        return False


class ParameterWindow(QMainWindow):
    button_clicked = pyqtSignal(dict)
    def __init__(self,  content=None):
        super(ParameterWindow, self).__init__()
        self.param = content.param_dict
        self.layout = QGridLayout()
        self.layout.setColumnStretch(0, 2)
        self.layout.setColumnStretch(1, 4)
        self.layout.setColumnStretch(2, 4)
        self.layout.setColumnStretch(3, 2)
        self.setWindowTitle("Update Parameters")
        self.button_clicked.connect(content.removeWindow)
        print(self.param)
        self.addWidgets()
        print("After addwidgets",self.param)


    def addWidgets(self):
        self.widget = QWidget(self)

        self.push = QPushButton("Update", self)
        self.push.clicked.connect(self.update)

        count = 0
        for key, value in self.param.items():
            count += 1
            self.layout.addWidget(QLabel(key), count, 1)
            self.layout.addWidget(QLineEdit(str(value)), count, 2)

        self.layout.addWidget(self.push, count + 1, 1,1,2)
        self.widget.setLayout(self.layout)
        self.setCentralWidget(self.widget)
        self.setWindowFlag(Qt.WindowCloseButtonHint, False)
        self.setWindowFlag(Qt.WindowMinMaxButtonsHint, False)
        self.setLayout(self.layout)

    def update(self):
        res = self.param
        i = 0
        for obj in self.widget.children():
            if isinstance(obj, QLineEdit):
                key = list(res.keys())[i]
                if obj.text()[0].isdigit():
                    if type(res[key]) == float:
                        res[key] = float(obj.text())
                    else:
                        res[key] = int(obj.text())
                else:
                    res[key] = obj.text()
                i += 1

        self.button_clicked.emit(res)
        self.close()


