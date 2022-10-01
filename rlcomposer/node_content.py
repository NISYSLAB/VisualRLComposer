from collections import OrderedDict
import json
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

    def __init__(self, content=None):
        super(ParameterWindow, self).__init__()
        self.param = content.param_dict
        self.layout = QGridLayout()

        self.setWindowTitle("Update Parameters")
        self.button_clicked.connect(content.removeWindow)
        print(self.param)
        self.addWidgets()
        print("After addwidgets", self.param)

    def addWidgets(self):
        self.widget = QWidget(self)

        self.push = QPushButton("Update", self)
        self.push.clicked.connect(self.update)

        count = 0

        for outputs in self.param['Outputs']:
            count += 1
            self.layout.addWidget(QLabel('Output -->  '), count, 0)
            self.layout.addWidget(QLabel('Name'), count, 1)
            self.layout.addWidget(QLineEdit(outputs['Name']), count, 2)
            self.layout.addWidget(QLabel('Shape'), count, 3)
            self.layout.addWidget(QLineEdit(str(outputs['Shape'])), count, 4)
            self.layout.addWidget(QLabel('Is_Process_Parallel'), count, 5)
            self.layout.addWidget(QLineEdit(str(outputs['Is_Process_Parallel'])), count, 6)

        for inputs in self.param['Inputs']:
            count += 1
            self.layout.addWidget(QLabel('Input -->  '), count, 0)
            self.layout.addWidget(QLabel('Name'), count, 1)
            self.layout.addWidget(QLineEdit(inputs['Name']), count, 2)
            self.layout.addWidget(QLabel('Shape'), count, 3)
            self.layout.addWidget(QLineEdit(str(inputs['Shape'])), count, 4)
            self.layout.addWidget(QLabel('Is_Process_Parallel'), count, 5)
            self.layout.addWidget(QLineEdit(str(inputs['Is_Process_Parallel'])), count, 6)

        for states in self.param['States']:
            count += 1
            self.layout.addWidget(QLabel('State -->  '), count, 0)
            self.layout.addWidget(QLabel('Name'), count, 1)
            self.layout.addWidget(QLineEdit(states['Name']), count, 2)
            self.layout.addWidget(QLabel('Shape'), count, 3)
            self.layout.addWidget(QLineEdit(str(states['Shape'])), count, 4)
            self.layout.addWidget(QLabel('Is_Process_Parallel'), count, 5)
            self.layout.addWidget(QLineEdit(str(states['Is_Process_Parallel'])), count, 6)

        if self.param['Arguments'][0]:
            count += 1
            self.layout.addWidget(QLabel('Arguments -->  '), count, 0)
            self.layout.addWidget(QLineEdit(str(self.param['Arguments'][0])), count, 1, 1, 6)

        self.layout.addWidget(self.push, count + 1, 0, 1, 7)
        self.widget.setLayout(self.layout)
        self.setCentralWidget(self.widget)
        self.setWindowFlag(Qt.WindowCloseButtonHint, False)
        self.setWindowFlag(Qt.WindowMinMaxButtonsHint, False)
        self.setLayout(self.layout)

    def update(self):
        res = self.param
        keys = list(res.keys())
        inner_keys = ['Name', 'Shape', 'Is_Process_Parallel']
        relation_dict = {'State -->  ': 'States', 'Input -->  ': 'Inputs', 'Output -->  ': 'Outputs',
                         'Arguments -->  ': 'Arguments'}
        group_key_count = {k: -1 for k in keys}
        prev_key = ''

        for obj in self.widget.children():
            if isinstance(obj, QLabel):
                if relation_dict.get(obj.text(), 0) in keys:
                    type_key = relation_dict[obj.text()]
                    group_key_count[type_key] += 1
                    inner_keys = list(res[type_key][group_key_count[type_key]].keys())
                if obj.text() in inner_keys:
                    prev_key = obj.text()

            if isinstance(obj, QLineEdit):
                if obj.text()[0].isdigit():
                    if type(res[type_key][group_key_count[type_key]][prev_key]) == float:
                        res[type_key][group_key_count[type_key]][prev_key] = float(obj.text())
                    else:
                        res[type_key][group_key_count[type_key]][prev_key] = int(obj.text())
                elif type_key == 'Arguments':
                    res[type_key][group_key_count[type_key]] = json.loads(obj.text().replace("\'", "\""))
                else:
                    res[type_key][group_key_count[type_key]][prev_key] = obj.text()

        self.button_clicked.emit(res)
        self.close()


