from collections import OrderedDict

from PyQt5.QtWidgets import *

from serializer import Serialize

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


    def __init__(self, parent=None):
        super().__init__(parent)
        self.initUI()

    def initUI(self):
        self.layout = QVBoxLayout()
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(self.layout)

        self.wdg_label = QLabel("Some Title")
        self.layout.addWidget(self.wdg_label)
        # self.content = QTextEdit("asadasad")
        self.content = QDMTextEdit()
        self.layout.addWidget(self.content)
        self.push = QPushButton("Apply", self)
        self.layout.addWidget(self.push)

    def serialize(self):
        return OrderedDict([

        ])

    def deserialize(self, data, hashmap={}):
        return False


class QDMTextEdit(QTextEdit):
    def keyPressEvent(self, event):
        print("içeri bastım")
        super().keyPressEvent(event)
