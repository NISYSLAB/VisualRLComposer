from collections import OrderedDict

from PyQt5.QtWidgets import *

from serializer import Serialize


class QDMNodeContentWidget(QWidget, Serialize):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.initUI()

    def initUI(self):
        self.layout = QVBoxLayout()
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(self.layout)

        self.wdg_label = QLabel("Some Title")
        self.layout.addWidget(self.wdg_label)
        self.content = QTextEdit("asadasad")
        self.layout.addWidget(QDMTextEdit())
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
