from PyQt5.QtWidgets import *

from .scene import Scene
from .graphics_view import QDMGraphicsView


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

        # create graphic view
        self.view = QDMGraphicsView(self.scene.grScene, self)
        self.view.setScene(self.scene.grScene)
        self.layout.addWidget(self.view)
