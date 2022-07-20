from PyQt5.QtWidgets import *
from PyQt5.QtGui import QIcon

from .scene import Scene
from .graphics_view import QDMGraphicsView


class RLComposerWindow(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.scene_list = []
        self.view_list = []
        self.view = None
        self.scene = None
        self.initUI()

    def initUI(self):
        self.layout = QVBoxLayout()
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(self.layout)
        self.scene_tab = QTabWidget()
        self.layout.addWidget(self.scene_tab)
        self.scene_tab.currentChanged.connect(self.onTabChange)

        self.tabButton = QToolButton(self)
        self.tabButton.setIcon(QIcon('assets/plus.svg'))
        self.tabButton.clicked.connect(self.add_page)

        self.scene_tab.setCornerWidget(self.tabButton)
        self.add_page()

    def add_page(self):
        # create graphics scene
        self.scene_list.append(Scene())
        # create graphic view
        self.view_list.append(QDMGraphicsView(self.scene_list[-1].grScene, self))
        self.view_list[-1].setScene(self.scene_list[-1].grScene)
        self.scene_tab.addTab(self.view_list[-1], f"Scene {self.scene_tab.count()+1}")

    def onTabChange(self, i):
        self.scene = self.scene_list[i]
        self.view = self.view_list[i]

    def get_scene(self):
        return self.scene
