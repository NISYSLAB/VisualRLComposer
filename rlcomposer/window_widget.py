from PyQt5.QtWidgets import *
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import pyqtSlot

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
        self.tabButton.setIcon(QIcon('rlcomposer/rl/assets/plus.svg'))
        self.tabButton.clicked.connect(self.add_page)
        self.scene_tab.tabBarDoubleClicked.connect(self.double_click)

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

    def double_click(self, tab_index):
        rect = self.scene_tab.tabBar().tabRect(tab_index)
        top_margin = 3
        left_margin = 6
        self.__edit = QLineEdit(self)
        self.__edit.show()
        self.__edit.move(rect.left() + left_margin, rect.top() + top_margin)
        self.__edit.resize(rect.width() - 2 * left_margin, rect.height() - 2 * top_margin)
        self.__edit.setText(self.scene_tab.tabText(tab_index))
        self.__edit.selectAll()
        self.__edit.setFocus()
        self.__edit.editingFinished.connect(self.finish_rename)

    @pyqtSlot()
    def finish_rename(self):
        self.scene_tab.setTabText(self.scene_tab.currentIndex(), self.__edit.text())
        self.__edit.deleteLater()

    def get_current_tab_name(self):
        return self.scene_tab.tabText(self.scene_tab.currentIndex())