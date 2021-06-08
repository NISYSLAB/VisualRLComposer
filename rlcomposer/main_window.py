from PyQt5.QtCore import *
from PyQt5.QtWidgets import *

from dock_widget import QDMDockWidget
from window_widget import RLComposerWindow
import os

class RLMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.fname = None
        self.initUI()

    def createActionMenu(self, name, shortcut, tooltip, callback):
        act = QAction(name, self)
        act.setShortcut(shortcut)
        act.setToolTip(tooltip)
        act.triggered.connect(callback)
        return act

    def initUI(self):
        # create node editor widget
        menu = self.menuBar()
        fileMenu = menu.addMenu("File")
        fileMenu.addAction(self.createActionMenu("New", "CTRL+N", "Create new flow", self.clickedFileNew))
        fileMenu.addSeparator()
        fileMenu.addAction(self.createActionMenu("Open", "CTRL+O", "Open new flow", self.clickedFileOpen))
        fileMenu.addAction(self.createActionMenu("Save", "CTRL+S", "Save flow", self.clickedFileSave))
        fileMenu.addAction(self.createActionMenu("Save as", "CTRL+Shift+S", "Save flow as", self.clickedFileSaveAs))
        fileMenu.addSeparator()
        fileMenu.addAction(self.createActionMenu("Exit", "CTRL+Q", "Exit program", self.clickedFileExit))

        editMenu = menu.addMenu("Edit")
        editMenu.addAction(self.createActionMenu("Undo", "CTRL+Z", "Undo one step", self.clickedEditUndo))
        editMenu.addAction(self.createActionMenu("Redo", "CTRL+Y", "Redo one step", self.clickedEditRedo))
        editMenu.addSeparator()
        editMenu.addAction(self.createActionMenu("Delete", "Del", "Delete selected items", self.clickedEditDelete))
        editMenu.addSeparator()
        editMenu.addAction(self.createActionMenu("History", "CTRL+H", "Show history stack", self.clickedEditHistory))

        self.window_widget = RLComposerWindow(self)
        self.setCentralWidget(self.window_widget)
        self.setWindowTitle("Visual RL Composer")
        self.dock = QDMDockWidget("dock part", self.window_widget.scene)
        self.addDockWidget(Qt.RightDockWidgetArea, self.dock)

        # set window properties
        self.setGeometry(200, 200, 800, 600)
        self.show()

    def clickedFileNew(self):
        self.centralWidget().scene.clear()
        self.centralWidget().scene.history.stack = []

    def clickedFileOpen(self):
        fname, filt = QFileDialog.getOpenFileName(self, "Open file")
        if fname == "":
            return
        if os.path.isfile(fname):
            self.centralWidget().scene.loadFromFile(fname)
        self.centralWidget().scene.history.stack = []
    def clickedFileSave(self):
        if self.fname == None:
            self.clickedFileSaveAs()
        else:
            self.centralWidget().scene.saveToFile(self.fname)

    def clickedFileSaveAs(self):
        fname, filt = QFileDialog.getSaveFileName(self, "Save file As")
        if fname == "":
            return
        self.fname = fname
        self.clickedFileSave()

    def clickedFileExit(self):
        pass

    def clickedEditUndo(self):
        self.centralWidget().scene.history.undo()

    def clickedEditRedo(self):
        self.centralWidget().scene.history.redo()

    def clickedEditDelete(self):
        self.centralWidget().scene.grScene.views()[0].deleteSelected()

    def clickedEditHistory(self):
        ix = 0
        for item in self.centralWidget().scene.history.stack:
            print("#", ix, "--", item["desc"])
            ix += 1