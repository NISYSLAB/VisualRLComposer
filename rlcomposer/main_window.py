from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from .interface import Interface



import os

DEBUG = True




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
        fileMenu.addAction(self.createActionMenu("Exit", "CTRL+Q", "Exit program", self.closeEvent))

        editMenu = menu.addMenu("Edit")
        editMenu.addAction(self.createActionMenu("Undo", "CTRL+Z", "Undo one step", self.clickedEditUndo))
        editMenu.addAction(self.createActionMenu("Redo", "CTRL+Y", "Redo one step", self.clickedEditRedo))
        editMenu.addSeparator()
        editMenu.addAction(self.createActionMenu("Delete", "Del", "Delete selected items", self.clickedEditDelete))
        editMenu.addSeparator()
        editMenu.addAction(self.createActionMenu("History", "CTRL+H", "Show history stack", self.clickedEditHistory))

        self.widget = Interface(self)
        self.window_widget = self.widget.window_widget
        self.window_widget.scene.addIsModifiedListener(self.createTitle)
        self.createTitle()
        self.fname = self.widget.fname
        self.setCentralWidget(self.widget)

        # set window properties
        self.setWindowState(Qt.WindowMaximized)
        self.setWindowIcon(QIcon('assets/logo.png'))
        self.show()

    def createTitle(self):
        title = "Visual RL Composer - "
        if self.fname is None:
            title += "New"
        else:
            title += os.path.basename(self.fname)

        if self.window_widget.scene.is_modified:
            title += "*"

        self.setWindowTitle(title)

    def clickedFileNew(self):
        if self.fileSaved():
            self.window_widget.scene.clear()
            self.fname = None
            self.window_widget.scene.history.stack = []
            self.createTitle()


    def clickedFileOpen(self):
        if self.fileSaved():
            fname, filt = QFileDialog.getOpenFileName(self, "Open file")
            if fname == "":
                return
            if os.path.isfile(fname):
                self.window_widget.scene.loadFromFile(fname)
                self.fname = fname
                self.createTitle()

        self.window_widget.scene.history.stack = []
    def clickedFileSave(self):
        if self.fname == None:
            return self.clickedFileSaveAs()
        self.window_widget.scene.saveToFile(self.fname)
        return True

    def clickedFileSaveAs(self):
        fname, filt = QFileDialog.getSaveFileName(self, "Save file As")
        if fname == "":
            return False
        self.fname = fname
        self.clickedFileSave()
        return True

    def closeEvent(self, event):
        if self.fileSaved():
            event.accept()
        else:
            event.ignore()

    def fileSaved(self):
        if not self.isChanged():
            return True
        msgBox = QMessageBox()
        msgBox.setIcon(QMessageBox.Question)
        msgBox.setText("The file has been changed.\n Do you want to save your file?")
        msgBox.setWindowTitle("Are you sure?")
        msgBox.setStandardButtons(QMessageBox.Save | QMessageBox.Close | QMessageBox.Cancel)
        res = msgBox.exec()

        # res = QMessageBox.warning(self, "Are you sure?", "The file has been changed.\n Do you want to save your file?",
        #                           QMessageBox.Save | QMessageBox. | QMessageBox.Cancel)

        if res == QMessageBox.Save:
            return self.clickedFileSave()
        elif res == QMessageBox.Cancel:
            return False
        return True

    def isChanged(self):
        return self.window_widget.scene.is_modified

    def clickedEditUndo(self):
        self.window_widget.scene.history.undo()

    def clickedEditRedo(self):
        self.window_widget.scene.history.redo()

    def clickedEditDelete(self):
        self.window_widget.scene.grScene.views()[0].deleteSelected()

    def clickedEditHistory(self):
        ix = 0
        for item in self.window_widget.scene.history.stack:
            print("#", ix, "--", item["desc"])
            ix += 1