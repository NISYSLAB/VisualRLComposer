from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
import sys
from dock_widget import QDMDockWidget
from window_widget import RLComposerWindow
from tensorboard_widget import Tensorboard
from plot_widget import MplCanvas
from custom_network_widget import NetConfigWidget
from treeview_widget import FunctionTree

import os
import random

class Stream(QObject):
    """Redirects console output to text widget."""
    newText = pyqtSignal(str)

    def write(self, text):
        self.newText.emit(str(text))


class RLMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.fname = None
        # sys.stdout = Stream(newText=self.onUpdateText)

        # Initialize a timer
        self.timer = QTimer(self)
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

        layout = QGridLayout()
        layout.setRowStretch(0, 6)
        layout.setRowStretch(1, 4)
        layout.setColumnStretch(0, 8)
        layout.setColumnStretch(1, 2)


        self.window_widget = RLComposerWindow(self)
        self.window_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.window_widget.scene.addIsModifiedListener(self.createTitle)
        layout.addWidget(self.window_widget,0,0)
        # layout.addWidget(QPushButton("grid button"),0,1)
        # self.setCentralWidget(self.window_widget)

        # self.dock = QDMDockWidget("dock part", self.window_widget.scene)
        # self.addDockWidget(Qt.RightDockWidgetArea, self.dock)


        self.tensorboard = Tensorboard()
        self.tensorboard.delayed_load()

        self.canvas = MplCanvas(self, self.window_widget.scene, width=5, height=4, dpi=100)

        self.netconf = NetConfigWidget(self, '')

        self.plot_tab = QTabWidget(self)
        self.plot_tab.addTab(self.tensorboard, 'Tensorboard')
        self.plot_tab.addTab(self.canvas, "Plots")
        self.plot_tab.addTab(self.netconf, "Custom Network")
        self.plot_tab.currentChanged.connect(self.onTabChange)
        layout.addWidget(self.plot_tab, 1, 0)


        self.tree = FunctionTree(self.window_widget.scene)
        layout.addWidget(self.tree, 0,1)

        # Create the text output widget.
        self.process = QTextEdit()
        self.process.ensureCursorVisible()
        self.process.setLineWrapColumnOrWidth(500)
        self.process.setLineWrapMode(QTextEdit.FixedPixelWidth)
        layout.addWidget(self.process, 1, 1)

        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)

        # set window properties
        self.setGeometry(200, 200, 800, 600)
        self.createTitle()
        self.show()

    # def onUpdateText(self, text):
    #     """Write console output to text widget."""
    #     cursor = self.process.textCursor()
    #     cursor.movePosition(QTextCursor.End)
    #     cursor.insertText(text)
    #     self.process.setTextCursor(cursor)
    #     self.process.ensureCursorVisible()

    def onTabChange(self,i): #changed!
        if i==1:
            self.canvas.setupTimer()
        else:
            self.canvas.removeTimer()
        QMessageBox.information(self,
                  "Tab Index Changed!",
                  "Current Tab Index: %d" % i ) #changed!

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