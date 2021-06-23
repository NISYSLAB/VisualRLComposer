from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from interface import Interface



import os

DEBUG = True

class Worker(QRunnable):
    '''
    Worker thread

    Inherits from QRunnable to handler worker thread setup, signals and wrap-up.

    :param callback: The function callback to run on this worker thread. Supplied args and
                     kwargs will be passed through to the runner.
    :type callback: function
    :param args: Arguments to pass to the callback function
    :param kwargs: Keywords to pass to the callback function

    '''

    def __init__(self, fn):
        super(Worker, self).__init__()
        # Store constructor arguments (re-used for processing)
        self.fn = fn

    @pyqtSlot()
    def run(self):
        '''
        Initialise the runner function with passed args, kwargs.
        '''
        self.fn()
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
        # self.timer = QTimer(self)
        # self.threadpool = QThreadPool()
        # print("Multithreading with maximum %d threads" % self.threadpool.maxThreadCount())
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
        self.fname = self.widget.fname
        self.setCentralWidget(self.widget)

        # set window properties
        self.setGeometry(200, 200, 800, 600)

        self.show()

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