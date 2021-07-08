from PyQt5.QtWidgets import *
import sys
from main_window import RLMainWindow

if __name__ == '__main__':
    app = QApplication(sys.argv)
    wnd = RLMainWindow()
    sys.exit(app.exec_())
    print("Exited")