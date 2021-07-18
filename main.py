from PyQt5.QtWidgets import *
import sys
from main_window import RLMainWindow

if __name__ == '__main__':
    import os

    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    app = QApplication(sys.argv)
    wnd = RLMainWindow()
    sys.exit(app.exec_())
    print("Exited")