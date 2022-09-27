from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QFont
import sys
import os

# Set NeuroWeaver Path
# sys.path.insert(0, '/home/mehul/Desktop/test/neuroweaver')
sys.path.insert(0, os.getcwd())
from rlcomposer.main_window import RLMainWindow

if __name__ == '__main__':
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    app = QApplication(sys.argv)
    # app.setStyle('QtCurve')
    # app.setFont(QFont("Helvetica", 9))
    wnd = RLMainWindow()
    sys.exit(app.exec_())
    print("Exited")
