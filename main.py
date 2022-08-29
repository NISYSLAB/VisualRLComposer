from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QFont
import sys
import os
from rlcomposer.main_window import RLMainWindow

if __name__ == '__main__':
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    # Set NeuroWeaver Path
    sys.path.insert(0, 'C:\\Users\\mehul\\Desktop\\Newfolder\\neuroweaver')
    app = QApplication(sys.argv)
    # app.setStyle('QtCurve')
    app.setFont(QFont("Helvetica", 9))
    wnd = RLMainWindow()
    sys.exit(app.exec_())
    print("Exited")
