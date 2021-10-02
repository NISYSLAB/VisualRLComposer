from PyQt5.QtWidgets import *

class TestPlots(QWidget):
    def __init__(self, reward, action, state, parent=None):
        super().__init__(parent)
        self.raw_plot_widget = reward
        self.state_plot_widget = state
        self.action_plot_widget = action
        self.initUI()


    def initUI(self):
        layout = QGridLayout(self)
        layout.setRowStretch(0, 1)

        layout.setColumnStretch(0, 10)
        layout.setColumnStretch(1, 10)
        layout.setColumnStretch(2, 10)


        self.setLayout(layout)

        layout.addWidget(self.raw_plot_widget)
        layout.addWidget(self.action_plot_widget)
        layout.addWidget(self.state_plot_widget)
        layout.addWidget(self.raw_plot_widget, 0, 0)
        layout.addWidget(self.state_plot_widget, 0, 1)
        layout.addWidget(self.action_plot_widget, 0, 2)
