from PyQt5.QtWidgets import *


class TestPlotButton(QWidget):
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

        self.reward_plot_button = QPushButton("Show Reward Plot", self)
        self.reward_plot_button.clicked.connect(lambda: self.open_window(self.raw_plot_widget))

        self.state_plot_button = QPushButton("Show State Plot", self)
        self.state_plot_button.clicked.connect(lambda: self.open_window(self.state_plot_widget))

        self.action_plot_button = QPushButton("Show Action Plot", self)
        self.action_plot_button.clicked.connect(lambda: self.open_window(self.action_plot_widget))

        layout.addWidget(self.reward_plot_button, 0, 0)
        layout.addWidget(self.state_plot_button, 0, 1)
        layout.addWidget(self.action_plot_button, 0, 2)

    def open_window(self, data):
        self.plot_window = data
        self.plot_window.show()

    def set_buttons_state(self, state_bool):
        self.reward_plot_button.setEnabled(state_bool)
        self.state_plot_button.setEnabled(state_bool)
        self.action_plot_button.setEnabled(state_bool)
