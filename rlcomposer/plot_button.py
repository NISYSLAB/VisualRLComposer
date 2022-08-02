from PyQt5.QtWidgets import *
from PyQt5.QtCore import *


class PlotButton(QWidget):
    def __init__(self, testing_reward, testing_action, testing_state, training_reward, training_action, parent=None):
        super().__init__(parent)
        self.testing_reward_widget = testing_reward
        self.testing_action_widget = testing_action
        self.testing_state_widget = testing_state
        self.training_reward_widget = training_reward
        self.training_action_widget = training_action

        self.initUI()

    def initUI(self):
        layout = QGridLayout(self)
        #layout.setRowStretch(0, 1)
        #layout.setColumnStretch(0, 10)
        #layout.setColumnStretch(1, 10)
        #layout.setColumnStretch(2, 10)
        self.setLayout(layout)

        self.training_plot_label = QLabel('Training Plots')
        self.training_plot_label.setAlignment(Qt.AlignCenter)
        self.testing_plot_label = QLabel('Testing Plots')
        self.testing_plot_label.setAlignment(Qt.AlignCenter)

        self.training_reward_plot_button = QPushButton("Show Reward Plot", self)
        self.training_reward_plot_button.clicked.connect(lambda: self.training_reward_widget.show())

        self.training_action_plot_button = QPushButton("Show Action Plot", self)
        self.training_action_plot_button.clicked.connect(lambda: self.training_action_widget.show())

        self.testing_reward_plot_button = QPushButton("Show Reward Plot", self)
        self.testing_reward_plot_button.clicked.connect(lambda: self.testing_reward_widget.show())

        self.testing_state_plot_button = QPushButton("Show State Plot", self)
        self.testing_state_plot_button.clicked.connect(lambda: self.testing_state_widget.show())

        self.testing_action_plot_button = QPushButton("Show Action Plot", self)
        self.testing_action_plot_button.clicked.connect(lambda: self.testing_action_widget.show())

        layout.addWidget(self.training_plot_label, 0, 0, 1, 3)
        sub_layout = QHBoxLayout()
        sub_layout.addWidget(self.training_reward_plot_button)
        sub_layout.addWidget(self.training_action_plot_button)
        layout.addLayout(sub_layout, 1, 0, 1, 3)

        layout.addWidget(self.testing_plot_label, 2, 0, 1, 3)
        layout.addWidget(self.testing_reward_plot_button, 3, 0)
        layout.addWidget(self.testing_state_plot_button, 3, 1)
        layout.addWidget(self.testing_action_plot_button, 3, 2)

    def set_testing_buttons(self, state_bool):
        self.testing_reward_plot_button.setEnabled(state_bool)
        self.testing_state_plot_button.setEnabled(state_bool)
        self.testing_action_plot_button.setEnabled(state_bool)

    def set_training_buttons(self, state_bool):
        self.training_reward_plot_button.setEnabled(state_bool)
        self.training_action_plot_button.setEnabled(state_bool)
