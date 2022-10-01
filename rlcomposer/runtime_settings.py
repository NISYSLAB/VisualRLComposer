from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import gym
import numpy as np

DEBUG = True


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


class RuntimeSettingsWindow(QMainWindow):
    button_clicked = pyqtSignal(dict)

    def __init__(self, content=None):
        super(RuntimeSettingsWindow, self).__init__()

        self.param = content.runtime_param
        self.layout = QGridLayout()
        self.setWindowTitle("Runtime Settings")
        self.setFixedWidth(1500)
        self.setFixedHeight(700)

        self.button_clicked.connect(content.removeWindow)
        self.addWidgets()

    def addWidgets(self):
        self.widget = QWidget(self)

        self.push = QPushButton("Update", self)
        self.push.clicked.connect(self.update)

        self.layout.addWidget(QLabel("Iterations"), 0, 1)
        self.layout.addWidget(QLineEdit(str(self.param["Iterations"])), 0, 2)

        self.layout.addWidget(QLabel("Use Futures"), 1, 1)
        self.layout.addWidget(QLineEdit(str(self.param["Use Futures"])), 1, 2)

        self.layout.addWidget(QLabel("Max MSG Size"), 2, 1)
        self.layout.addWidget(QLineEdit(str(self.param["Max MSG Size"])), 2, 2)

        self.layout.addWidget(QLabel("Samples"), 3, 1)
        self.add_sample = NewLayerButton(self)
        self.layout.addWidget(self.add_sample, 3, 2)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.container = QWidget(self)
        self.lay = QVBoxLayout(self.container)
        self.lay.addStretch()
        self.scroll.setWidget(self.container)

        for i in self.param['Samples']:
            print(i, type(i))
            # self.add_layer(i)

        self.layout.addWidget(self.scroll, 5, 2)
        self.layout.addWidget(self.push, 6 + 1, 1, 1, 2)
        self.widget.setLayout(self.layout)
        self.setCentralWidget(self.widget)
        self.setWindowFlag(Qt.WindowCloseButtonHint, False)
        self.setWindowFlag(Qt.WindowMinMaxButtonsHint, False)
        self.setLayout(self.layout)

    def add_layer(self, obj):
        index = len(self.param["Samples"])
        obj.update(index)
        self.lay.insertWidget(index, obj)
        self.param["Samples"].insert(index, obj)
        print([layer for layer in self.param["Samples"]], index)

    def delete_layer(self, layer):
        self.lay.removeWidget(layer)
        layer.deleteLater()
        self.param["Samples"].remove(layer)
        print([layer for layer in self.param["Samples"]])

    def update(self):
        res = self.param
        i = 0
        for obj in self.widget.children():
            if isinstance(obj, QLineEdit):
                key = list(res.keys())[i]
                if obj.text()[0].isdigit():
                    if type(res[key]) == float:
                        res[key] = float(obj.text())
                    else:
                        res[key] = int(obj.text())
                else:
                    res[key] = obj.text()
                i += 1
        self.button_clicked.emit(res)
        self.close()


class NewLayerButton(QPushButton):
    def __init__(self, parent):
        super(NewLayerButton, self).__init__()
        self.par = parent
        self.clicked.connect(self.new_layer)
        self.setText("Add Sample Layer")

    def new_layer(self):
        self.par.add_layer(QueueSample(self.par))


class ClickButton(QPushButton):
    def __init__(self, parent, name, triggers, status=None, text=False):
        super(ClickButton, self).__init__(parent=parent)
        self.par = parent
        self.name = name
        self.setStatusTip(status)

        for trigger in triggers:
            self.clicked.connect(trigger)

        if text:
            self.setStyleSheet("text-align:left;padding:4px;")
            self.setText('  ' + name)


class TextEdit(QWidget):
    def __init__(self, par, name, edit=""):
        super(TextEdit, self).__init__()
        self.par = par
        self.label = QLabel()
        self.label.setText(name)
        self.edit = QLineEdit(edit)
        font = self.label.font()
        font.setPointSize(font.pointSize() - 2)
        self.edit.setFont(font)
        self.label.setFont(font)

        self.lay = QHBoxLayout(self)
        self.lay.setContentsMargins(-1, 0, -1, 0)
        self.lay.setSpacing(0)
        self.lay.addWidget(self.label)
        self.lay.addWidget(self.edit)

    @property
    def val(self):
        return self.edit.text()


class Layer(QWidget):
    def __init__(self, parent, n=0):
        super(Layer, self).__init__(parent)
        self.par = parent
        self.type = self.__class__.__name__
        self.n = n
        self.label = QLabel(self.type + ':' + str(self.n))
        self.del_button = ClickButton(self, 'Delete', [self.delete], status='Delete Layer')
        self.del_button.setIcon(QIcon('rlcomposer/rl/assets/delete.svg'))
        self.del_button.setFixedSize(30, 30)
        self.lay = QHBoxLayout(self)
        self.lay.setContentsMargins(-1, 0, -1, 0)
        font = self.label.font()
        font.setPointSize(font.pointSize() - 2)
        self.label.setFont(font)
        self.lay.setSpacing(0)
        self.lay.addWidget(self.label)

    def delete(self):
        self.par.delete_layer(self)

    def __repr__(self):
        return self.type + str(self.n)

    def update(self, n):
        self.n = n
        self.label.setText(self.type)


class QueueSample(Layer):
    def __init__(self, parent, n=0):
        super(QueueSample, self).__init__(parent, n)
        self.name = TextEdit(self.par, 'Name')
        self.shape = TextEdit(self.par, 'Shape')
        self.queue_type = TextEdit(self.par, 'Type')
        self.sample_push = TextEdit(self.par, 'Push', '[]')

        self.lay.addWidget(self.name)
        self.lay.addWidget(self.shape)
        self.lay.addWidget(self.queue_type)
        self.lay.addWidget(self.sample_push)
        self.lay.addWidget(self.del_button)

    def get_data(self):
        temp_name = self.name.val
        temp_shape = (int(self.shape.val),)
        temp_queue_type = self.queue_type.val
        temp_push = []
        print(self.sample_push.val[1:-1].split(','))
        for i in self.sample_push.val[1:-1].split(','):
            if is_number(i):
                temp_push.append(float(i))
            elif i.isalpha():
                print(i)
                if i == 'True':
                    temp_push.append(True)
                elif i == 'False':
                    temp_push.append(False)
            elif i == 'oscillator-v0':
                env = gym.make('oscillator-v0')
                temp_push = np.copy(env.reset())

        return {'Name': temp_name, 'Shape': temp_shape, 'Type': temp_queue_type, 'Push': temp_push}
