from PyQt5 import QtWidgets, QtGui, QtCore, QtSvg
import os
from rlcomposer.draw_nn import Dense, Conv2D, Model, Flatten, FeatureMap3D
# from stadium.core.defaults import CustomCnnPolicy, CustomMlpPolicy
import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# def get_icon(name):
#     icon_path = os.path.join(config.ICONS, name + '.svg')
#     return icon_path

class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space, data, features_dim: int = 256):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        self.feature_dimension = features_dim
        self.observation_space = observation_space

        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.cnn = data
        with th.no_grad():
            n_flatten = self.cnn(
                th.as_tensor(self.observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, self.feature_dimension), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))


class CallerSignal(QtCore.QObject):
    signal = QtCore.pyqtSignal(str)


class NetConfigWidget(QtWidgets.QWidget):
    def __init__(self, parent, name, config=None):
        super(NetConfigWidget, self).__init__(parent=parent)
        self.signal = CallerSignal()
        self.signal.signal.connect(self.caller)
        self.par = parent
        self.layers = []
        self.config = config
        self.flat = False
        self.initialized = False
        self.combo = None
        self.container = QtWidgets.QWidget(self)
        self.lay = QtWidgets.QVBoxLayout(self.container)
        self.lay.addStretch()
        self.display = QtSvg.QSvgWidget(self)
        self.display.setFixedHeight(150)
        self.main_lay = QtWidgets.QVBoxLayout(self)
        self.scroll = QtWidgets.QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setWidget(self.container)
        self.main_lay.addWidget(self.display)
        self.main_lay.addWidget(self.scroll)

    def caller(self, nn_type):
        self.build(nn_type)

    def build(self, nn_type='Mlp'):

        self.combo = QtWidgets.QGridLayout(self)
        self.layers = []
        self.container = QtWidgets.QWidget(self)
        self.lay = QtWidgets.QVBoxLayout(self.container)
        self.lay.addStretch()
        self.scroll.setWidget(self.container)

        if 'Cnn' in nn_type:
            self.flat = False
            for i, n_filters in enumerate([3, 4]):
                layer = Conv(
                    parent=self,
                    filters=3,
                    kernel=8,
                    stride=2,
                    n=i)
                self.add_layer(layer, update=False)
        else:
            self.flat = True

        for i, nodes in enumerate([64]):
            layer = FC(self, nodes, n=i)
            self.add_layer(layer, update=False)

        self.conv_layer_button = NewConvLayer(self)
        self.activation = NewActivationLayer(self)
        self.fc_layer_button = NewFCLayer(self)

        self.combo.addWidget(self.conv_layer_button, 0, 0)
        self.combo.addWidget(self.activation, 0, 1)
        self.combo.addWidget(self.fc_layer_button, 0, 2)
        if self.flat:
            self.conv_layer_button.setEnabled(False)

        self.lay.addLayout(self.combo)
        self.lay.addStretch()
        self.update_image()
        self.initialized = True

    def add_layer(self, obj, update=True):

        index = len(self.layers)
        if type(obj) is Conv and update:
            index = [type(x) is FC for x in self.layers].index(True)
        obj.update(index)
        self.lay.insertWidget(index, obj)
        self.layers.insert(index, obj)
        print([layer for layer in self.layers], index)
        if update:
            self.update_image()

    def delete_layer(self, layer):
        self.lay.removeWidget(layer)
        layer.deleteLater()
        self.layers.remove(layer)
        self.update_image()
        print([layer for layer in self.layers])

    def update_image(self):
        input_shape = self.par.getSpaceNames(self.par.tree.current_env)[2]
        output_shape = self.par.getSpaceNames(self.par.tree.current_env)[3]

        if input_shape == None:
            input_shape = 1
        if output_shape == None:
            output_shape = 1
        if type(input_shape) is int:
            input_shape = (input_shape, 1, 1)

        imgpath = os.path.join('assets/net.svg')
        print(imgpath)
        self.model = Model(input_shape=input_shape)
        if self.flat:
            self.model.add(Flatten())
        for i, layer in enumerate(self.layers):
            self.model.add(layer.to_draw())
            if type(layer) is Conv:
                try:
                    if type(self.layers[i + 1]) is FC:
                        self.model.add(Flatten())
                except IndexError:
                    self.model.add(Flatten())

        self.model.add(Dense(output_shape))
        print(self.model.feature_maps)
        print(self.model.layers)
        self.model.save_fig(imgpath)
        self.display.load(imgpath)

    # def blank(self):
    #     imgpath = os.path.join(config.UTILS, 'blank.svg')
    #     self.display.load(imgpath)

    def create_conf(self):
        conf = {}
        cnn_list = []
        if not self.flat:
            cnn_data = []
            for layer in self.layers:
                if type(layer) is Conv:
                    cnn_data.append(dict({"filters": layer.filters.val,
                                          "kernals": layer.kernel.val,
                                          "strides": layer.stride.val}))

            for i, layer in enumerate(cnn_data):
                print(cnn_data[i])
                if i == 0:
                    cnn_list.append(nn.Conv2d(in_channels=3,
                                              out_channels=cnn_data[i]["filters"],
                                              kernel_size=cnn_data[i]["kernals"],
                                              stride=cnn_data[i]["strides"],
                                              padding=0))
                else:
                    cnn_list.append(nn.Conv2d(in_channels=cnn_data[i - 1]["filters"],
                                              out_channels=cnn_data[i]["filters"],
                                              kernel_size=cnn_data[i]["kernals"],
                                              stride=cnn_data[i]["strides"],
                                              padding=0))
                cnn_list.append(nn.ReLU())
            cnn_list.append(nn.Flatten())


            #conf = {**dict(zip(names, [filters, kernels, strides]))}

        fc_layers = []
        for layer in self.layers:
            if type(layer) is FC:
                fc_layers.append(layer.nodes.val)

        if self.flat:
            if self.activation.function == None:
                conf = dict(
                    net_arch=fc_layers
                )
            else:
                conf = dict(
                    net_arch=fc_layers,
                    activation_fn=self.activation.function
                )
        else:
            if self.activation.function == None:
                conf = dict(
                    features_extractor_class=CustomCNN,
                    features_extractor_kwargs=dict(features_dim=fc_layers[0], data=nn.Sequential(*cnn_list)),
                    net_arch=fc_layers
                )
            else:
                conf = dict(
                    features_extractor_class=CustomCNN,
                    features_extractor_kwargs=dict(features_dim=fc_layers[0]),
                    net_arch=fc_layers,
                    activation_fn=self.activation.function
                )

        return conf


class ClickButton(QtWidgets.QPushButton):
    def __init__(self, parent, name, triggers, status=None, text=False):
        super(ClickButton, self).__init__(parent=parent)
        self.par = parent
        self.name = name
        self.setStatusTip(status)

        for trigger in triggers:
            self.clicked.connect(trigger)

        # icon = QtGui.QIcon(parent=self)
        # path = os.path.join(config.ICONS, name + '.svg')
        # icon.addPixmap(QtGui.QPixmap(path), size=QtCore.QSize(30, 30))
        # self.setIcon(icon)
        if text:
            self.setStyleSheet("text-align:left;padding:4px;")
            self.setText('  ' + name)


class DialSpin(QtWidgets.QWidget):
    def __init__(self, par, name, max, val=0):
        super(DialSpin, self).__init__()
        self.par = par
        self.label = QtWidgets.QLabel()
        self.label.setText(name)
        self.spin = QtWidgets.QSpinBox()
        font = self.label.font()
        font.setPointSize(font.pointSize() - 2)
        self.spin.setFont(font)
        self.label.setFont(font)
        self.spin.setRange(0, max)
        self.spin.setValue(val)
        self.lay = QtWidgets.QHBoxLayout(self)
        self.lay.setContentsMargins(-1, 0, -1, 0)
        self.lay.setSpacing(0)
        self.lay.addWidget(self.label)
        self.lay.addWidget(self.spin)
        self.spin.valueChanged.connect(self.par.update_image)

    @property
    def val(self):
        return self.spin.value()


class Layer(QtWidgets.QWidget):
    def __init__(self, parent, n=0):
        super(Layer, self).__init__(parent)
        self.par = parent
        self.type = self.__class__.__name__
        self.n = n
        self.label = QtWidgets.QLabel(self.type + ':' + str(self.n))
        self.del_button = ClickButton(self, 'Delete', [self.delete], status='Delete Layer')
        self.del_button.setIcon(QtGui.QIcon('assets/delete.svg'))
        self.del_button.setFixedSize(30, 30)
        self.lay = QtWidgets.QHBoxLayout(self)
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
        self.label.setText(self.type + ':' + str(self.n))


class Conv(Layer):
    def __init__(self, parent, filters=64, kernel=3, stride=1, padding='valid', n=0):
        super(Conv, self).__init__(parent, n)
        self.filters = DialSpin(self.par, 'Filters:', 512, val=filters)
        self.kernel = DialSpin(self.par, 'Kernel:', 15, val=kernel)
        self.stride = DialSpin(self.par, 'Stride:', 4, val=stride)

        self.padding = padding
        self.lay.addWidget(self.filters)

        self.lay.addWidget(self.kernel)
        self.lay.addWidget(self.stride)
        self.lay.addWidget(self.del_button)

    def to_draw(self):
        k, s = self.kernel.val, self.stride.val
        lay = Conv2D(filters=self.filters.val, kernel_size=(k, k), strides=(s, s), padding=self.padding)
        return lay


class FC(Layer):
    def __init__(self, parent, nodes=64, n=0):
        super(FC, self).__init__(parent, n)
        self.nodes = DialSpin(self.par, 'Nodes:', 2048, val=nodes)
        self.lay.addWidget(self.nodes)
        self.lay.addWidget(self.del_button)

    def to_draw(self):
        lay = Dense(units=self.nodes.val)
        return lay


class NewFCLayer(QtWidgets.QPushButton):
    def __init__(self, parent):
        super(NewFCLayer, self).__init__()

        self.par = parent
        self.clicked.connect(self.new_FC_layer)
        self.setText("Fully Connected Layer")

    def new_FC_layer(self):
        self.par.add_layer(FC(self.par))




class NewConvLayer(QtWidgets.QPushButton):
    def __init__(self, parent):
        super(NewConvLayer, self).__init__()
        self.par = parent
        self.clicked.connect(self.new_Conv_layer)
        self.setText("Convolutional Layer")

    def new_Conv_layer(self):
        self.par.add_layer(Conv(self.par))


class NewActivationLayer(QtWidgets.QComboBox):
    def __init__(self, parent):
        super(NewActivationLayer, self).__init__()

        self.par = parent
        self.function = None
        self.mapping = dict({'Activation Function': None,
                             'ReLU': nn.ReLU,
                             'Tanh': nn.Tanh,
                             'Sigmoid': nn.Sigmoid,
                             'ELU': nn.ELU,
                             'GLU': nn.GLU,
                             'Softmin': nn.Softmin,
                             'Softmax': nn.Softmax})
        self.activated.connect(self.new_layer)
        self.add_options()

    def new_layer(self):
        i = self.currentIndex() - 1
        object = self.mapping[self.currentText()]
        self.function = object()

    def add_options(self):
        for option in self.mapping.keys():
            self.addItem(option)
        self.model().item(0).setEnabled(False)