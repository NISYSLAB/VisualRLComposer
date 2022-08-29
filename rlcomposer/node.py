from .node_content import QDMNodeContentWidget
from .sockett import *
from .graphics.graphics_node import QDMGraphicsNode
from .serializer import Serialize
from .sockett import SocketT
from .rl.component_wrapper import ComponentWrapper
from .rl.env_wrapper import EnvWrapper
from .rl.reward_wrapper import RewardWrapper
from .rl.model_wrapper import ModelWrapper

class Node(Serialize):
    """
    Class for representing a node

    Attributes
    ----------
    scene: Scene class
        the scene where nodes are put into
    title: str
        title of the node
    inputs: list
        a list that contains the input socket objects
    outputs: list
        a list that contains the output socket objects
    inputNodes: list
        contains the title of the connected nodes from input sockets
    outputNodes: list
        contains the title of the connected nodes from output sockets
    content: QDMNodeContentWidget class
        the content part of the node
    grNode: QDMGraphicsNode class
        the object that contains the graphical and visual features of the node
    socket_spacing: int
        the distance between each socket circle

    Methods
    -------
    setPos(x,y)
        Set the positions of nodes on the scene
    getSocketPos(index, pos)
        Return the positions of the sockets
    updateConnectedEdges()
        Update the edge positions for each socket on a node
    remove()
        Remove the node from the scene
    serialize()
        Convert the object and its attributes to an ordered dictionary for serialization
    deserialize(data, hashmap)
        Initialize the object from a serialized data
    """


    def __init__(self, scene, title="Undefined", inputs=[], outputs=[], nodeType=None, model_name=None):

        super().__init__()
        self._title = title
        self.scene = scene
        self.nodeType = nodeType
        self.param = None
        self.model_name = model_name

        if self.title == "Testing Components":
            print("Inside Node Class Component")
            self.wrapper = ComponentWrapper(self.nodeType)
            self.param = self.wrapper.param
        elif self.title == "Environment":
            print("Inside Node Class Environment")
            self.wrapper = EnvWrapper(self.nodeType)
            self.param = self.wrapper.param
        elif self.title == "Reward":
            print("Inside Node Class Reward")
            self.wrapper = RewardWrapper(self.nodeType)
            self.param = self.wrapper.param
        elif self.title == "Models":
            self.wrapper = ModelWrapper(self.nodeType)
            if self.model_name is not None: self.wrapper.loadModel(self.model_name)
            self.param = self.wrapper.param

        if title!="Undefined":
            self.content = QDMNodeContentWidget(node=self)
            self.grNode = QDMGraphicsNode(self)
            self.scene.addNode(self)
            self.scene.grScene.addItem(self.grNode)
            self.title = title




        self.socket_spacing = 25

        # initializing sockets for node
        self.inputs = []
        self.outputs = []

        self.inputNodes = [None] * len(inputs)
        self.outputNodes = [None] * len(outputs)

        counter = 0
        for item in inputs:
            socket = SocketT(node=self, index=counter, pos=LEFT_TOP, is_input=1)
            self.inputs.append(socket)
            counter += 1

        counter = 0
        for item in outputs:
            socket = SocketT(node=self, index=counter, pos=RIGHT_BOTTOM, is_input=0)
            counter += 1
            self.outputs.append(socket)

    def __str__(self):
        return "<Node %s..%s>" % (hex(id(self))[2:5], hex(id(self))[-3:])



    @property
    def title(self):
        return self._title

    @title.setter
    def title(self, value):
        self._title = value
        self.grNode.title = self._title

    @property
    def pos(self):
        return self.grNode.pos()

    def updateWrapper(self):
        print("updateWrapper 1")
        self.wrapper.setParameters(self.param)
        self.scene._parameter_updated = True
        print(self.wrapper.param)
        print("inside Update Wrapper",self.param)

    def setPos(self, x, y):
        self.grNode.setPos(x, y)

    def getSocketPos(self, index, pos):
        x = 0 if (pos in (LEFT_TOP, LEFT_BOTTOM)) else self.grNode.width

        if pos in (LEFT_BOTTOM, RIGHT_BOTTOM):
            # start from bottom
            y = self.grNode.height - self.grNode.edge_size - self.grNode._padding - index * self.socket_spacing
        else:
            # start from top
            y = self.grNode.title_height + self.grNode._padding + self.grNode.edge_size + index * self.socket_spacing

        return [x, y]

    def updateConnectedEdges(self):
        for socket in self.inputs + self.outputs:
            if socket.hasEdge():
                socket.edge.updatePos()

    def remove(self):
        for socket in (self.inputs + self.outputs):
            if socket.hasEdge():
                socket.edge.remove()
        self.scene.grScene.removeItem(self.grNode)
        self.grNode = None
        self.scene.removeNode(self)

    def serialize(self):
        inputs, outputs = [], []

        for socket in self.inputs:
            inputs.append(socket.serialize())
        for socket in self.outputs:
            outputs.append(socket.serialize())

        return OrderedDict([
            ("id", self.id),
            ("title", self.title),
            ("x_pos", self.grNode.scenePos().x()),
            ("y_pos", self.grNode.scenePos().y()),
            ("inputs", inputs),
            ("outputs", outputs),
            ("input_nodes", self.inputNodes),
            ("output_nodes", self.outputNodes),
            ("content", self.content.serialize()),
            ("param", self.param),
            ("nodeType", self.nodeType),
            ("model_name", self.model_name),
        ])

    def deserialize(self, data, hashmap={}):
        self.id = data["id"]
        hashmap[data["id"]] = self

        self.param = data["param"]

        self.model_name = data["model_name"]
        self.nodeType = data["nodeType"]
        self.content = QDMNodeContentWidget(node=self)
        self.grNode = QDMGraphicsNode(self)
        self.title = data["title"]

        self.setPos(data["x_pos"], data["y_pos"])

        if self.title == "Component":
            print("Inside Node Class Component")
            self.wrapper = ComponentWrapper(self.nodeType)
            self.param = self.wrapper.param
        elif self.title == "Environment":
            print("Inside Node Class Environment")
            self.wrapper = EnvWrapper(self.nodeType)
        elif self.title == "Reward":
            print("Inside Node Class Reward")
            self.wrapper = RewardWrapper(self.nodeType)
        elif self.title == "Models":
            self.wrapper = ModelWrapper(self.nodeType)
            if self.model_name is not None: self.wrapper.loadModel(self.model_name)

        if self.param is None:
            self.param = self.wrapper.param
            self.content.param_dict = self.param

        data["inputs"].sort(key=lambda socket: socket["index"] + socket["position"] * 100)
        data["outputs"].sort(key=lambda socket: socket["index"] + socket["position"] * 100)
        self.inputs, self.outputs = [], []
        for socket_data in data["inputs"]:
            new_socket = SocketT(node=self, index=socket_data["index"], pos=socket_data["position"],
                                 is_input=socket_data["is_input"])
            new_socket.deserialize(socket_data, hashmap)
            self.inputs.append(new_socket)

        for socket_data in data["outputs"]:
            new_socket = SocketT(node=self, index=socket_data["index"], pos=socket_data["position"],
                                 is_input=socket_data["is_input"])
            new_socket.deserialize(socket_data, hashmap)
            self.outputs.append(new_socket)

        self.inputNodes = data["input_nodes"]
        self.outputNodes = data["output_nodes"]

        self.scene.addNode(self)
        self.scene.grScene.addItem(self.grNode)

        return True
