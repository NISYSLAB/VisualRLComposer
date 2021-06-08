from node_content import QDMNodeContentWidget
from socket import *
from graphics_node import QDMGraphicsNode
from serializer import Serialize
from socket import Socket

class Node(Serialize):
    def __init__(self, scene, title="Undefined Node", inputs=[], outputs=[]):
        super().__init__()
        self._title = title
        self.scene = scene

        self.content = QDMNodeContentWidget()
        self.grNode = QDMGraphicsNode(self)
        self.title = title

        self.scene.addNode(self)
        self.scene.grScene.addItem(self.grNode)

        self.socket_spacing = 25

        # initializing sockets for node
        self.inputs = []
        self.outputs = []

        self.inputNodes = [None] * len(inputs)
        self.outputNodes = [None] * len(outputs)

        counter = 0
        for item in inputs:
            socket = Socket(node=self, index=counter, pos=LEFT_TOP, is_input=1)
            self.inputs.append(socket)
            counter += 1

        counter = 0
        for item in outputs:
            socket = Socket(node=self, index=counter, pos=RIGHT_BOTTOM, is_input=0)
            counter += 1
            self.outputs.append(socket)

    def printName(self):
        print("a")

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
            ("content", self.content.serialize()
             )
        ])

    def deserialize(self, data, hashmap={}):
        self.id = data["id"]
        hashmap[data["id"]] = self

        self.setPos(data["x_pos"], data["y_pos"])
        self.title = data["title"]

        data["inputs"].sort(key=lambda socket: socket["index"] + socket["position"] * 100)
        data["outputs"].sort(key=lambda socket: socket["index"] + socket["position"] * 100)
        self.inputs, self.outputs = [], []
        for socket_data in data["inputs"]:
            new_socket = Socket(node=self, index=socket_data["index"], pos=socket_data["position"],
                                is_input=socket_data["is_input"])
            new_socket.deserialize(socket_data, hashmap)
            self.inputs.append(new_socket)

        for socket_data in data["outputs"]:
            new_socket = Socket(node=self, index=socket_data["index"], pos=socket_data["position"],
                                is_input=socket_data["is_input"])
            new_socket.deserialize(socket_data, hashmap)
            self.outputs.append(new_socket)

        self.inputNodes = data["input_nodes"]
        self.outputNodes = data["output_nodes"]

        return True
