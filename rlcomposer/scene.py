import json
from collections import OrderedDict
import random

from edge import Edge
from node import Node
from serializer import Serialize
from graphics_scene import QDMGraphicsScene
from scene_history import SceneHistory

DEBUG = False

class Scene(Serialize):
    def __init__(self):
        super().__init__()
        self.nodes = []
        self.edges = []
        self.width, self.height = 3200, 3200

        self.initUI()
        self.history = SceneHistory(self)

        self._is_modified = False
        self._is_modified_listener = []

    @property
    def is_modified(self):
        return self._is_modified

    @is_modified.setter
    def is_modified(self, value):
        if not self._is_modified and value:
            self._is_modified = value
            for callback in self._is_modified_listener:
                callback()
        self._is_modified = value

    def addIsModifiedListener(self, callback):
        self._is_modified_listener.append(callback)

    def initUI(self):
        self.grScene = QDMGraphicsScene(self)
        self.grScene.setGrScene(self.width, self.height)

    def addNode(self, node):
        self.nodes.append(node)

    def addEdge(self, edge):
        self.edges.append(edge)

    def removeNode(self, node):
        self.nodes.remove(node)

    def removeEdge(self, edge):
        self.edges.remove(edge)

    def generateNode(self, title, inpNum, outNum, nodeType):
        node1 = Node(self, title, inputs=[0 for x in range(inpNum)], outputs=[0 for x in range(outNum)], nodeType=nodeType)
        node1.setPos(random.randint(-300, 300), random.randint(-300, 300))

    def saveToFile(self, file):
        with open(file, "w") as f:
            f.write(json.dumps(self.serialize(), indent=4))
            print("Succesfully saved to", file)

            self.is_modified = False

    def loadFromFile(self, file):
        with open(file, "r") as f:
            data = f.read()
            data = json.loads(data, encoding="utf-8")
            self.deserialize(data)

            self.is_modified = False

    def clear(self):
        while len(self.nodes) > 0:
            self.nodes[0].remove()

        self.is_modified = False

    def serialize(self):
        nodes, edges = [], []
        for node in self.nodes: nodes.append(node.serialize())
        for edge in self.edges: edges.append(edge.serialize())
        return OrderedDict([
            ("id", self.id),
            ("scene_width", self.width),
            ("scene_height", self.height),
            ("nodes", nodes),
            ("edges", edges)
        ])

    def deserialize(self, data, hashmap={}):
        if DEBUG: print("deserializating data", data)

        self.clear()
        hashmap = {}

        # Creating Nodes
        for node_data in data["nodes"]:
            Node(self).deserialize(node_data, hashmap)

        # Creating Edges
        for edge_data in data["edges"]:
            Edge(self).deserialize(edge_data, hashmap)

        return True
