import json
from collections import OrderedDict

from edge import Edge
from node import Node
from serializer import Serialize
from graphics.graphics_scene import QDMGraphicsScene
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

    def saveToFile(self, file):
        with open(file, "w") as f:
            f.write(json.dumps(self.serialize(), indent=4))
        print("Succesfully saved to", file)

    def loadFromFile(self, file):
        with open(file, "r") as f:
            data = f.read()
            data = json.loads(data, encoding="utf-8")
            self.deserialize(data)

    def clear(self):
        while len(self.nodes) > 0:
            self.nodes[0].remove()

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
