import json
from collections import OrderedDict
import random
import os
import xmltodict
import xml.etree.ElementTree as et
import json

from .edge import Edge
from .node import Node
from .serializer import Serialize
from .graphics.graphics_scene import QDMGraphicsScene
from .scene_history import SceneHistory

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

        self._parameter_updated = True

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

    def generateNode(self, title, inpNum, outNum, nodeType, model_name):
        node1 = Node(self, title, inputs=[0 for x in range(inpNum)], outputs=[0 for x in range(outNum)], nodeType=nodeType, model_name=model_name)
        node1.setPos(random.randint(-300, 300), random.randint(-300, 300))

    def saveToFile(self, file):
        with open(file, "w") as f:
            f.write(json.dumps(self.serialize(), indent=4))
            print("Succesfully saved to", file)

            self.is_modified = False

    def loadFromFile(self, file):
        with open(file, "r") as f:
            data = f.read()
            data = json.loads(data)
            self.deserialize(data)

            self.is_modified = False

    def loadFromGraphML(self, filename):
        xml_doc_path = os.path.abspath(filename)
        xml_tree = et.parse(xml_doc_path)
        root = xml_tree.getroot()
        # set encoding to and method proper
        to_string = et.tostring(root, encoding='UTF-8', method='xml')
        xml_to_dict = xmltodict.parse(to_string)
        print(json.dumps(xml_to_dict))
        data = {'nodes': [], 'edges': []}

        for edge in xml_to_dict['ns0:graphml']['ns0:graph']['ns0:edge']:
            data['edges'].append(OrderedDict([
                ('id', edge['@id']),
                ('start', edge['@source']),
                ('end', edge['@target'])
            ]))

        for node in xml_to_dict['ns0:graphml']['ns0:graph']['ns0:node']:
            if 'Reward' in node['ns0:data']['ns2:ShapeNode']['ns2:NodeLabel']:
                title = 'Reward'
            elif len(node['ns0:data']['ns2:ShapeNode']['ns2:NodeLabel']) <= 4:
                title = 'Models'
            else:
                title = 'Environment'
            inputs, outputs = [], []
            for edge in data['edges']:
                if edge['start'] == node['@id']:
                    outputs.append(OrderedDict([
                        ('id', node['@id']),
                        ('index', 0),
                        ('position', 4),
                        ('is_input', 0)
                    ]))
                if edge['end'] == node['@id']:
                    inputs.append(OrderedDict([
                        ('id', node['@id']),
                        ('index', 0),
                        ('position', 1),
                        ('is_input', 1)
                    ]))

            temp_dict = OrderedDict([
                ('id', node['@id']),
                ("title", title),
                ("x_pos", float(node['ns0:data']['ns2:ShapeNode']['ns2:Geometry']['@x'])),
                ("y_pos", float(node['ns0:data']['ns2:ShapeNode']['ns2:Geometry']['@y'])),
                ("inputs", inputs),
                ("outputs", outputs),
                ("input_nodes", []),
                ("output_nodes", []),
                ("content", {}),
                ("param", None),
                ("nodeType", node['ns0:data']['ns2:ShapeNode']['ns2:NodeLabel']),
                ("model_name", None),
            ])
            data['nodes'].append(temp_dict)

        self.deserialize(data)

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
