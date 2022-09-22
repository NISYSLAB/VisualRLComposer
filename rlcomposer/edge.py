from collections import OrderedDict
from itertools import cycle

from .serializer import Serialize
from .graphics.graphics_edge import *


class Edge(Serialize):
    """
    Class for representing an edge

    Attributes
    ----------
    start_socket: Socket class
        the socket object that the edge starts from
    end_socket: Socket class
        the socket object that the edge ends
    grEdge: QDMGraphicsEdgeDirect class
        the object that contains the graphical and visual features of the edge
    scene: Scene class
        the scene where edges are put into

    Methods
    -------
    updatePos()
        Update the x,y coordinates of the edges as they moved
    removeFromSockets()
        If edge is deleted, this function removes the edge information from the
        sockets that are connected each other via the current edge object
    remove()
        Check whether the socket has any edge or not, equivalent to checking
        whether it is connected or not
    serialize()
        Convert the object and its attributes to an ordered dictionary for serialization
    deserialize(data, hashmap)
        Initialize the object from a serialized data
    """

    def __init__(self, scene, start_socket=None, end_socket=None):
        super().__init__()
        self.scene = scene

        self.start_socket = start_socket
        self.end_socket = end_socket

        self.options = []
        self.options_cycle = None
        self.value = None

        self.grEdge = QDMGraphicsEdgeShaped(self)

        if self.start_socket is not None:
            self.updatePos()

        self.scene.grScene.addItem(self.grEdge)
        self.scene.addEdge(self)

    def __str__(self):
        return "<Edge %s..%s>" % (hex(id(self))[2:5], hex(id(self))[-3:])

    def iter_value(self):
        if self.options_cycle is None:
            self.options_cycle = cycle(self.options)
        self.value = next(self.options_cycle)
        self.grEdge.setEdgeValue(self.value)
        self.grEdge.update()
        print(self.value)

    @property
    def start_socket(self):
        return self._start_socket

    @start_socket.setter
    def start_socket(self, value):
        self._start_socket = value
        if self.start_socket is not None:
            self.start_socket.edge = self

    @property
    def end_socket(self):
        return self._end_socket

    @end_socket.setter
    def end_socket(self, value):
        self._end_socket = value
        if self.end_socket is not None:
            self.end_socket.edge = self

    def updatePos(self):
        source_pos = self.start_socket.getSocketPos()
        source_pos[0] += self.start_socket.node.grNode.pos().x()
        source_pos[1] += self.start_socket.node.grNode.pos().y()
        self.grEdge.setSource(*source_pos)
        if self.end_socket is not None:
            end_pos = self.end_socket.getSocketPos()
            end_pos[0] += self.end_socket.node.grNode.pos().x()
            end_pos[1] += self.end_socket.node.grNode.pos().y()
            self.grEdge.setEnd(*end_pos)
        else:
            self.grEdge.setEnd(*source_pos)
        self.grEdge.update()

    def removeFromSockets(self):
        if self.start_socket is not None:
            try:
                self.start_socket.node.outputNodes[self.start_socket.index] = None
            except:
                pass
            self.start_socket.edge = None
        if self.end_socket is not None:
            self.end_socket.edge = None
            try:
                self.end_socket.node.inputNodes[self.end_socket.index] = None
            except:
                pass

        self.end_socket = None
        self.start_socket = None

    def remove(self):
        self.removeFromSockets()
        self.scene.grScene.removeItem(self.grEdge)
        self.grEdge = None
        try:
            self.scene.removeEdge(self)
        except ValueError:
            pass

    def serialize(self):
        return OrderedDict([
            ("id", self.id),
            ("start", self.start_socket.id),
            ("end", self.end_socket.id)
        ])

    def deserialize(self, data, hashmap={}):
        self.id = data["id"]
        self.start_socket = hashmap[data["start"]]
        self.end_socket = hashmap[data["end"]]
        if self.start_socket is not None:
            self.updatePos()
        return True
