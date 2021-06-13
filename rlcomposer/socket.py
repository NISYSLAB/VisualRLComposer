from collections import OrderedDict

from serializer import Serialize
from graphics_socket import QDMGraphicsSocket

LEFT_TOP = 1
LEFT_BOTTOM = 2
RIGHT_TOP = 3
RIGHT_BOTTOM = 4
DEBUG = False


class Socket(Serialize):
    """
    Class for representing a socket

    Attributes
    ----------
    edge: Edge class
        the edge object that is connected from or to the current socket
    grSocket: QDMGraphicsSocket class
        the object that contains the graphical and visual features of the socket
    index: int
        the order of the socket that sockets will be positioned according to their
        indexes, (e.g. 0 means the first socket)
    isInput: int
        shows if it is an input or a output socket (e.g. 1 for input sockets)
    node: Node class
        the node object where the socket is positioned on
    pos: int
        the integer that represents the position of the socket

    Methods
    -------
    getSocketPos()
        Get the x,y positions of the sockets
    setConnectedEdge(edge=None)
        Initialize the edge attribute if the socket is connected to any edge
    hasEdge()
        Check whether the socket has any edge or not, equivalent to checking
        whether it is connected or not
    serialize()
        Convert the object and its attributes to an ordered dictionary for serialization
    deserialize(data, hashmap)
        Initialize the object from a serialized data
    """


    def __init__(self, node, index=0, pos=LEFT_TOP, is_input=1):
        super().__init__()
        self.node = node
        self.index = index
        self.pos = pos
        self.isInput = is_input

        # print("Socket -- creating with", self.index, self.pos, "for node", self.node)

        self.grSocket = QDMGraphicsSocket(self)

        self.grSocket.setPos(*self.node.getSocketPos(index, pos))

        self.edge = None

    def __str__(self):
        return "<Socket %s..%s>" % (hex(id(self))[2:5], hex(id(self))[-3:])

    def getSocketPos(self):
        if DEBUG: print("  GSP: ", self.index, self.pos, "node:", self.node)
        res = self.node.getSocketPos(self.index, self.pos)
        if DEBUG: print("  res", res)
        return res

    def setConnectedEdge(self, edge=None):
        self.edge = edge

    def hasEdge(self):
        return self.edge is not None

    def serialize(self):
        return OrderedDict([
            ("id", self.id),
            ("index", self.index),
            ("position", self.pos),
            ("is_input", self.isInput)
        ])

    def deserialize(self, data, hashmap={}):
        self.id = data["id"]
        hashmap[data["id"]] = self
        return True
