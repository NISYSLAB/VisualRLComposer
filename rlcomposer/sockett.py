from collections import OrderedDict

from .serializer import Serialize
from .graphics.graphics_socket import QDMGraphicsSocket

LEFT_TOP = 1
LEFT_BOTTOM = 2
RIGHT_TOP = 3
RIGHT_BOTTOM = 4
DEBUG = False


class SocketT(Serialize):
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

        self.grSocket = QDMGraphicsSocket(self)

        self.grSocket.setPos(*self.node.getSocketPos(index, pos))

        self.edge = None

    def __str__(self):
        # changes the printing of the object
        return "<Socket %s..%s>" % (hex(id(self))[2:5], hex(id(self))[-3:])

    def getSocketPos(self):
        # returns the socket position as coordinates of [x,y]
        return self.node.getSocketPos(self.index, self.pos)

    def setConnectedEdge(self, edge=None):
        # assigns the edge object to the socket
        self.edge = edge

    def hasEdge(self):
        # boolean function to check whether the socket is connected or not
        return self.edge is not None

    def serialize(self):
        # serializing function that is used for saving the socket object and it
        # returns a dictionary
        return OrderedDict([
            ("id", self.id),
            ("index", self.index),
            ("position", self.pos),
            ("is_input", self.isInput)
        ])

    def deserialize(self, data, hashmap={}):
        # deserializing function that is used for loading the socket
        self.id = data["id"]
        hashmap[data["id"]] = self
        return True
