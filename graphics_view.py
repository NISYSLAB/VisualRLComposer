from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

from graphics_edge import QDMGraphicsEdge
from graphics_node import QDMGraphicsNode
from graphics_socket import QDMGraphicsSocket
from edge import Edge



MODE_NOOP = 1
MODE_EDGE_DRAG = 2
DEBUG = False


class QDMGraphicsView(QGraphicsView):
    def __init__(self, grScene, parent=None):
        super().__init__(parent)
        self.grScene = grScene

        self.initUI()
        self.editingFlag = False
        self.setScene(self.grScene)

        self.mode = MODE_NOOP

        self.zoomInFactor = 1.25
        self.zoom = 10
        self.zoomStep = 1
        self.zoomClamp = False
        self.zoomRange = [0, 10]

    def initUI(self):
        self.setRenderHints(
            QPainter.Antialiasing | QPainter.HighQualityAntialiasing | QPainter.TextAntialiasing | QPainter.SmoothPixmapTransform)

        self.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)

        # self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        # self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setDragMode(QGraphicsView.RubberBandDrag)

    def mousePressEvent(self, event):
        if event.button() == Qt.MiddleButton:
            self.middleMouseButtonPress(event)
        elif event.button() == Qt.LeftButton:
            self.leftMouseButtonPress(event)
        elif event.button() == Qt.RightButton:
            self.rightMouseButtonPress(event)
        else:
            super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MiddleButton:
            self.middleMouseButtonRelease(event)
        elif event.button() == Qt.LeftButton:
            self.leftMouseButtonRelease(event)
        elif event.button() == Qt.RightButton:
            self.rightMouseButtonRelease(event)
        else:
            super().mouseReleaseEvent(event)

    def middleMouseButtonPress(self, event):
        releaseEvent = QMouseEvent(QEvent.MouseButtonRelease, event.localPos(), event.screenPos(),
                                   Qt.LeftButton, Qt.NoButton, event.modifiers())
        super().mouseReleaseEvent(releaseEvent)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        fakeEvent = QMouseEvent(event.type(), event.localPos(), event.screenPos(),
                                Qt.LeftButton, event.buttons() | Qt.LeftButton, event.modifiers())
        super().mousePressEvent(fakeEvent)

    def middleMouseButtonRelease(self, event):
        fakeEvent = QMouseEvent(event.type(), event.localPos(), event.screenPos(),
                                Qt.LeftButton, event.buttons() & Qt.LeftButton, event.modifiers())
        super().mouseReleaseEvent(fakeEvent)
        self.setDragMode(QGraphicsView.NoDrag)

    def leftMouseButtonPress(self, event):

        # self.last_lmb_click_scene_pos = self.mapToScene(event.pos())

        item = self.getItemClicked(event)
        print(item)
        if type(item) is QDMGraphicsSocket:
            if (self.mode == MODE_NOOP) and (item.socket.isInput != 1):
                self.mode = MODE_EDGE_DRAG
                self.edgeDragStart(item)
                return

        if self.mode == MODE_EDGE_DRAG:
            res = self.edgeDragEnd(item)
            if res: return

        super().mousePressEvent(event)

    def leftMouseButtonRelease(self, event):

        item = self.getItemClicked(event)

        if self.mode == MODE_EDGE_DRAG:

            # new_lmb_release_scene_pos = self.mapToScene(event.pos())

            res = self.edgeDragEnd(item)
            if res: return

        super().mouseReleaseEvent(event)

    def rightMouseButtonPress(self, event):

        super().mousePressEvent(event)

        item = self.getItemClicked(event)
        if DEBUG:
            if isinstance(item, QDMGraphicsEdge): print("Edge:", item.edge, "Start-end sockets:",
                                                        item.edge.start_socket, item.edge.end_socket)
            if type(item) is QDMGraphicsSocket: print("Socket: ", item.socket, "has edges", item.socket.edge)
            if type(item) is QDMGraphicsNode: print("Node: ", item.node)
            if item is None:
                print("Scene:")
                print(' Nodes:')
                for node in self.grScene.scene.nodes:
                    print("  ", node)
                print(" Edges:")
                for edge in self.grScene.scene.edges:
                    print("  ", edge)

    def rightMouseButtonRelease(self, event):
        return super().mouseReleaseEvent(event)

    def mouseMoveEvent(self, event):
        if self.mode == MODE_EDGE_DRAG:
            pos = self.mapToScene(event.pos())
            self.dragEdge.grEdge.setEnd(pos.x(), pos.y())
            self.dragEdge.grEdge.update()

        super().mouseMoveEvent(event)

    def edgeDragEnd(self, item):
        self.mode = MODE_NOOP
        if type(item) is QDMGraphicsSocket:
            if (item.socket != self.last_start_socket) and (item.socket.node != self.last_start_socket.node) and (
                    item.socket.isInput == 1 and self.last_start_socket.isInput != 1):
                if DEBUG: print('View::edgeDragEnd ~   previous edge:', self.previousEdge)
                if self.previousEdge is not None: self.previousEdge.remove()
                if DEBUG: print('View::edgeDragEnd ~   assign End Socket', item.socket)
                if item.socket.hasEdge(): item.socket.edge.remove()
                if DEBUG: print('View::edgeDragEnd ~  previous edge removed')

                self.dragEdge.start_socket = self.last_start_socket
                self.dragEdge.end_socket = item.socket
                # print(self.dragEdge.end_socket)
                self.dragEdge.end_socket.setConnectedEdge(self.dragEdge)
                self.dragEdge.start_socket.setConnectedEdge(self.dragEdge)

                self.dragEdge.start_socket.node.outputNodes[
                    self.dragEdge.start_socket.index] = self.dragEdge.end_socket.node.title
                self.dragEdge.end_socket.node.inputNodes[
                    self.dragEdge.end_socket.index] = self.dragEdge.start_socket.node.title
                if DEBUG: print('View::edgeDragEnd ~  reassigned start & end sockets to drag edge')
                self.dragEdge.updatePos()
                if DEBUG: print('View::edgeDragEnd ~  updatePos')
                self.grScene.scene.history.storeHistory("Created new edge by dragging between " +
                                                        self.dragEdge.start_socket.node.title + " and "
                                                        + self.dragEdge.end_socket.node.title)

                return True

        self.dragEdge.remove()
        self.dragEdge = None
        if self.previousEdge is not None:
            self.previousEdge.start_socket.edge = self.previousEdge
        return False

    def edgeDragStart(self, item):
        if DEBUG: print('View::edgeDragStart ~ Start dragging edge')
        if DEBUG: print('View::edgeDragStart ~   assign Start Socket to:', item.socket)

        self.previousEdge = item.socket.edge
        self.last_start_socket = item.socket
        self.dragEdge = Edge(self.grScene.scene, start_socket=item.socket, end_socket=None)
        if DEBUG: print('View::edgeDragStart ~   dragEdge:', self.dragEdge)

    def wheelEvent(self, event):
        modifiers = QApplication.keyboardModifiers()
        if (modifiers == Qt.ControlModifier):
            zoomOut = 1 / self.zoomInFactor

            if event.angleDelta().y() > 0:
                zoomFactor = self.zoomInFactor
                self.zoom = self.zoom + self.zoomStep
            else:
                zoomFactor = zoomOut
                self.zoom = self.zoom - self.zoomStep

            self.scale(zoomFactor, zoomFactor)
        else:
            super().wheelEvent(event)

    def keyPressEvent(self, event):
        # if event.key() == Qt.Key_Delete:
        #     self.deleteSelected()
        # # elif event.key() == Qt.Key_S and event.modifiers() & Qt.ControlModifier:
        # #     self.grScene.scene.saveToFile("example.json.txt")
        # # elif event.key() == Qt.Key_L and event.modifiers() & Qt.ControlModifier:
        # #     self.grScene.scene.loadFromFile("example.json.txt")
        # elif event.key() == Qt.Key_Z and event.modifiers() & Qt.ControlModifier:
        #     self.grScene.scene.history.undo()
        # elif event.key() == Qt.Key_Y and event.modifiers() & Qt.ControlModifier:
        #     self.grScene.scene.history.redo()
        # elif event.key() == Qt.Key_H and event.modifiers() & Qt.ControlModifier:
        #     ix = 0
        #     for item in self.grScene.scene.history.stack:
        #         print("#", ix, "--", item["desc"])
        #         ix += 1
        # else:
        super().keyPressEvent(event)

    def getItemClicked(self, event):
        pos = event.pos()
        obj = self.itemAt(pos)
        return obj

    def deleteSelected(self):
        for item in self.grScene.selectedItems():
            if isinstance(item, QDMGraphicsEdge):
                item.edge.remove()
            elif hasattr(item, "node"):
                item.node.remove()
        self.grScene.scene.history.storeHistory("Delete selected")