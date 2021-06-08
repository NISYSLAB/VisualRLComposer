from graphics_edge import QDMGraphicsEdge

DEBUG = False

class SceneHistory():
    def __init__(self, scene):
        self.scene = scene

        self.stack = []
        self.current_index = -1
        self.limit = 16

    def undo(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.restoreHistory()

    def redo(self):
        if self.current_index + 1 < len(self.stack):
            self.current_index += 1
            self.restoreHistory()

    def restoreHistory(self):
        self.restoreHistoryStamp(self.stack[self.current_index])

    def storeHistory(self, desc):

        if self.current_index + 1 < len(self.stack):
            self.stack = self.stack[0:self.current_index + 1]

        if self.current_index >= self.limit:
            self.stack = self.stack[1:]
            self.current_index -= 1

        stamp = self.createHistoryStamp(desc)

        self.stack.append(stamp)
        self.current_index += 1
        if DEBUG: print("  -- setting step to:", self.current_index)

    def restoreHistoryStamp(self, stamp):
        if DEBUG: print("restoreHistoryStamp : 0")
        self.scene.deserialize(stamp["snap"])
        if DEBUG: print("restoreHistoryStamp : 1")


        for edge_id in stamp["select"]["edges"]:
            for edge in self.scene.edges:
                if edge.id == edge_id:
                    edge.grEdge.setSelected(True)
                    break

        if DEBUG: print("restoreHistoryStamp : 2")

        for node_id in stamp["select"]["nodes"]:
            for node in self.scene.nodes:
                if node.id == node_id:
                    node.grNode.setSelected(True)
                    break

        if DEBUG: print("restoreHistoryStamp : 3")

    def createHistoryStamp(self, desc):
        sel_obj = {
            "nodes": [],
            "edges": [],
        }

        for item in self.scene.grScene.selectedItems():
            if hasattr(item, "node"):
                sel_obj["nodes"].append(item.node.id)
            elif isinstance(item, QDMGraphicsEdge):
                sel_obj["edges"].append(item.edge.id)

        stamp = {
            "desc": desc,
            "snap": self.scene.serialize(),
            "select": sel_obj,
        }
        return stamp
