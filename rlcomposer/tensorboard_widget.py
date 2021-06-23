from PyQt5.QtCore import *
from PyQt5.QtWebEngineWidgets import QWebEngineView

class Tensorboard(QWebEngineView):
    def __init__(self):
        super(Tensorboard, self).__init__()
        self._update()

    def _update(self):
        self.load(QUrl("https://www.mfitzp.com/qna/qwebengineview-open-links-new-window/"))
        self.setZoomFactor(0.6)