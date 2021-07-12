from PyQt5.QtCore import *
from PyQt5.QtWebEngineWidgets import QWebEngineView

class Tensorboard(QWebEngineView):
    def __init__(self):
        super(Tensorboard, self).__init__()
        self.timer = QTimer()
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self._update)

    def delayed_load(self, delay_ms=2500):
        self.timer.start(delay_ms)

    def _update(self):
        self.load(QUrl('http://localhost:6006/#scalars&_smoothingWeight=0.99'))
        self.setZoomFactor(0.6)