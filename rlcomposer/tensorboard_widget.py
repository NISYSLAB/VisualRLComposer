from PyQt5.QtCore import *
from PyQt5.QtWebEngineWidgets import QWebEngineView

class Tensorboard(QWebEngineView):
    """
    Class for the Tensorboard Widget

    Attributes
    ----------
    timer: QTimer class
        the timer object that is used for delayed loading
    Methods
    -------
    delayed_load(delay_ms=2500)
        Starts the timer with a delay of "delay_ms" value
    _update()
        Loads the URL that tensorboard is running
    """
    def __init__(self):
        super(Tensorboard, self).__init__()
        # initialize the timer
        self.timer = QTimer()
        # it only works only once
        self.timer.setSingleShot(True)
        # timer is connected to _update() function
        self.timer.timeout.connect(self._update)

    def delayed_load(self, delay_ms=10000):
        self.load(QUrl('http://localhost:6006/'))
        self.timer.start(delay_ms)

    def _update(self):
        self.load(QUrl('http://localhost:6006/'))
        self.setZoomFactor(1)