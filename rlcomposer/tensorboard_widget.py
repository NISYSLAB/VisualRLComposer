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

    url_Signal = pyqtSignal(str)

    def __init__(self):
        super(Tensorboard, self).__init__()
        self.url = 'http://localhost:6006/'
        # initialize the timer
        # self.timer = QTimer()
        # self.timer.setSingleShot(True)
        # self.timer.timeout.connect(self._update)

        self.timer_initial = QTimer()
        self.timer_initial.setSingleShot(True)
        self.timer_initial.timeout.connect(self._update)

    def initial_load(self, delay_ms=3000):
        self.timer_initial.start(delay_ms)

    #def delayed_load(self, delay_ms=10000):
    #    self.load(QUrl('http://localhost:6006/'))
    #    self.timer.start(delay_ms)

    def setURL(self, url):
        delay_ms = 3000
        if url != 'Null':
            self.url = url
        self.timer_initial.start(delay_ms)


    def _update(self):
        self.load(QUrl(self.url))
        self.setZoomFactor(1.1)
        self.timer_initial.stop()
