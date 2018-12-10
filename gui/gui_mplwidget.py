from PyQt4.QtGui import QSizePolicy, QVBoxLayout, QWidget
from matplotlib.backends.backend_qt4agg import (FigureCanvasQTAgg as FigureCanvas)
from matplotlib.figure import Figure


class MplCanvas(FigureCanvas):
    def __init__(self):
        self.fig = Figure(tight_layout=True)
        self.ax = self.fig.gca()
        FigureCanvas.__init__(self, self.fig)
        FigureCanvas.setSizePolicy(self,
                                   QSizePolicy.Expanding,
                                   QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)


class MplWidget(QWidget):
    def __init__(self, parent=None):
        QWidget.__init__(self, parent)
        self.canvas = MplCanvas()
        self.vbl = QVBoxLayout()
        self.vbl.addWidget(self.canvas)
        self.setLayout(self.vbl)

    def clear(self):
        self.canvas.ax.cla()
        self.canvas.draw()
