import logging

from PIL.ImageQt import ImageQt
from PyQt4 import QtCore, QtGui, uic
from PyQt4.QtGui import QMainWindow, QPixmap, QWidget
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.figure import SubplotParams

from gui._ring_label import RingImageQLabel
from gui.gui_mplwidget import MplWidget

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('ring-gui')
sp = SubplotParams(left=0., bottom=0., right=1., top=1.)
mydpi = 72


class RingWindow(QMainWindow):
    image: RingImageQLabel

    def __init__(self):
        super(RingWindow, self).__init__()
        uic.loadUi('./gui_ring.ui', self)
        self.zSpin.valueChanged.connect(self.on_zvalue_change)
        self.openButton.pressed.connect(self.on_open_button)
        self.dnaSpin.valueChanged.connect(self.on_dnaval_change)
        self.actSpin.valueChanged.connect(self.on_actval_change)

        # fig = Figure((self.image.width() / mydpi, self.image.height() / mydpi), subplotpars=sp, dpi=mydpi)
        self.file = "/Users/Fabio/data/lab/airyscan/nil.czi"

    @QtCore.pyqtSlot()
    def on_open_button(self):
        logger.info('on_open_button')
        qfd = QtGui.QFileDialog()
        path = os.path.dirname(self.file)
        flt = "zeiss(*.czi)"
        f = QtGui.QFileDialog.getOpenFileName(qfd, "Open File", path, flt)
        if len(f) > 0:
            self.image.file = f
            self.image.zstack = self.zSpin.value()
            self.image.channel = self.dnaSpin.value()
            self.nchLbl.setText("%d channels" % self.image.n_channels)
            self.nzsLbl.setText("%d z-stacks" % self.image.n_zstack)
            self.nfrLbl.setText("%d %s" % (self.image.n_frames, "frames" if self.image.n_frames > 1 else "frame"))

    @QtCore.pyqtSlot()
    def on_img_click(self, ev):
        logger.info('on_img_click')

    @QtCore.pyqtSlot()
    def on_zvalue_change(self):
        logger.info('on_zvalue_change')
        self.image.zstack = self.zSpin.value() % self.image.n_zstack
        self.zSpin.setValue(self.image.zstack)

    @QtCore.pyqtSlot()
    def on_dnaval_change(self):
        logger.info('on_dnaval_change')
        self.image.channel = self.dnaSpin.value() % self.image.n_channels
        self.dnaSpin.setValue(self.image.channel)

    @QtCore.pyqtSlot()
    def on_actval_change(self):
        logger.info('on_actval_change')
        # self.image.zstack = self.zSpin.value()


if __name__ == '__main__':
    import sys
    import os

    from PyQt4.QtCore import QT_VERSION_STR
    from PyQt4.Qt import PYQT_VERSION_STR

    base_path = os.path.abspath('%s' % os.getcwd())
    logging.info('Qt version:' + QT_VERSION_STR)
    logging.info('PyQt version:' + PYQT_VERSION_STR)
    logging.info('Working dir:' + os.getcwd())
    logging.info('Base dir:' + base_path)
    os.chdir(base_path)

    app = QtGui.QApplication(sys.argv)

    gui = RingWindow()
    gui.show()

    sys.exit(app.exec_())

    # qfd = QtGui.QFileDialog()
    # path = "/Volumes/Kidbeat/data"
    # # f = QtGui.QFileDialog.getOpenFileName(qfd, "Select Directory", path, filter)
    # f = QtGui.QFileDialog.getExistingDirectory(qfd, "Select Directory")
    # print(f)
