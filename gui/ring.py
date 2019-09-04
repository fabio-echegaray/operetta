import logging

import numpy as np
import pandas as pd
import seaborn as sns
from PyQt4 import QtCore, QtGui, uic
from PyQt4.QtGui import QMainWindow, QWidget
from matplotlib.figure import SubplotParams
import matplotlib.ticker as ticker
from matplotlib.ticker import EngFormatter

from gui._ring_label import RingImageQLabel
from gui.gui_mplwidget import MplWidget
import measurements as m

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('ring.gui')
sp = SubplotParams(left=0., bottom=0., right=1., top=1.)
mydpi = 72

pd.set_option('display.width', 320)
pd.set_option('display.max_columns', 20)
pd.set_option('display.max_rows', 100)


class GraphWidget(QWidget):
    graphWidget: MplWidget

    def __init__(self):
        super(GraphWidget, self).__init__()
        uic.loadUi('./gui_ring_graph.ui', self)
        self.graphWidget.clear()
        self.format_ax()

    @property
    def canvas(self):
        return self.graphWidget.canvas

    @property
    def ax(self):
        return self.canvas.ax

    def format_ax(self):
        self.ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
        self.ax.xaxis.set_minor_locator(ticker.MultipleLocator(5))
        self.ax.yaxis.set_major_locator(ticker.MultipleLocator(1e4))
        self.ax.yaxis.set_minor_locator(ticker.MultipleLocator(5e3))
        self.ax.yaxis.set_major_formatter(EngFormatter(unit=''))
        self.ax.set_ylim((0, 3e4))


class RingWindow(QMainWindow):
    image: RingImageQLabel
    statusbar: QtGui.QStatusBar

    def __init__(self):
        super(RingWindow, self).__init__()
        uic.loadUi('./gui_ring.ui', self)
        self.move(50, 100)
        self.zSpin.valueChanged.connect(self.on_zvalue_change)
        self.openButton.pressed.connect(self.on_open_button)
        self.addButton.pressed.connect(self.on_add_button)
        self.measureButton.pressed.connect(self.on_me_button)
        self.dnaSpin.valueChanged.connect(self.on_dnaval_change)
        self.actSpin.valueChanged.connect(self.on_actval_change)
        self.dnaChk.toggled.connect(self.on_img_toggle)
        self.actChk.toggled.connect(self.on_img_toggle)
        self.image.clicked.connect(self.on_img_click)
        self.renderChk.stateChanged.connect(self.on_render_chk)

        self.image.dna_channel = self.dnaSpin.value()
        self.image.act_channel = self.actSpin.value()

        self.grph = GraphWidget()
        ph = self.geometry().height()
        px = self.geometry().x()
        py = self.geometry().y()
        dw = self.grph.width()
        dh = self.grph.height()
        self.grph.setGeometry(px, py + ph + 50, dw, dh)
        self.grph.show()

        self.measure_n = 0

        self.df = pd.DataFrame()

        self.file = "/Users/Fabio/data/lab/airyscan/nil.czi"

    def closeEvent(self, event):
        if not self.df.empty:
            self.df.loc[:, "condition"] = self.experimentLineEdit.text()
            self.df.loc[:, "l"] = self.df.loc[:, "l"].apply(lambda v: np.array2string(v, separator=','))
            self.df.to_csv("out.csv")
        self.grph.close()

    def showEvent(self, event):
        self.setFocus()

    def keyPressEvent(self, event):
        key = event.key()
        print(key)

        if key == QtCore.Qt.Key_A:
            print('a pressed')
            # self.image.clear()
            self.image.paint_measures()
        elif key == QtCore.Qt.Key_Left:
            print('Left Arrow Pressed')

    def _graph_tendency(self):
        df = pd.DataFrame(self.image.measurements).drop(['x', 'y', 'c', 'ls0', 'ls1', 'd', 'sum'], axis=1)
        # flix = df.loc[:, "l"].apply(lambda v: v.ptp() > 1000)
        # df = df[flix]
        df.loc[:, "x"] = df.loc[:, "l"].apply(lambda v: np.arange(start=0, stop=len(v), step=1))
        df = m.vector_column_to_long_fmt(df, val_col="l", ix_col="x")
        sns.lineplot(x="x", y="l", data=df, ax=self.grph.ax, color='k', ci="sd", zorder=20)
        self.grph.ax.set_ylabel('')
        self.grph.ax.set_xlabel('')
        self.grph.canvas.draw()

    def _graph(self, alpha=1.0):
        if self.image.measurements is not None:
            self.grph.ax.cla()
            for me in self.image.measurements:
                x = np.arange(start=0, stop=len(me['l']), step=1)
                self.grph.ax.plot(x, me['l'], linewidth=0.5, linestyle='-', color=me['c'], alpha=alpha, zorder=10)
            self.grph.format_ax()
            self.statusbar.showMessage("ptp: %s" % ["%d " % me['d'] for me in self.image.measurements])
            self.grph.canvas.draw()

    @QtCore.pyqtSlot()
    def on_img_toggle(self):
        logger.info('on_img_toggle')
        if self.dnaChk.isChecked():
            self.image.active_ch = "dna"
        if self.actChk.isChecked():
            self.image.active_ch = "act"

    @QtCore.pyqtSlot()
    def on_render_chk(self):
        logger.info('on_render_chk')
        self.image.render = self.renderChk.isChecked()

    @QtCore.pyqtSlot()
    def on_open_button(self):
        logger.info('on_open_button')
        qfd = QtGui.QFileDialog()
        path = os.path.dirname(self.file)
        if self.image.file is not None:
            self.statusbar.showMessage("current file: %s" % os.path.basename(self.image.file))
        flt = "zeiss(*.czi)"
        f = QtGui.QFileDialog.getOpenFileName(qfd, "Open File", path, flt)
        if len(f) > 0:
            self.image.file = f
            self.image.zstack = self.zSpin.value()
            self.image.dna_channel = self.dnaSpin.value()
            self.nchLbl.setText("%d channels" % self.image.n_channels)
            self.nzsLbl.setText("%d z-stacks" % self.image.n_zstack)
            self.nfrLbl.setText("%d %s" % (self.image.n_frames, "frames" if self.image.n_frames > 1 else "frame"))

    @QtCore.pyqtSlot()
    def on_img_click(self):
        logger.info('on_img_click')
        self._graph()

    @QtCore.pyqtSlot()
    def on_me_button(self):
        logger.info('on_me_button')
        self.image.paint_measures()
        self._graph(alpha=0.2)
        self._graph_tendency()

    @QtCore.pyqtSlot()
    def on_zvalue_change(self):
        logger.info('on_zvalue_change')
        self.image.zstack = self.zSpin.value() % self.image.n_zstack
        self.zSpin.setValue(self.image.zstack)
        self._graph()

    @QtCore.pyqtSlot()
    def on_dnaval_change(self):
        logger.info('on_dnaval_change')
        val = self.dnaSpin.value() % self.image.n_channels
        self.dnaSpin.setValue(val)
        self.image.dna_channel = val
        if self.dnaChk.isChecked():
            self.image.active_ch = "dna"

    @QtCore.pyqtSlot()
    def on_actval_change(self):
        logger.info('on_actval_change')
        val = self.actSpin.value() % self.image.n_channels
        self.actSpin.setValue(val)
        self.image.act_channel = val
        if self.actChk.isChecked():
            self.image.active_ch = "act"

    @QtCore.pyqtSlot()
    def on_add_button(self):
        logger.info('on_add_button')
        if self.image.measurements is not None:
            new = pd.DataFrame(self.image.measurements)
            new.loc[:, "m"] = self.measure_n
            new.loc[:, "z"] = self.image.zstack
            new.loc[:, "file"] = os.path.basename(self.image.file)
            dl = 0.05
            new.loc[:, "x"] = new.loc[:, "l"].apply(lambda v: np.arange(start=0, stop=len(v), step=1))
            self.df = self.df.append(new, ignore_index=True, sort=False)
            self.measure_n += 1
            print(self.df)


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
