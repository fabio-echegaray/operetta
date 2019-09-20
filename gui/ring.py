import os
import sys
import logging

import numpy as np
import pandas as pd
import seaborn as sns
from PyQt4 import Qt, QtCore, QtGui, uic
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
    linePicked = Qt.pyqtSignal()

    def __init__(self):
        super(GraphWidget, self).__init__()
        path = os.path.join(sys.path[0], __package__)
        uic.loadUi(os.path.join(path, 'gui_ring_graph.ui'), self)
        self.canvas.callbacks.connect('pick_event', self.on_pick)

        self.selectedLine = None

        self.graphWidget.clear()
        self.format_ax()

    @property
    def canvas(self):
        return self.graphWidget.canvas

    @property
    def ax(self):
        return self.canvas.ax

    def clear(self):
        self.graphWidget.clear()
        self.selectedLine = None

    def on_pick(self, event):
        logger.info('on_pick')
        for l in self.ax.lines:
            l.set_linewidth(0.1)
        event.artist.set_linewidth(0.5)
        # logger.debug([l.get_label() for l in self.ax.lines])
        self.selectedLine = int(event.artist.get_label())
        self.emit(QtCore.SIGNAL('linePicked()'))
        self.canvas.draw()

    def format_ax(self):
        self.ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        self.ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.5))
        self.ax.yaxis.set_major_locator(ticker.MultipleLocator(1e4))
        self.ax.yaxis.set_minor_locator(ticker.MultipleLocator(5e3))
        self.ax.yaxis.set_major_formatter(EngFormatter(unit=''))
        # self.ax.set_ylim((0, 3e4))

    def resizeEvent(self, event):
        self.graphWidget.setFixedWidth(self.width())


class RingWindow(QMainWindow):
    image: RingImageQLabel
    statusbar: QtGui.QStatusBar

    def __init__(self):
        super(RingWindow, self).__init__()
        path = os.path.join(sys.path[0], __package__)

        uic.loadUi(os.path.join(path, 'gui_ring.ui'), self)
        self.move(50, 0)

        self.ctrl = QWidget()
        uic.loadUi(os.path.join(path, 'gui_ring_controls.ui'), self.ctrl)
        self.ctrl.show()

        self.ctrl.zSpin.valueChanged.connect(self.onZValueChange)
        self.ctrl.openButton.pressed.connect(self.onOpenButton)
        self.ctrl.addButton.pressed.connect(self.onAddButton)
        self.ctrl.measureButton.pressed.connect(self.onMeasureButton)
        self.ctrl.dnaSpin.valueChanged.connect(self.onDnaValChange)
        self.ctrl.actSpin.valueChanged.connect(self.onActValChange)
        self.ctrl.dnaChk.toggled.connect(self.onImgToggle)
        self.ctrl.actChk.toggled.connect(self.onImgToggle)
        self.ctrl.renderChk.stateChanged.connect(self.onRenderChk)

        self.image.clicked.connect(self.onImgUpdate)
        self.image.lineUpdated.connect(self.onImgUpdate)
        self.image.linePicked.connect(self.onLinePickedFromImage)
        self.image.dnaChannel = self.ctrl.dnaSpin.value()
        self.image.actChannel = self.ctrl.actSpin.value()
        # layout = QtGui.QBoxLayout()

        self.grph = GraphWidget()
        self.grph.show()

        self.grph.linePicked.connect(self.onLinePickedFromGraph)

        self.ctrl.setWindowFlags(self.ctrl.windowFlags() & ~QtCore.Qt.WindowStaysOnTopHint)
        self.grph.setWindowFlags(self.grph.windowFlags() & ~QtCore.Qt.WindowStaysOnTopHint)

        self.image.dnaChannel = self.ctrl.dnaSpin.value()
        self.image.actChannel = self.ctrl.actSpin.value()

        self.measure_n = 0
        self.selectedLine = None

        self.df = pd.DataFrame()
        self.file = "/Users/Fabio/data/lab/airyscan/nil.czi"

        self.resizeEvent(None)
        self.moveEvent(None)

    def resizeEvent(self, event):
        # this is a hack to resize everithing when the user resizes the main window
        self.grph.setFixedWidth(self.width())
        self.image.setFixedWidth(self.width())
        self.image.setFixedHeight(self.height())
        self.image.resizeEvent(None)
        self.moveEvent(None)

    def moveEvent(self, QMoveEvent):
        px = self.geometry().x()
        py = self.geometry().y()
        pw = self.geometry().width()
        ph = self.geometry().height()

        dw = self.ctrl.width()
        dh = self.ctrl.height()
        self.ctrl.setGeometry(px + pw, py, dw, dh)

        dw = self.grph.width()
        dh = self.grph.height()
        self.grph.setGeometry(px, py + ph + 20, dw, dh)
        # super(RingWindow, self).mouseMoveEvent(event)

    def closeEvent(self, event):
        if not self.df.empty:
            self.df.loc[:, "condition"] = self.ctrl.experimentLineEdit.text()
            self.df.loc[:, "l"] = self.df.loc[:, "l"].apply(lambda v: np.array2string(v, separator=','))
            self.df.to_csv(os.path.join(os.path.dirname(self.image.file), "ringlines.csv"))
        self.grph.close()
        self.ctrl.close()

    def focusInEvent(self, QFocusEvent):
        logger.debug('focusInEvent')
        self.ctrl.activateWindow()
        self.grph.activateWindow()

    def showEvent(self, event):
        self.setFocus()

    def _graphTendency(self):
        df = pd.DataFrame(self.image.measurements).drop(['x', 'y', 'c', 'ls0', 'ls1', 'd', 'sum'], axis=1)
        df.loc[:, "xx"] = df.loc[:, "l"].apply(
            lambda v: np.arange(start=0, stop=len(v) * self.image.dl, step=self.image.dl))
        df = m.vector_column_to_long_fmt(df, val_col="l", ix_col="xx")
        sns.lineplot(x="xx", y="l", data=df, ax=self.grph.ax, color='k', ci="sd", zorder=20)
        self.grph.ax.set_ylabel('')
        self.grph.ax.set_xlabel('')
        self.grph.canvas.draw()

    def _graph(self, alpha=1.0):
        self.grph.clear()
        if self.image.measurements is not None:
            for me in self.image.measurements:
                x = np.arange(start=0, stop=len(me['l']) * self.image.dl, step=self.image.dl)
                lw = 0.1 if self.image.selectedLine is not None and me != self.image.selectedLine else 0.5
                self.grph.ax.plot(x, me['l'], linewidth=lw, linestyle='-', color=me['c'], alpha=alpha, zorder=10,
                                  picker=5, label=me['n'])
            self.grph.format_ax()
            self.statusbar.showMessage("ptp: %s" % ["%d " % me['d'] for me in self.image.measurements])
            self.grph.canvas.draw()

    @QtCore.pyqtSlot()
    def onImgToggle(self):
        logger.info('onImgToggle')
        if self.ctrl.dnaChk.isChecked():
            self.image.activeCh = "dna"
        if self.ctrl.actChk.isChecked():
            self.image.activeCh = "act"

    @QtCore.pyqtSlot()
    def onRenderChk(self):
        logger.info('onRenderChk')
        self.image.render = self.ctrl.renderChk.isChecked()

    @QtCore.pyqtSlot()
    def onOpenButton(self):
        logger.info('onOpenButton')
        qfd = QtGui.QFileDialog()
        path = os.path.dirname(self.file)
        if self.image.file is not None:
            self.statusbar.showMessage("current file: %s" % os.path.basename(self.image.file))
        flt = "zeiss(*.czi)"
        f = QtGui.QFileDialog.getOpenFileName(qfd, "Open File", path, flt)
        if len(f) > 0:
            self.image.file = f
            self.image.zstack = self.ctrl.zSpin.value()
            self.image.dnaChannel = self.ctrl.dnaSpin.value()
            self.ctrl.nchLbl.setText("%d channels" % self.image.nChannels)
            self.ctrl.nzsLbl.setText("%d z-stacks" % self.image.nZstack)
            self.ctrl.nfrLbl.setText("%d %s" % (self.image.nFrames, "frames" if self.image.nFrames > 1 else "frame"))

    @QtCore.pyqtSlot()
    def onImgUpdate(self):
        logger.info('onImgUpdate')
        self.ctrl.renderChk.setChecked(True)
        self._graph()

    @QtCore.pyqtSlot()
    def onMeasureButton(self):
        logger.info('onMeasureButton')
        self.image.paint_measures()
        self._graph(alpha=0.2)
        self._graphTendency()

    @QtCore.pyqtSlot()
    def onZValueChange(self):
        logger.info('onZValueChange')
        self.image.zstack = self.ctrl.zSpin.value() % self.image.nZstack
        self.ctrl.zSpin.setValue(self.image.zstack)
        self._graph()

    @QtCore.pyqtSlot()
    def onDnaValChange(self):
        logger.info('onDnaValChange')
        val = self.ctrl.dnaSpin.value() % self.image.nChannels
        self.ctrl.dnaSpin.setValue(val)
        self.image.dnaChannel = val
        if self.ctrl.dnaChk.isChecked():
            self.image.activeCh = "dna"
        self.ctrl.dnaChk.setChecked(True)

    @QtCore.pyqtSlot()
    def onActValChange(self):
        logger.info('onActValChange')
        val = self.ctrl.actSpin.value() % self.image.nChannels
        self.ctrl.actSpin.setValue(val)
        self.image.actChannel = val
        if self.ctrl.actChk.isChecked():
            self.image.activeCh = "act"
        self.ctrl.actChk.setChecked(True)

    @QtCore.pyqtSlot()
    def onAddButton(self):
        logger.info('onAddButton')
        if self.image.measurements is not None:
            new = pd.DataFrame(self.image.measurements)
            if self.selectedLine is not None:
                new = new.loc[new["n"] == self.selectedLine]
            new.loc[:, "m"] = self.measure_n
            new.loc[:, "z"] = self.image.zstack
            new.loc[:, "file"] = os.path.basename(self.image.file)
            # new.loc[:, "x"] = new.loc[:, "l"].apply(lambda v: np.arange(start=0, stop=len(v), step=self.image.dl))
            self.df = self.df.append(new, ignore_index=True, sort=False)
            self.measure_n += 1
            print(self.df)

    @QtCore.pyqtSlot()
    def onLinePickedFromGraph(self):
        logger.info('onLinePickedFromGraph')
        self.selectedLine = self.grph.selectedLine if self.grph.selectedLine is not None else None
        if self.selectedLine is not None:
            self.image.selectedLine = self.selectedLine
            self.statusbar.showMessage("line %d selected" % self.selectedLine)

    @QtCore.pyqtSlot()
    def onLinePickedFromImage(self):
        logger.info('onLinePickedFromImage')
        self.selectedLine = self.image.selectedLine['n'] if self.image.selectedLine is not None else None
        if self.selectedLine is not None:
            self.statusbar.showMessage("line %d selected" % self.selectedLine)


if __name__ == '__main__':
    import sys
    import os

    from PyQt4.QtCore import QT_VERSION_STR
    from PyQt4.Qt import PYQT_VERSION_STR

    base_path = os.path.abspath('%s' % os.getcwd())
    logging.getLogger('matplotlib').setLevel(logging.ERROR)
    logging.info('Qt version:' + QT_VERSION_STR)
    logging.info('PyQt version:' + PYQT_VERSION_STR)
    logging.info('Working dir:' + os.getcwd())
    logging.info('Base dir:' + base_path)
    os.chdir(base_path)

    app = QtGui.QApplication(sys.argv)

    gui = RingWindow()
    gui.show()

    sys.exit(app.exec_())
