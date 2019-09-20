import math
import logging
import itertools

import numpy as np
import seaborn as sns
from PyQt4 import Qt, QtCore, QtGui
from PyQt4.QtGui import QBrush, QColor, QPainter, QPen, QPixmap
from shapely.geometry.point import Point
from skimage import draw

from gui._image_loading import find_image, retrieve_image
import measurements as m

logger = logging.getLogger('gui.ring.label')
_nlin = 20
_colors = sns.husl_palette(_nlin, h=.5).as_hex()


def distance(a, b):
    return np.sqrt((a.x() - b.x()) ** 2 + (a.y() - b.y()) ** 2)


def is_between(c, a, b):
    dist = distance(a, c) + distance(c, b) - distance(a, b)
    return math.isclose(dist, 0, abs_tol=1)


class RingImageQLabel(QtGui.QLabel):
    clicked = Qt.pyqtSignal()
    lineUpdated = Qt.pyqtSignal()
    linePicked = Qt.pyqtSignal()

    def __init__(self, parent, file=None):
        QtGui.QLabel.__init__(self, parent)
        self.selected = True
        self._file = file
        self.nucleiSelected = None
        self.dataHasChanged = False
        self.resolution = None
        self.imagePixmap = None
        self.dwidth = 0
        self.dheight = 0
        self.nd = None

        self.images = None
        self.pix_per_um = None
        self.dt = None
        self.dl = 0.05
        self.nFrames = None
        self.nChannels = None
        self.nZstack = None
        self._zstack = 0

        self._dnach = 0
        self._actch = 0
        self._activeCh = "dna"

        self._dnaimage = None
        self._actimage = None
        self._boudaries = None

        self._render = True
        self.selNuc = None
        self.measurements = None
        self.mousePos = Qt.QPoint(0, 0)
        self._selectedLine = None
        self.measureLocked = False

        self.setMouseTracking(True)
        self.clear()

    @property
    def activeCh(self):
        return self._activeCh

    @activeCh.setter
    def activeCh(self, value):
        if value is not None:
            self._activeCh = value
            self._repaint()

    @property
    def selectedLine(self):
        # return self._selectedLine['n'] if self._selectedLine is not None else None
        return self._selectedLine

    @selectedLine.setter
    def selectedLine(self, value):
        if type(value) == dict:
            self._selectedLine = value
        elif type(value) == int:
            self._selectedLine = None
            for me in self.measurements:
                if me['n'] == value:
                    self._selectedLine = me
                    self._repaint()
        else:
            self._selectedLine = None

    @property
    def render(self):
        return self._render

    @render.setter
    def render(self, value):
        if value is not None:
            self._render = value
            self._repaint()
            self.setMouseTracking(self._render)

    @property
    def zstack(self):
        return self._zstack

    @zstack.setter
    def zstack(self, value):
        if value is not None:
            self._zstack = int(value)
            self._boudaries = None
            if self.images is not None:
                self._dnaimage = retrieve_image(self.images, channel=self._dnach, number_of_channels=self.nChannels,
                                                zstack=self.zstack, number_of_zstacks=self.nZstack, frame=0)
                self._actimage = retrieve_image(self.images, channel=self._actch, number_of_channels=self.nChannels,
                                                zstack=self.zstack, number_of_zstacks=self.nZstack, frame=0)
            if self.selNuc is not None:
                p = self.selNuc.centroid
                self._measure(p.x, p.y)
            self._repaint()

    @property
    def dnaChannel(self):
        return self._dnach

    @dnaChannel.setter
    def dnaChannel(self, value):
        if value is not None:
            self._dnach = int(value)

            if self.file is not None:
                self._dnaimage = retrieve_image(self.images, channel=self._dnach, number_of_channels=self.nChannels,
                                                zstack=self.zstack, number_of_zstacks=self.nZstack, frame=0)
                self._boudaries = None
                if self.selNuc is not None:
                    p = self.selNuc.centroid
                    self._measure(p.x, p.y)
                    if self.activeCh == "dna":
                        self._repaint()

    @property
    def actChannel(self):
        return self._actch

    @actChannel.setter
    def actChannel(self, value):
        if value is not None:
            self._actch = int(value)

            if self.file is not None:
                self._actimage = retrieve_image(self.images, channel=self._actch, number_of_channels=self.nChannels,
                                                zstack=self.zstack, number_of_zstacks=self.nZstack, frame=0)
                if self.activeCh == "act":
                    self._repaint()

    @property
    def file(self):
        return self._file

    @file.setter
    def file(self, file):
        if file is not None:
            logger.info('Loading %s' % file)
            self._file = file
            self.images, self.pix_per_um, self.dt, self.nFrames, self.nChannels = find_image(file)
            self.nZstack = int(len(self.images) / self.nFrames / self.nChannels)
            self._repaint()
            logger.info("pixels per um: %0.4f" % self.pix_per_um)

    def clear(self):
        imgarr = np.zeros(shape=(512, 512), dtype=np.uint32)
        qtimage = QtGui.QImage(imgarr.data, imgarr.shape[1], imgarr.shape[0], imgarr.strides[0],
                               QtGui.QImage.Format_RGB32)
        self.imagePixmap = QtGui.QPixmap(qtimage)
        self.setPixmap(self.imagePixmap)

    def _repaint(self):
        self.dataHasChanged = True
        self.repaint()

    def _measure(self, x, y):
        self.measurements = None
        if self._dnaimage is not None and self._boudaries is None:
            logger.debug("computing nuclei boundaries")
            lbl, self._boudaries = m.nuclei_segmentation(self._dnaimage, simp_px=self.pix_per_um / 4)

        if self._boudaries is not None:
            pt = Point(x, y)

            for nucleus in self._boudaries:
                if nucleus["boundary"].contains(pt):
                    self.selNuc = nucleus["boundary"]

            if self.selNuc is None: return
            lines = m.measure_lines_around_polygon(self._actimage, self.selNuc, rng_thick=4, dl=self.dl,
                                                   n_lines=_nlin, pix_per_um=self.pix_per_um)
            self.measurements = list()
            for k, ((ls, l), colr) in enumerate(zip(lines, itertools.cycle(_colors))):
                self.measurements.append({'n': k, 'x': x, 'y': y, 'l': l, 'c': colr,
                                          'ls0': ls.coords[0], 'ls1': ls.coords[1],
                                          'd': max(l) - min(l), 'sum': np.sum(l)})

    def mouseMoveEvent(self, event):
        # logger.debug('mouseMoveEvent')
        if not self.measureLocked and event.type() == QtCore.QEvent.MouseMove and self.measurements is not None:
            if event.buttons() == QtCore.Qt.NoButton:
                pos = event.pos()
                # convert to image pixel coords
                x = pos.x() * self.dwidth / self.width()
                y = pos.y() * self.dheight / self.height()
                self.mousePos = Qt.QPoint(x, y)

                # print("------------------------------------------------------")
                for me in self.measurements:
                    pts = [Qt.QPoint(x, y) for x, y in [me['ls0'], me['ls1']]]
                    # print("X %d %d %d | Y %d %d %d" % (
                    #     min(pts[0].x(), pts[1].x()), self.mouse_pos.x(), max(pts[0].x(), pts[1].x()),
                    #     min(pts[0].y(), pts[1].y()), self.mouse_pos.y(), max(pts[0].y(), pts[1].y())))
                    if is_between(self.mousePos, pts[0], pts[1]):
                        if me != self.selectedLine:
                            self.selectedLine = me
                            self.emit(QtCore.SIGNAL('lineUpdated()'))
                            self._repaint()
                            break

    def mouseReleaseEvent(self, ev):
        pos = ev.pos()
        # convert to image pixel coords
        x = pos.x() * self.dwidth / self.width()
        y = pos.y() * self.dheight / self.height()
        logger.debug('clicked! X%d Y%d' % (x, y))

        anyLineSelected = False
        lineChanged = False
        if self.measurements is not None:
            for me in self.measurements:
                pts = [Qt.QPoint(x, y) for x, y in [me['ls0'], me['ls1']]]
                if is_between(self.mousePos, pts[0], pts[1]):
                    anyLineSelected = True
                    if me != self.selectedLine:
                        lineChanged = True
                        break

        if anyLineSelected and not lineChanged and not self.measureLocked:
            self.clicked.emit()
            self.measureLocked = True
            self.emit(QtCore.SIGNAL('linePicked()'))
        else:
            self.measureLocked = False
            self.selectedLine = None
            self.selNuc = None
            self._measure(x, y)
            self._repaint()
            self.clicked.emit()

    def paint_measures(self):
        logger.debug("painting measurement")
        data = retrieve_image(self.images, channel=self._actch, number_of_channels=self.nChannels,
                              zstack=self.zstack, number_of_zstacks=self.nZstack, frame=0)
        self.dwidth, self.dheight = data.shape
        # print(data.shape, self.dwidth, self.dheight)

        # map the data range to 0 - 255
        img8bit = ((data - data.min()) / (data.ptp() / 255.0)).astype(np.uint8)

        for me in self.measurements:
            r0, c0, r1, c1 = np.array(list(me['ls0']) + list(me['ls1'])).astype(int)
            rr, cc = draw.line(r0, c0, r1, c1)
            img8bit[cc, rr] = 255
            rr, cc = draw.circle(r0, c0, 3)
            img8bit[cc, rr] = 255

        qtimage = QtGui.QImage(img8bit.repeat(4), self.dwidth, self.dheight, QtGui.QImage.Format_RGB32)
        self.imagePixmap = QPixmap(qtimage)
        self.setPixmap(self.imagePixmap)
        return

    def resizeEvent(self, QResizeEvent):
        # this is a hack to resize everithing when the user resizes the main window
        if self.dwidth == 0: return
        ratio = self.dheight / self.dwidth
        self.setFixedWidth(self.width())
        self.setFixedHeight(int(self.width()) * ratio)

    def paintEvent(self, event):
        if self.dataHasChanged:
            self.dataHasChanged = False
            ch = self.actChannel if self.activeCh == "act" else self.dnaChannel
            data = retrieve_image(self.images, channel=ch, number_of_channels=self.nChannels,
                                  zstack=self.zstack, number_of_zstacks=self.nZstack, frame=0)
            self.dwidth, self.dheight = data.shape

            # map the data range to 0 - 255
            img_8bit = ((data - data.min()) / (data.ptp() / 255.0)).astype(np.uint8)
            qtimage = QtGui.QImage(img_8bit.repeat(4), self.dwidth, self.dheight, QtGui.QImage.Format_RGB32)
            self.imagePixmap = QPixmap(qtimage)
            if self.render:
                self._drawMeasurements()
            self.setPixmap(self.imagePixmap)

        return QtGui.QLabel.paintEvent(self, event)

    def _drawMeasurements(self):
        if self.selNuc is None: return

        painter = QPainter()
        painter.begin(self.imagePixmap)
        painter.setRenderHint(QPainter.Antialiasing)

        rng_thick = 3
        rng_thick *= self.pix_per_um

        if self.activeCh == "dna":
            # get nuclei boundary as a polygon
            nucb_qpoints_e = [Qt.QPoint(x, y) for x, y in self.selNuc.buffer(rng_thick).exterior.coords]
            nucb_qpoints_i = [Qt.QPoint(x, y) for x, y in self.selNuc.exterior.coords]

            painter.setPen(QPen(QBrush(QColor('white')), 3))
            painter.drawPolygon(Qt.QPolygon(nucb_qpoints_i))
            painter.drawPolygon(Qt.QPolygon(nucb_qpoints_e))

            nucb_poly = Qt.QPolygon(nucb_qpoints_e).subtracted(Qt.QPolygon(nucb_qpoints_i))
            brush = QBrush(QtCore.Qt.BDiagPattern)
            brush.setColor(QColor('white'))
            painter.setBrush(brush)
            painter.setPen(QPen(QBrush(QColor('transparent')), 0))

            painter.drawPolygon(nucb_poly)

        elif self.activeCh == "act":
            nuc_pen = QPen(QBrush(QColor('red')), 2)
            nuc_pen.setStyle(QtCore.Qt.DotLine)
            painter.setPen(nuc_pen)
            for n in [e["boundary"] for e in self._boudaries]:
                # get nuclei boundary as a polygon
                nucb_qpoints = [Qt.QPoint(x, y) for x, y in n.exterior.coords]
                painter.drawPolygon(Qt.QPolygon(nucb_qpoints))

        for me in self.measurements:
            painter.setPen(
                QPen(QBrush(QColor(me['c'])), 2 * self.pix_per_um if me == self.selectedLine else self.pix_per_um))
            pts = [Qt.QPoint(x, y) for x, y in [me['ls0'], me['ls1']]]
            painter.drawLine(pts[0], pts[1])

        painter.end()
