import logging

import numpy as np
from PyQt4 import Qt, QtCore, QtGui
from PyQt4.QtGui import QBrush, QColor, QPainter, QPen, QPixmap
from gui._image_loading import find_image, retrieve_image
import measurements as m
from shapely.geometry.point import Point
from shapely import affinity
from shapely.wkt import dumps
from shapely.geometry import LineString, MultiLineString, Polygon
from skimage import draw

logger = logging.getLogger('gui.ring.label')
_colors = ["#e5d429", "#a0c334", "#fa5477", "#b5525c", "#000272"]


class RingImageQLabel(QtGui.QLabel):
    clicked = Qt.pyqtSignal()

    def __init__(self, parent, file=None):
        QtGui.QLabel.__init__(self, parent)
        self.selected = True
        self._file = file
        self.nucleiSelected = None
        self.dataHasChanged = False
        self.resolution = None
        self.image_pixmap = None
        self.dwidth = 0
        self.dheight = 0
        self.nd = None

        self.images = None
        self.pix_per_um = None
        self.dt = None
        self.n_frames = None
        self.n_channels = None
        self.n_zstack = None
        self._zstack = 0

        self._dnach = 0
        self._actch = 0
        self._active_ch = "dna"

        self._dnaimage = None
        self._actimage = None
        self._boudaries = None

        self.sel_nuc = None
        self.measurements = None

        self.clear()

    @property
    def active_ch(self):
        return self._active_ch

    @active_ch.setter
    def active_ch(self, value):
        if value is not None:
            self._active_ch = value
            self._repaint()

    @property
    def zstack(self):
        return self._zstack

    @zstack.setter
    def zstack(self, value):
        if value is not None:
            self._zstack = int(value)
            self._boudaries = None
            if self.images is not None:
                self._dnaimage = retrieve_image(self.images, channel=self._dnach, number_of_channels=self.n_channels,
                                                zstack=self.zstack, number_of_zstacks=self.n_zstack, frame=0)
                self._actimage = retrieve_image(self.images, channel=self._actch, number_of_channels=self.n_channels,
                                                zstack=self.zstack, number_of_zstacks=self.n_zstack, frame=0)
            if self.sel_nuc is not None:
                p = self.sel_nuc.centroid
                self._measure(p.x, p.y)
            self._repaint()

    @property
    def dna_channel(self):
        return self._dnach

    @dna_channel.setter
    def dna_channel(self, value):
        if value is not None:
            self._dnach = int(value)

            if self.file is not None:
                self._dnaimage = retrieve_image(self.images, channel=self._dnach, number_of_channels=self.n_channels,
                                                zstack=self.zstack, number_of_zstacks=self.n_zstack, frame=0)
                self._boudaries = None
                p = self.sel_nuc.centroid
                self._measure(p.x, p.y)
                if self.active_ch == "dna":
                    self._repaint()

    @property
    def act_channel(self):
        return self._actch

    @act_channel.setter
    def act_channel(self, value):
        if value is not None:
            self._actch = int(value)

            if self.file is not None:
                self._actimage = retrieve_image(self.images, channel=self._actch, number_of_channels=self.n_channels,
                                                zstack=self.zstack, number_of_zstacks=self.n_zstack, frame=0)
                if self.active_ch == "act":
                    self._repaint()

    @property
    def file(self):
        return self._file

    @file.setter
    def file(self, file):
        if file is not None:
            logger.info('Loading %s' % file)
            self._file = file
            self.images, self.pix_per_um, self.dt, self.n_frames, self.n_channels = find_image(file)
            self.n_zstack = int(len(self.images) / self.n_frames / self.n_channels)
            self.dataHasChanged = True
            self.repaint()

    def clear(self):
        imgarr = np.zeros(shape=(512, 512), dtype=np.uint32)
        qtimage = QtGui.QImage(imgarr.data, imgarr.shape[1], imgarr.shape[0], imgarr.strides[0],
                               QtGui.QImage.Format_RGB32)
        self.image_pixmap = QtGui.QPixmap(qtimage)
        self.setPixmap(self.image_pixmap)

    def _repaint(self):
        self.dataHasChanged = True
        self.repaint()

    def _measure(self, x, y):
        if self._dnaimage is not None and self._boudaries is None:
            logger.debug("computing nuclei boundaries")
            lbl, self._boudaries = m.nuclei_segmentation(self._dnaimage)

        if self._boudaries is not None:
            pt = Point(x, y)

            for nucleus in self._boudaries:
                if nucleus["boundary"].contains(pt):
                    self.sel_nuc = nucleus["boundary"]

            rngtck = 3 * self.pix_per_um
            pol = self.sel_nuc.buffer(-rngtck / 2)
            lines = m.measure_lines_around_polygon(self._actimage, pol, pix_per_um=self.pix_per_um)
            self.measurements = list()
            for k, (ls, l) in enumerate(lines):
                self.measurements.append({'n': k, 'x': x, 'y': y, 'l': l, 'c': _colors[k],
                                          'ls0': ls.coords[0], 'ls1': ls.coords[1],
                                          'd': max(l) - min(l), 'sum': np.sum(l)})

    def mouseReleaseEvent(self, ev):
        pos = ev.pos()
        # convert to image pixel coords
        x = pos.x() * self.dwidth / self.width()
        y = pos.y() * self.dheight / self.height()
        logger.info('clicked! X%d Y%d' % (x, y))

        self.sel_nuc = None
        self._measure(x, y)

        # self.emit(QtCore.SIGNAL('clicked()'))
        self.clicked.emit()
        self._repaint()

    def paintMeasure(self):
        logger.debug("painting measurement")
        data = retrieve_image(self.images, channel=self._actch, number_of_channels=self.n_channels,
                              zstack=self.zstack, number_of_zstacks=self.n_zstack, frame=0)
        self.dwidth, self.dheight = data.shape
        print(data.shape, self.dwidth, self.dheight)

        # map the data range to 0 - 255
        img_8bit = ((data - data.min()) / (data.ptp() / 255.0)).astype(np.uint8)

        for me in self.measurements:
            r0, c0, r1, c1 = np.array(list(me['ls0']) + list(me['ls1'])).astype(int)
            rr, cc = draw.line(r0, c0, r1, c1)
            img_8bit[cc, rr] = 255

        qtimage = QtGui.QImage(img_8bit.repeat(4), self.dwidth, self.dheight, QtGui.QImage.Format_RGB32)
        self.image_pixmap = QPixmap(qtimage)
        # self.draw_measurements()
        self.setPixmap(self.image_pixmap)
        return

    def paintEvent(self, event):
        if self.dataHasChanged:
            self.dataHasChanged = False
            ch = self.act_channel if self.active_ch == "act" else self.dna_channel
            data = retrieve_image(self.images, channel=ch, number_of_channels=self.n_channels,
                                  zstack=self.zstack, number_of_zstacks=self.n_zstack, frame=0)
            self.dwidth, self.dheight = data.shape

            # map the data range to 0 - 255
            img_8bit = ((data - data.min()) / (data.ptp() / 255.0)).astype(np.uint8)
            qtimage = QtGui.QImage(img_8bit.repeat(4), self.dwidth, self.dheight, QtGui.QImage.Format_RGB32)
            self.image_pixmap = QPixmap(qtimage)
            self.draw_measurements()
            self.setPixmap(self.image_pixmap)

        return QtGui.QLabel.paintEvent(self, event)

    def draw_measurements(self):
        if self.sel_nuc is None: return

        painter = QPainter()
        painter.begin(self.image_pixmap)
        painter.setRenderHint(QPainter.Antialiasing)

        rng_thick = 3
        rng_thick *= self.pix_per_um

        if self.active_ch == "dna":
            # get nuclei boundary as a polygon
            nucb_qpoints_e = [Qt.QPoint(x, y) for x, y in self.sel_nuc.buffer(rng_thick).exterior.coords]
            nucb_qpoints_i = [Qt.QPoint(x, y) for x, y in self.sel_nuc.exterior.coords]

            painter.setPen(QPen(QBrush(QColor('white')), 3))
            painter.drawPolygon(Qt.QPolygon(nucb_qpoints_i))
            painter.drawPolygon(Qt.QPolygon(nucb_qpoints_e))

            nucb_poly = Qt.QPolygon(nucb_qpoints_e).subtracted(Qt.QPolygon(nucb_qpoints_i))
            brush = QBrush(QtCore.Qt.BDiagPattern)
            brush.setColor(QColor('white'))
            painter.setBrush(brush)
            painter.setPen(QPen(QBrush(QColor('transparent')), 0))

            painter.drawPolygon(nucb_poly)

            for me in self.measurements:
                painter.setPen(QPen(QBrush(QColor(me['c'])), 5))
                pts = [Qt.QPoint(x, y) for x, y in [me['ls0'], me['ls1']]]
                painter.drawLine(pts[0], pts[1])

        elif self.active_ch == "act":
            nuc_pen = QPen(QBrush(QColor('red')), 2)
            nuc_pen.setStyle(QtCore.Qt.DotLine)
            painter.setPen(nuc_pen)
            for n in [e["boundary"] for e in self._boudaries]:
                # get nuclei boundary as a polygon
                nucb_qpoints = [Qt.QPoint(x, y) for x, y in n.exterior.coords]
                painter.drawPolygon(Qt.QPolygon(nucb_qpoints))

            for me in self.measurements:
                painter.setPen(QPen(QBrush(QColor(me['c'])), 5))
                pts = [Qt.QPoint(x, y) for x, y in [me['ls0'], me['ls1']]]
                painter.drawLine(pts[0], pts[1])

        painter.end()
