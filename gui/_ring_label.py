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

logger = logging.getLogger('gui.ring')


class RingImageQLabel(QtGui.QLabel):
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
        self._channel = 0

        self._image = None
        self._boudaries = None

        self.sel_nuc = None
        self.measurement = None

        self.clear()

    @property
    def zstack(self):
        return self._zstack

    @zstack.setter
    def zstack(self, value):
        if value is not None:
            self._zstack = int(value)
            self._repaint()

    @property
    def channel(self):
        return self._channel

    @channel.setter
    def channel(self, value):
        if value is not None:
            self._channel = int(value)
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

    def mouseReleaseEvent(self, ev):
        pos = ev.pos()
        # convert to image pixel coords
        x = pos.x() * self.dwidth / self.width()
        y = pos.y() * self.dheight / self.height()
        logger.info('clicked! X%d Y%d' % (x, y))
        self.emit(QtCore.SIGNAL('clicked()'))

        self.sel_nuc = None

        if self._image is not None:
            logger.debug("computing boundaries")
            lbl, self._boudaries = m.nuclei_segmentation(self._image)

        if self._boudaries is not None:
            pt = Point(x, y)

            for nucleus in self._boudaries:
                if nucleus["boundary"].contains(pt):
                    self.sel_nuc = nucleus["boundary"]

            lines = m.measure_lines_around_polygon(self._image, self.sel_nuc, pix_per_um=self.pix_per_um)
            self.measurement = list()
            for k, (ls, l) in enumerate(lines):
                # print(np.array2string(l, separator=','))
                self.measurement.append({'n': k, 'x': x, 'y': y, 'l': l,
                                         'ls0': ls.coords[0], 'ls1': ls.coords[1],
                                         'd': max(l) - min(l), 'sum': np.sum(l)})

            self._repaint()

    def paintEvent(self, event):
        if self.dataHasChanged:
            self.dataHasChanged = False
            data = retrieve_image(self.images, channel=self.channel, number_of_channels=self.n_channels,
                                  zstack=self.zstack, number_of_zstacks=self.n_zstack, frame=0)

            self.dwidth, self.dheight = data.shape
            self._image = data

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

        # get nuclei boundary as a polygon
        rng_thick = 3
        rng_thick *= self.pix_per_um
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

        painter.setPen(QPen(QBrush(QColor('blue')), 5))
        for me in self.measurement:
            pts = [Qt.QPoint(x, y) for x, y in [me['ls0'], me['ls1']]]
            painter.drawLine(pts[0], pts[1])

        painter.end()
