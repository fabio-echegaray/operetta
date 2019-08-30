import logging

import numpy as np
import pandas as pd
from PyQt4 import Qt, QtCore, QtGui
from PyQt4.QtGui import QBrush, QColor, QPainter, QPen, QPixmap
from gui._image_loading import find_image, retrieve_image

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

        self.images = None
        self.pix_per_um = None
        self.dt = None
        self.n_frames = None
        self.n_channels = None
        self.n_zstack = None
        self._zstack = 0
        self._channel = 0

        self.clear()

    @property
    def zstack(self):
        return self._zstack

    @zstack.setter
    def zstack(self, value):
        if value is not None:
            self._zstack = int(value)
            self.dataHasChanged = True
            self.repaint()

    @property
    def channel(self):
        return self._channel

    @channel.setter
    def channel(self, value):
        if value is not None:
            self._channel = int(value)
            self.dataHasChanged = True
            self.repaint()

    @property
    def file(self):
        return self._file

    @file.setter
    def file(self, file):
        if file is not None:
            logger.info('Loading %s' % file)
            self._file = file
            self.images, self.pix_per_um, self.dt, self.n_frames, self.n_channels, zstack = find_image(file)
            self.n_zstack = int(len(self.images) / self.n_frames / self.n_channels)
            self.dataHasChanged = True
            self.repaint()

    def clear(self):
        imgarr = np.zeros(shape=(512, 512), dtype=np.uint32)
        qtimage = QtGui.QImage(imgarr.data, imgarr.shape[1], imgarr.shape[0], imgarr.strides[0],
                               QtGui.QImage.Format_RGB32)
        self.image_pixmap = QtGui.QPixmap(qtimage)
        self.setPixmap(self.image_pixmap)

    def mouseReleaseEvent(self, ev):
        pos = ev.pos()
        # convert to image pixel coords
        x = pos.x() * self.dwidth / self.width()
        y = pos.y() * self.dheight / self.height()
        logger.info('clicked! X%d Y%d' % (x, y))
        self.emit(QtCore.SIGNAL('clicked()'))

    def paintEvent(self, event):
        if self.dataHasChanged:
            self.dataHasChanged = False
            data = retrieve_image(self.images, channel=self.channel, number_of_channels=self.n_channels,
                                  zstack=self.zstack, number_of_zstacks=self.n_zstack,
                                  frame=0, number_of_frames=self.n_frames)

            self.dwidth, self.dheight = data.shape
            # map the data range to 0 - 255
            img_8bit = ((data - data.min()) / (data.ptp() / 255.0)).astype(np.uint8)
            qtimage = QtGui.QImage(img_8bit.repeat(4), self.dwidth, self.dheight, QtGui.QImage.Format_RGB32)
            self.image_pixmap = QPixmap(qtimage)
            # self.draw_measurements()
            self.setPixmap(self.image_pixmap)
        return QtGui.QLabel.paintEvent(self, event)
