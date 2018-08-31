import logging
# from PyQt5 import QtCore, QtGui, QtWidgets, uic
# from PyQt5.QtCore import QT_VERSION_STR
# from PyQt5.Qt import PYQT_VERSION_STR
# from PyQt5.QtGui import QBrush, QColor, QPainter, QPen, QPixmap
from PyQt4 import uic, QtCore
from PyQt4.QtCore import *
from PyQt4.QtGui import *


import os
import numpy as np
import os
import re
import sys

# import matplotlib
# import matplotlib.gridspec
import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches
# from matplotlib.path import Path
# matplotlib.rcParams['backend'] = 'Qt4Agg'
# matplotlib.rcParams['backend.qt4'] = 'PyQt4'



import numpy as np
# import pandas as pd
import seaborn as sns
from math import sqrt
# import cv2
import skimage.feature as feature
from skimage.io import imread
from scipy import ndimage as ndi
import skimage.morphology
import skimage.segmentation
from skimage import exposure
from skimage.filters import gaussian
from skimage.morphology import erosion, square
from skimage.segmentation import active_contour
import skimage.filters as filters
import skimage.segmentation as segmentation
import skimage.measure as measure
import skimage.morphology as morphology
import skimage.exposure as exposure
import skimage.draw as draw
import skimage.color as color

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('hhlab')


def nuclei_segmentation(image, ax=None):
    image_gray = color.rgb2gray(image)

    filename = 'out/blobs.npy'
    if os.path.exists(filename):
        logger.info('reading blob file data')
        blobs_log = np.load(filename)
    else:
        logger.info('applying blobs algorithm')
        blobs_log = feature.blob_log(image_gray, max_sigma=30, num_sigma=10, threshold=.1)
        # Compute radii in the 3rd column.
        blobs_log[:, 2] = blobs_log[:, 2] * sqrt(2)

        np.save(filename, blobs_log)


    # apply threshold
    thresh_otsu = filters.threshold_otsu(image)
    thresh = image >= thresh_otsu

    # thresh = filters.gaussian(thresh,sigma=2.0)
    # block_size = 21
    # thresh = filters.threshold_local(image,block_size)
    thresh = morphology.remove_small_holes(thresh)
    thresh = morphology.remove_small_objects(thresh)
    # thresh = morphology.closing(image > thresh, morphology.square(3))

    # remove artifacts connected to image border
    cleared = segmentation.clear_border(thresh)
    # ax.imshow(cleared, interpolation='nearest')

    radii_valid = blobs_log[:, 2] > 20
    if ax is not None:
        ax.imshow(cleared)
        radii = blobs_log[radii_valid]
        excluded = blobs_log[~radii_valid]
        for blob in radii:
            y, x, r = blob
            c = plt.Circle((x, y), r, color='lime', linewidth=1, fill=False)
            ax.add_patch(c)
        for blob in excluded:
            y, x, r = blob
            c = plt.Circle((x, y), r, color='red', linewidth=0.2, fill=False)
            ax.add_patch(c)

        ax.set_title('hoechst blobs')
        ax.set_axis_off()

    return cleared, radii_valid


def nuclei_features(image, ax=None, area_thresh=100):

    def polygon_area(x, y):
        return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

    # label image regions
    # label_image = measure.label(image)
    # image_label_overlay = color.label2rgb(label_image, image=image)

    # Display the image and plot all contours found
    contours = measure.find_contours(image, 0.9)
    # for n, contour in enumerate(contours):
    #     ax.plot(contour[:, 1], contour[:, 0], linewidth=1)


    _list = list()
    for k, contr in enumerate(contours):
        if polygon_area(contr[:,0],contr[:,1])>area_thresh:
            _list.append({
                'id': k,
                # 'properties': region,
                'contour': contr
            })
            if ax is not None:
                ax.plot(contr[:, 1], contr[:, 0], linewidth=1)

    # boundaries_list = list()
    # for k, region in enumerate(measure.regionprops(label_image)):
    #     # take regions with large enough areas
    #     if region.area >= area_thresh:
    #         # draw rectangle around segmented coins
    #         minr, minc, maxr, maxc = region.bbox
    #         rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
    #                                   fill=False, edgecolor='red', linewidth=1)
    #         ax.add_patch(rect)
    #
    #         # rect = image[minr:maxr, minc:maxc]
    #         # contours = measure.find_contours(rect, 0.9)
    #         # for contour in contours:
    #         #     ax.plot(contour[:, 1], contour[:, 0], linewidth=1)
    #         #
    #         for contr in contours:
    #             mask = np.logical_and(
    #                     np.logical_and(minr<=contr[:,0], contr[:,0]<=maxr),
    #                     np.logical_and(minc<=contr[:,1], contr[:,1]<=maxc)
    #             )
    #             if mask.any():
    #                 boundaries_list.append({
    #                     'id': k,
    #                     'properties': region,
    #                     'contour': contr
    #                 })
    #                 ax.plot(contr[:, 1], contr[:, 0], linewidth=1)


    return _list


class ExplorationGui(QWidget):
    def __init__(self, hfname, efname, pfname, tfname):
        QWidget.__init__(self)
        uic.loadUi('gui_selection.ui', self)

        self.current_nuclei_id = 0
        self.centrosome_dropped = False
        self.resolution = 4.5

        logger.info('reading image files')
        self.hoechst = imread(hfname)
        self.edu = imread(efname)
        self.pericentrin = imread(pfname)
        self.tubulin = imread(tfname)

        # fig = plt.figure(figsize=(10, 5), dpi=100)
        # fig.clf()
        # ax1 = plt.subplot(1, 2, 1)
        # ax2 = plt.subplot(1, 2, 2, sharex=ax1)
        # ax2.axis('equal')
        # fig.tight_layout()

        ax1=ax2=None
        imgseg, radii = nuclei_segmentation(self.hoechst, ax=ax1)
        r=6 #[um]
        self.features = nuclei_features(imgseg,ax=ax2, area_thresh=(r*self.resolution)**2*np.pi)
        if ax1 is not None:
            plt.show(block=False)

        # self.imgCell.clear()
        # self.imgEdu.clear()

        self.render_cell()

        # self.prevButton.pressed.connect( self.clear)
        self.prevButton.pressed.connect( self.on_prev_button)
        self.nextButton.pressed.connect( self.on_next_button)

        logger.debug(self.updatesEnabled())
        logger.debug(self.imgCell.updatesEnabled())

    def clear(self):
        pixmap = QPixmap('test.png')
        self.imgCell.setPixmap(pixmap)
        self.imgEdu.setPixmap(pixmap)
        self.mplEduHist.clear()
        self.lblId.setText("holi")
        logger.debug(self.imgCell.width())
        logger.debug(self.imgCell.height())
        QApplication.processEvents()

    def render_cell(self):
        logger.info('render_cell')
        logger.debug('self.current_nuclei_id: %d'%self.current_nuclei_id)

        feat = self.features[self.current_nuclei_id]
        contour=feat['contour']
        r=contour[:,0]
        c=contour[:,1]
        logger.debug('feat id: %d'%feat['id'])

        rr, cc = draw.polygon(r, c)


        maxw, maxh =self.hoechst.shape
        minr = int(r.mean()-self.imgCell.width()/2.0)
        maxr = int(r.mean()+self.imgCell.width()/2.0)
        minc = int(c.mean()-self.imgCell.height()/2.0)
        maxc = int(c.mean()+self.imgCell.height()/2.0)
        minr = minr if minr>0 else 0
        minc = minc if minc>0 else 0
        maxr = maxr if maxr<maxw else maxw
        maxc = maxc if maxc<maxh else maxh
        data=self.hoechst[minr:maxr, minc:maxc].copy()
        img=((data - data.min()) / (data.ptp() / 255.0)).astype(np.uint8).repeat(4)
        qtimage = QImage(img.data, self.imgCell.width(), self.imgCell.height(), QImage.Format_RGB32)

        image_pixmap = QPixmap(qtimage)

        painter = QPainter()
        painter.begin(image_pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(QPen(QBrush(QColor('red')), 2))
        nucb_qpoints = [QPoint(x * self.resolution, y * self.resolution) for x, y in contour]
        nucb_poly = QPolygon(nucb_qpoints)
        painter.drawPolygon(nucb_poly)
        painter.end()

        # image_pixmap = QPixmap('test.png')
        self.imgCell.setPixmap(image_pixmap)
        # self.imgCell.setPixmap(QPixmap.fromImage(qtimage))

        minr = int(r.mean()-self.imgEdu.width()/2.0)
        maxr = int(r.mean()+self.imgEdu.width()/2.0)
        minc = int(c.mean()-self.imgEdu.height()/2.0)
        maxc = int(c.mean()+self.imgEdu.height()/2.0)
        minr = minr if minr>0 else 0
        minc = minc if minc>0 else 0
        maxr = maxr if maxr<maxw else maxw
        maxc = maxc if maxc<maxh else maxh
        data=self.edu[minr:maxr, minc:maxc].copy()
        img=((data - data.min()) / (data.ptp() / 255.0)).astype(np.uint8).repeat(4)
        qtimage = QImage(img.data, self.imgEdu.width(), self.imgEdu.height(), QImage.Format_RGB32)
        image_pixmap = QPixmap(qtimage)
        self.imgEdu.setPixmap(image_pixmap)

        # self.hist_edu=exposure.histogram(self.edu[rr, cc])
        # logger.debug(self.hist_edu)

        self.mplEduHist.clear()
        if feat['id']%2==0: sns.distplot(self.edu[rr, cc], kde=False, rug=True, ax=self.mplEduHist.canvas.ax)

        self.lblId.setText('%d'%feat['id'])
        self.update()


    @QtCore.pyqtSlot()
    def on_prev_button(self):
        logger.info('on_prev_button')
        self.current_nuclei_id = (self.current_nuclei_id-1)%len(self.features)
        self.render_cell()


    @QtCore.pyqtSlot()
    def on_next_button(self):
        logger.info('on_next_button')
        self.current_nuclei_id = (self.current_nuclei_id+1)%len(self.features)
        self.render_cell()



if __name__ == '__main__':

    b_path = '/Users/Fabio/data/20180817 U2OS cenpf peric edu formfix/'

    base_path = os.path.abspath('%s' % os.getcwd())
    logging.info('Qt version:' + QT_VERSION_STR)
    logging.info('PyQt version:' + PYQT_VERSION_STR)
    logging.info('Working dir:' + os.getcwd())
    logging.info('Base dir:' + base_path)
    os.chdir(base_path)

    app = QApplication(sys.argv)

    gui = ExplorationGui(
        hfname=b_path + 'Capture 3 - Position 0 [64] Montage.Project Maximum Z_XY1534504412_Z0_T0_C0.tif',
        tfname=b_path + 'Capture 3 - Position 0 [64] Montage.Project Maximum Z_XY1534504412_Z0_T0_C1.tif',
        pfname=b_path + 'Capture 3 - Position 0 [64] Montage.Project Maximum Z_XY1534504412_Z0_T0_C2.tif',
        efname=b_path + 'Capture 3 - Position 0 [64] Montage.Project Maximum Z_XY1534504412_Z0_T0_C3.tif'
    )
    gui.show()
    sys.exit(app.exec_())
