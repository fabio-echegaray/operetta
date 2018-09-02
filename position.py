import sys

import matplotlib
import seaborn as sns
import skimage
import skimage.draw as draw
import skimage.exposure as exposure
from PyQt4 import QtCore, uic
from PyQt4.QtCore import *
from PyQt4.QtGui import *
from matplotlib.ticker import EngFormatter
from skimage.io import imread

from measurements import *

matplotlib.use('TkAgg')

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('hhlab')

color_red = (65535, 0, 0)
color_green = (0, 65535, 0)
color_blue = (0, 0, 65535)
color_yellow = (65535, 65535, 0)


def qimage_crop(image, qlabel, r, c):
    maxw, maxh, chan = image.shape
    minr = int(r - qlabel.height() / 2.0)
    maxr = int(r + qlabel.height() / 2.0)
    minc = int(c - qlabel.width() / 2.0)
    maxc = int(c + qlabel.width() / 2.0)
    minr = minr if minr > 0 else 0
    minc = minc if minc > 0 else 0
    maxr = maxr if maxr < maxw else maxw
    maxc = maxc if maxc < maxh else maxh
    logger.debug((minr, maxr, minc, maxc))

    data = skimage.img_as_ubyte(image[minr:maxr, minc:maxc].copy())
    # img = ((data - data.min()) / (data.ptp() / 255.0)).astype(np.uint8).repeat(3)
    qtimage = QImage(data.flatten(), qlabel.width(), qlabel.height(), QImage.Format_RGB888)

    return qtimage, (minr, maxr, minc, maxc)


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

        self.pericentrin = exposure.equalize_adapthist(self.pericentrin)

        self.hoechst_show = color.gray2rgb(self.hoechst)
        self.edu_show = color.gray2rgb(self.edu)
        self.pericentrin_show = color.gray2rgb(self.pericentrin)
        self.tubulin_show = color.gray2rgb(self.tubulin)

        r = 6  # [um]
        imgseg, radii = nuclei_segmentation(self.hoechst)
        self.nuclei_features = nuclei_features(imgseg, area_thresh=(r * self.resolution) ** 2 * np.pi)
        # self.centrosomes = centrosomes(imgseg, max_sigma=self.resolution * 1.5)

        self.render_cell()

        self.prevButton.pressed.connect(self.on_prev_button)
        self.nextButton.pressed.connect(self.on_next_button)

    def render_cell(self):
        logger.info('render_cell')
        logger.debug('self.current_nuclei_id: %d' % self.current_nuclei_id)

        feat = self.nuclei_features[self.current_nuclei_id]
        contour = feat['contour']
        r = contour[:, 0].astype(np.uint16)
        c = contour[:, 1].astype(np.uint16)
        logger.debug('feat id: %d' % feat['id'])

        rr, cc = draw.polygon(r, c)
        ravg, cavg = r.mean(), c.mean()
        dr, dc = int(ravg - self.imgCell.height() * 0.5), int(cavg - self.imgCell.width() * 0.5)

        # get images for rendering features
        _, (minr, maxr, minc, maxc) = qimage_crop(self.pericentrin_show, self.imgPericentrin, ravg, cavg)
        qimg_edu, _ = qimage_crop(self.edu_show, self.imgEdu, ravg, cavg)
        pcens = skimage.img_as_ubyte(self.pericentrin_show[minr:maxr, minc:maxc].copy())
        pcen = self.pericentrin[minr:maxr, minc:maxc]

        centrs = centrosomes(pcen, max_sigma=self.resolution * 1.5)
        # centrs = [c for c in self.centrosomes if ((minr <= c[1] <= maxr) and ((minc <= c[0] <= maxc)))]

        tdrw = self.hoechst_show.copy()
        tdrw[:, :, 0] = 0
        try:
            draw.set_color(tdrw, (r, c), color_red, alpha=1)
            draw.set_color(pcens, (r - dr, c - dc), color_red, alpha=1)
        except:
            pass
        qimg_hoechst, _ = qimage_crop(tdrw, self.imgCell, ravg, cavg)

        for centr in centrs:
            y, x, r = centr.astype(np.uint16)
            rcir, ccir = draw.circle_perimeter(y, x, 4)
            draw.set_color(pcens, (rcir, ccir), color_yellow, alpha=1)
        ql = self.imgPericentrin
        qimg_peri = QImage(pcens.flatten(), ql.width(), ql.height(), QImage.Format_RGB888)

        self.imgEdu.setPixmap(QPixmap.fromImage(qimg_edu))
        self.imgCell.setPixmap(QPixmap.fromImage(qimg_hoechst))
        self.imgPericentrin.setPixmap(QPixmap.fromImage(qimg_peri))

        self.mplEduHist.clear()
        sns.distplot(self.edu[rr, cc], kde=False, rug=True, ax=self.mplEduHist.canvas.ax)
        self.mplEduHist.canvas.ax.xaxis.set_major_formatter(EngFormatter())
        self.mplEduHist.canvas.ax.set_xlim([0, 2 ** 16])
        self.mplEduHist.canvas.draw()

        self.lblEduMin.setText('min %0.1f' % self.edu[rr, cc].min())
        self.lblEduMax.setText('max %0.1f' % self.edu[rr, cc].max())
        self.lblEduAvg.setText('avg %0.1f' % self.edu[rr, cc].mean())
        self.lblId.setText('id %d' % feat['id'])
        self.update()

    @QtCore.pyqtSlot()
    def on_prev_button(self):
        logger.info('on_prev_button')
        self.current_nuclei_id = (self.current_nuclei_id - 1) % len(self.nuclei_features)
        self.render_cell()

    @QtCore.pyqtSlot()
    def on_next_button(self):
        logger.info('on_next_button')
        self.current_nuclei_id = (self.current_nuclei_id + 1) % len(self.nuclei_features)
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
