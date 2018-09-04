import sys

import matplotlib
import seaborn as sns
import skimage
import skimage.draw as draw
from PyQt4 import QtCore, uic
from PyQt4.QtCore import *
from PyQt4.QtGui import QApplication, QImage, QPixmap, QWidget
from matplotlib.ticker import EngFormatter
from skimage.io import imread

from measurements import *

matplotlib.use('Agg')

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

        self.pericentrin = exposure.equalize_adapthist(self.pericentrin, clip_limit=0.02)

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

        ravg, cavg = r.mean(), c.mean()
        dr, dc = int(ravg - self.imgCell.height() * 0.5), int(cavg - self.imgCell.width() * 0.5)
        if dr < 0 or dc < 0:
            logger.warning('nucleus in the edge of the frame')
            # return

        r_n, c_n = draw.polygon(r, c)
        # get images for rendering features
        _, (minr, maxr, minc, maxc) = qimage_crop(self.pericentrin_show, self.imgPericentrin, ravg, cavg)
        qimg_edu, _ = qimage_crop(self.edu_show, self.imgEdu, ravg, cavg)
        # qimg_tubulin, _ = qimage_crop(self.tubulin_show, self.imgTubulin, ravg, cavg)
        peric_render = skimage.img_as_ubyte(self.pericentrin_show[minr:maxr, minc:maxc].copy())
        pcen = self.pericentrin[minr:maxr, minc:maxc]
        ptub = self.tubulin[minr:maxr, minc:maxc]
        phoec = self.hoechst[minr:maxr, minc:maxc]

        tubulin_render = skimage.img_as_ubyte(self.tubulin_show[minr:maxr, minc:maxc].copy())
        # tubulin_render[:, :, 0] = 0
        # tubulin_render[:, :, 2] = 0

        hoechst_render = self.hoechst_show.copy()
        hoechst_render[:, :, 0] = 0

        centrs = centrosomes(pcen, max_sigma=self.resolution * 1.5)
        # centrs = [c for c in self.centrosomes if ((minr <= c[1] <= maxr) and ((minc <= c[0] <= maxc)))]
        cell_bounds, gabor = cell_boundary(ptub, phoec)
        logger.debug('cell_bounds ' + str(cell_bounds))

        try:
            draw.set_color(hoechst_render, (r, c), color_red, alpha=1)
            draw.set_color(peric_render, (r - dr, c - dc), color_red, alpha=1)
            draw.set_color(tubulin_render, (r - dr, c - dc), color_red, alpha=1)
        except:
            pass

        for centr in centrs:
            y, x, r = centr.astype(np.uint16)
            rcir, ccir = draw.circle_perimeter(y, x, 4)
            draw.set_color(peric_render, (rcir, ccir), color_yellow, alpha=1)
            draw.set_color(tubulin_render, (rcir, ccir), color_yellow, alpha=1)


        for b in cell_bounds:
            bnd = b['boundary'].astype(np.uint16)
            r_c, c_c = draw.polygon_perimeter(bnd[:, 1], bnd[:, 0])
            draw.set_color(peric_render, (r_c, c_c), color_green, alpha=1)
            draw.set_color(tubulin_render, (r_c, c_c), color_green, alpha=1)

        ql = self.imgPericentrin
        qimg_hoechst, _ = qimage_crop(hoechst_render, self.imgCell, ravg, cavg)
        qimg_peri = QImage(peric_render.flatten(), ql.width(), ql.height(), QImage.Format_RGB888)
        qimg_tubulin = QImage(tubulin_render.flatten(), ql.width(), ql.height(), QImage.Format_RGB888)
        self.imgEdu.setPixmap(QPixmap.fromImage(qimg_edu))
        self.imgCell.setPixmap(QPixmap.fromImage(qimg_hoechst))
        self.imgPericentrin.setPixmap(QPixmap.fromImage(qimg_peri))
        self.imgTubulin.setPixmap(QPixmap.fromImage(qimg_tubulin))

        self.mplEduHist.clear()
        sns.distplot(self.edu[r_n, c_n], kde=False, rug=True, ax=self.mplEduHist.canvas.ax)
        self.mplEduHist.canvas.ax.xaxis.set_major_formatter(EngFormatter())
        self.mplEduHist.canvas.ax.set_xlim([0, 2 ** 16])
        self.mplEduHist.canvas.draw()

        self.lblEduMin.setText('min ' + eng_string(self.edu[r_n, c_n].min(), format='%0.1f', si=True))
        self.lblEduMax.setText('max ' + eng_string(self.edu[r_n, c_n].max(), format='%0.1f', si=True))
        self.lblEduAvg.setText('avg ' + eng_string(self.edu[r_n, c_n].mean(), format='%0.1f', si=True))
        self.lblTubMin.setText('min ' + eng_string(self.tubulin[r_n, c_n].min(), format='%0.1f', si=True))
        self.lblTubMax.setText('max ' + eng_string(self.tubulin[r_n, c_n].max(), format='%0.1f', si=True))
        self.lblTubAvg.setText('avg ' + eng_string(self.tubulin[r_n, c_n].mean(), format='%0.1f', si=True))
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
