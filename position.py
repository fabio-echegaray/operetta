import sys

import pandas as pd
import seaborn as sns
import skimage.draw as draw
from PIL import Image
from PIL.ImageQt import ImageQt
from PyQt4 import QtCore, uic
from PyQt4.Qt import QMutex
from PyQt4.QtCore import *
from PyQt4.QtCore import QThread
from PyQt4.QtGui import QApplication, QPixmap, QWidget
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure, SubplotParams
from matplotlib.ticker import EngFormatter
from skimage.io import imread

from measurements import *

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('hhlab')

color_red = (65535, 0, 0)
color_green = (0, 65535, 0)
color_blue = (0, 0, 65535)
color_yellow = (65535, 65535, 0)
sp = SubplotParams(left=0., bottom=0., right=1., top=1.)
mydpi = 72


def get_crop_bbox(image, qlabel, r, c):
    maxw, maxh = image.shape
    minr = int(r - qlabel.height() / 2.0)
    maxr = int(r + qlabel.height() / 2.0)
    minc = int(c - qlabel.width() / 2.0)
    maxc = int(c + qlabel.width() / 2.0)
    minr = minr if minr > 0 else 0
    minc = minc if minc > 0 else 0
    maxr = maxr if maxr < maxw else maxw
    maxc = maxc if maxc < maxh else maxh
    if maxr - minr < qlabel.height():
        if minr == 0:
            maxr = qlabel.height()
        else:
            minr = maxr - qlabel.height()
    if maxc - minc < qlabel.width():
        if minc == 0:
            maxc = qlabel.width()
        else:
            minc = maxc - qlabel.width()

    return minr, maxr, minc, maxc


def canvas_to_pil(canvas):
    canvas.draw()
    s = canvas.tostring_rgb()
    w, h = canvas.get_width_height()[::-1]
    im = Image.frombytes("RGB", (w, h), s)
    return im


class renderImagesThread(QThread):
    def __init__(self, hoechst, edu, pericentrin, tubulin, cell_list, nucleus_list, centrosome_list, largeQLabel,
                 smallQLabel):
        """
        Make a new thread instance with images to render and features
        as parameters.

        :param subreddits: A list of subreddit names
        :type subreddits: list
        """
        QThread.__init__(self)

        self.hoechst = hoechst
        self.edu = edu
        self.pericentrin = pericentrin
        self.tubulin = tubulin
        self.cells = cell_list
        self.nuclei = nucleus_list
        self.centrosomes = centrosome_list
        self.id = -1
        self.nucleus = None
        self.cell = None
        self.centrosome = None
        self.lqlbl = largeQLabel
        self.sqlbl = smallQLabel

    def __del__(self):
        self.wait()

    def setIndividual(self, id, nuc, cll, cntr):
        """
        Sets the individual for feature plotting.

        :param nuc: A shapely polygon for the nucleus boundary
        :type nuc: Polygon
        :param cll: A shapely polygon for the cell boundary
        :type cll: Polygon
        :param cll: A list of shapely points for centrosomes
        :type cll: list
        """
        self.id = id
        self.nucleus = nuc
        self.cell = cll
        self.centrosome = cntr

    def run(self):
        """
        Go over every item in the
        """

        cen = self.nucleus.centroid

        # create a matplotlib axis and plot edu image
        fig = Figure((self.sqlbl.width() / mydpi, self.sqlbl.height() / mydpi), subplotpars=sp, dpi=mydpi)
        canvas = FigureCanvas(fig)
        ax = fig.gca()
        ax.set_aspect('equal')
        ax.set_axis_off()
        l, b, w, h = fig.bbox.bounds

        ax.plot(cen.x, cen.y, color='red', marker='+', linewidth=1, solid_capstyle='round', zorder=2)

        x, y = self.nucleus.exterior.xy
        ax.plot(x, y, color='red', linewidth=1, solid_capstyle='round', zorder=2)

        ax.set_xlim(cen.x - w / 2, cen.x + w / 2)
        ax.set_ylim(cen.y - h / 2, cen.y + h / 2)
        # plot edu + boundaries
        ax.imshow(self.edu, cmap='gray')

        self.qimg_edu = ImageQt(canvas_to_pil(canvas))

        # plot rest of features
        fig = Figure((self.lqlbl.width() / mydpi, self.lqlbl.height() / mydpi), subplotpars=sp, dpi=mydpi)
        canvas = FigureCanvas(fig)
        ax = fig.gca()
        ax.set_aspect('equal')
        ax.set_axis_off()
        l, b, w, h = fig.bbox.bounds

        x, y = self.nucleus.exterior.xy
        ax.plot(x, y, color='red', linewidth=1, solid_capstyle='round', zorder=2)
        ax.plot(cen.x, cen.y, color='red', marker='+', linewidth=1, solid_capstyle='round', zorder=2)

        ax.set_xlim(cen.x - w / 2, cen.x + w / 2)
        ax.set_ylim(cen.y - h / 2, cen.y + h / 2)
        # plot edu + boundaries
        ax.imshow(self.edu, cmap='gray')

        if self.cell is not None:
            x, y = self.cell.exterior.xy
            ax.plot(x, y, color='green', linewidth=1, solid_capstyle='round', zorder=1)

        if self.centrosome is not None:
            for cn in self.centrosome:
                c = plt.Circle((cn.x, cn.y), facecolor='none', edgecolor='yellow', linewidth=1, zorder=5)
                ax.add_artist(c)

        ax.imshow(self.hoechst, cmap='gray')
        self.qimg_hoechst = ImageQt(canvas_to_pil(canvas))

        ax.imshow(self.pericentrin, cmap='gray')
        self.qimg_peri = ImageQt(canvas_to_pil(canvas))

        ax.imshow(self.tubulin, cmap='gray')
        self.qimg_tubulin = ImageQt(canvas_to_pil(canvas))


class ExplorationGui(QWidget):
    def __init__(self, hfname, efname, pfname, tfname):
        QWidget.__init__(self)
        uic.loadUi('gui_selection.ui', self)
        for l in [self.lblEduMin, self.lblEduMax, self.lblEduAvg, self.lblTubMin, self.lblTubMax, self.lblTubAvg]:
            l.setStyleSheet('color: grey')

        self.current_nuclei_id = 0
        self.centrosome_dropped = False
        self.resolution = 4.5

        logger.info('reading image files')
        self.hoechst = imread(hfname)
        self.edu = imread(efname)
        self.pericentrin = imread(pfname)
        self.tubulin = imread(tfname)

        self.pericentrin = exposure.equalize_adapthist(self.pericentrin, clip_limit=0.03)

        filename = 'out/nuclei.npy'
        if os.path.exists(filename):
            logger.info('reading nuclei file data')
            self.nuclei_features = np.load(filename)
        else:
            logger.info('applying nuclei algorithm')
            r = 6  # [um]
            imgseg, radii = nuclei_segmentation(self.hoechst)
            self.nuclei_features = nuclei_features(imgseg, area_thresh=(r * self.resolution) ** 2 * np.pi)
            np.save(filename, self.nuclei_features)

        filename = 'out/centrosomes.npy'
        if os.path.exists(filename):
            logger.info('reading centrosome file data')
            self.centrosomes = np.load(filename)
        else:
            logger.info('applying centrosome algorithm')
            self.centrosomes = centrosomes(self.pericentrin, max_sigma=self.resolution * 0.2)
            np.save(filename, self.centrosomes)

        filename = 'out/cells.npy'
        if os.path.exists(filename):
            logger.info('reading cell boundary file data')
            self.cells = np.load(filename)
        else:
            logger.info('applying cell boundary algorithm')
            self.cells, _ = cell_boundary(self.tubulin, self.hoechst)
            np.save(filename, self.cells)

        self.prevButton.pressed.connect(self.on_prev_button)
        self.nextButton.pressed.connect(self.on_next_button)
        self.plotButton.pressed.connect(self.plot_everything_debug)

        self.renderingThread = renderImagesThread(self.hoechst, self.edu, self.pericentrin, self.tubulin, self.cells,
                                                  self.nuclei_features, self.centrosomes, self.imgCell, self.imgEdu)
        self.renderingThread.finished.connect(self.render_images)
        self.renderingThread.terminated.connect(self.trigger_render)
        self.renderingMutex = QMutex()

        self.build_df()
        self.render_cell()

    def build_df(self):
        self.df = pd.DataFrame()
        for nuc in self.nuclei_features:
            valid, cell, nuclei, cntrsmes = get_nuclei_features(self.hoechst, nuc['boundary'], self.cells,
                                                                self.nuclei_features, self.centrosomes)
            if valid:
                twocntr = len(cntrsmes) == 2
                c1 = cntrsmes[0]
                c2 = cntrsmes[1] if twocntr else None
                nuc_bnd = nuc['boundary'].astype(np.uint16)
                c, r = nuc_bnd[:, 0], nuc_bnd[:, 1]
                r_n, c_n = draw.polygon(r, c)

                d = pd.DataFrame(data={'id_n': [nuc['id']],
                                       'dna': [self.hoechst[r_n, c_n].mean() * nuclei.area],
                                       'edu_avg': [self.edu[r_n, c_n].mean()],
                                       'edu_max': [self.edu[r_n, c_n].max()],
                                       'centrosomes': [len(cntrsmes)],
                                       'c1_d_nuc_centr': [nuclei.centroid.distance(c1)],
                                       'c2_d_nuc_centr': [nuclei.centroid.distance(c2) if twocntr else np.nan],
                                       'c1_d_nuc_bound': [nuclei.exterior.distance(c1)],
                                       'c2_d_nuc_bound': [nuclei.exterior.distance(c2) if twocntr else np.nan],
                                       'c1_d_cell_centr': [cell.centroid.distance(c1)],
                                       'c2_d_cell_centr': [cell.centroid.distance(c2) if twocntr else np.nan],
                                       'c1_d_cell_bound': [cell.exterior.distance(c1)],
                                       'c2_d_cell_bound': [cell.exterior.distance(c2) if twocntr else np.nan],
                                       })
                self.df = self.df.append(d, ignore_index=True)

    def render_cell(self):
        logger.info('render_cell')
        logger.debug('self.current_nuclei_id: %d' % self.current_nuclei_id)

        nuc_f = self.nuclei_features[self.current_nuclei_id]
        nuc_bnd = nuc_f['boundary']

        valid, cell, nucleus, cntrsmes = get_nuclei_features(self.hoechst, nuc_bnd, self.cells, self.nuclei_features,
                                                             self.centrosomes)

        self.renderingMutex.lock()
        if self.renderingThread.isRunning():
            logger.debug('terminating rendering thread')
            self.renderingThread.quit()
        else:
            self.renderingThread.setIndividual(self.current_nuclei_id, nucleus, cell, cntrsmes)
            self.trigger_render()
        self.renderingMutex.unlock()

        c, r = nuc_bnd[:, 0], nuc_bnd[:, 1]
        r_n, c_n = draw.polygon(r, c)
        logger.debug('feat id: %d' % nuc_f['id'])

        self.mplEduHist.hide()
        self.imgEdu.clear()
        self.imgCell.clear()
        self.imgPericentrin.clear()
        self.imgTubulin.clear()

        self.lblEduMin.setText('min ' + eng_string(self.edu[r_n, c_n].min(), format='%0.1f', si=True))
        self.lblEduMax.setText('max ' + eng_string(self.edu[r_n, c_n].max(), format='%0.1f', si=True))
        self.lblEduAvg.setText('avg ' + eng_string(self.edu[r_n, c_n].mean(), format='%0.1f', si=True))
        self.lblTubMin.setText('min ' + eng_string(self.tubulin[r_n, c_n].min(), format='%0.1f', si=True))
        self.lblTubMax.setText('max ' + eng_string(self.tubulin[r_n, c_n].max(), format='%0.1f', si=True))
        self.lblTubAvg.setText('avg ' + eng_string(self.tubulin[r_n, c_n].mean(), format='%0.1f', si=True))
        self.lblCentr_n.setText('%d' % (len(cntrsmes) if cntrsmes is not None else 0))
        self.lblId.setText('id %d' % nuc_f['id'])
        self.lblOK.setText('OK' if valid else 'no OK')

        fig = Figure((self.imgCentrCloseup.width() / mydpi, self.imgCentrCloseup.height() / mydpi), subplotpars=sp,
                     dpi=mydpi)
        canvas = FigureCanvas(fig)
        ax = fig.gca()
        ax.set_aspect('equal')
        ax.set_axis_off()
        l, b, w, h = fig.bbox.bounds
        if valid:
            c1 = cntrsmes[0]
            self.lblDist_nc.setText('%0.2f' % nucleus.centroid.distance(c1))
            self.lblDist_nb.setText('%0.2f' % nucleus.exterior.distance(c1))
            self.lblDist_cc.setText('%0.2f' % cell.centroid.distance(c1))
            self.lblDist_cb.setText('%0.2f' % cell.exterior.distance(c1))

            ax.imshow(self.pericentrin, cmap='gray')
            for cn in cntrsmes:
                c = plt.Circle((cn.x, cn.y), radius=10, facecolor='none', edgecolor='yellow',
                               linestyle='--', linewidth=1, zorder=5)
                ax.add_artist(c)

            c1 = cntrsmes[0]
            ax.set_xlim(c1.x - w / 8, c1.x + w / 8)
            ax.set_ylim(c1.y - h / 8, c1.y + h / 8)

            qimg_closeup = ImageQt(canvas_to_pil(canvas))
            self.imgCentrCloseup.setPixmap(QPixmap.fromImage(qimg_closeup))
        else:
            self.lblDist_nc.setText('-')
            self.lblDist_nb.setText('-')
            self.lblDist_cc.setText('-')
            self.lblDist_cb.setText('-')

            qimg_closeup = ImageQt(Image.new('RGB', (int(w), int(h))))
            self.imgCentrCloseup.setPixmap(QPixmap.fromImage(qimg_closeup))
        self.update()

    @QtCore.pyqtSlot()
    def trigger_render(self):
        self.renderingThread.start()

    @QtCore.pyqtSlot()
    def render_images(self):
        if self.current_nuclei_id != self.renderingThread.id:
            logger.info('rendering thread finished but for a different id! self %d thread %d' % (
                self.current_nuclei_id, self.renderingThread.id))

            self.render_cell()
            return

        if self.renderingMutex.tryLock():
            self.imgEdu.setPixmap(QPixmap.fromImage(self.renderingThread.qimg_edu))
            self.imgCell.setPixmap(QPixmap.fromImage(self.renderingThread.qimg_hoechst))
            self.imgPericentrin.setPixmap(QPixmap.fromImage(self.renderingThread.qimg_peri))
            self.imgTubulin.setPixmap(QPixmap.fromImage(self.renderingThread.qimg_tubulin))

            nuc_f = self.nuclei_features[self.current_nuclei_id]
            nuc_bnd = nuc_f['boundary']
            c, r = nuc_bnd[:, 0], nuc_bnd[:, 1]
            r_n, c_n = draw.polygon(r, c)

            self.mplEduHist.clear()
            sns.distplot(self.edu[r_n, c_n], kde=False, rug=True, ax=self.mplEduHist.canvas.ax)
            self.mplEduHist.canvas.ax.xaxis.set_major_formatter(EngFormatter())
            self.mplEduHist.canvas.ax.set_xlim([0, 2 ** 16])
            self.mplEduHist.canvas.draw()
            self.mplEduHist.show()

            self.renderingMutex.unlock()

    @QtCore.pyqtSlot()
    def plot_everything_debug(self):
        f = plt.figure(20)
        ax = f.gca()
        ax.set_aspect('equal')
        ax.imshow(self.hoechst, origin='lower')
        for nuc_feat in self.nuclei_features:
            nuc_bnd = nuc_feat['boundary']
            nuc = Polygon(nuc_bnd)
            x, y = nuc.exterior.xy
            ax.plot(x, y, color='red', linewidth=3, solid_capstyle='round', zorder=2)
        for cll in self.cells:
            cel = Polygon(cll['boundary'])
            x, y = cel.exterior.xy
            ax.plot(x, y, color='green', linewidth=3, solid_capstyle='round', zorder=1)
        for cn in self.centrosomes:
            c = plt.Circle((cn[0], cn[1]), facecolor='none', edgecolor='yellow', linewidth=3, zorder=5)
            ax.add_artist(c)

        f = plt.figure(30)
        ax1 = f.add_subplot(211)
        ax2 = f.add_subplot(212)
        df = self.df
        df = df.loc[:, ['edu_avg', 'c1_d_nuc_centr', 'c1_d_nuc_bound', 'c1_d_cell_centr', 'c1_d_cell_bound']]
        dd = pd.melt(df, id_vars=['edu_avg'])
        sns.scatterplot(x='edu_avg', y='value', hue='variable', data=dd, ax=ax1)
        sns.boxplot(x='variable', y='value', data=dd, ax=ax2)
        sns.swarmplot(x='variable', y='value', data=dd, ax=ax2, color=".25")
        ax1.xaxis.set_major_formatter(EngFormatter())
        ax1.set_ylabel('distance [um]')
        ax2.set_ylabel('distance [um]')
        # ax1.set_title('')

        f = plt.figure(31)
        ax1 = f.add_subplot(111)
        ax1.xaxis.set_major_formatter(EngFormatter())
        ax1.yaxis.set_major_formatter(EngFormatter())
        ax1.set_ylabel('distance [um]')

        df = self.df
        df = df.loc[:, ['dna', 'c1_d_nuc_centr', 'c1_d_nuc_bound', 'c1_d_cell_centr', 'c1_d_cell_bound']]
        df = df.rename(columns={'c1_d_nuc_centr': 'nuclear centroid',
                                'c1_d_nuc_bound': 'nuclear boundary',
                                'c1_d_cell_centr': 'cell centroid',
                                'c1_d_cell_bound': 'cell boundary',})
        dd = pd.melt(df, id_vars=['dna'])
        dd = dd.rename(columns={'value': 'distance'})

        g = sns.FacetGrid(dd, col="variable", col_wrap=2)
        # g = g.map(plt.scatter, "dna", "distance", edgecolor="w")
        g = g.map(sns.regplot, "dna", "distance")
        g.add_legend()

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
    base_path = os.path.abspath('%s' % os.getcwd())
    logging.info('Qt version:' + QT_VERSION_STR)
    logging.info('PyQt version:' + PYQT_VERSION_STR)
    logging.info('Working dir:' + os.getcwd())
    logging.info('Base dir:' + base_path)
    os.chdir(base_path)

    app = QApplication(sys.argv)

    b_path = '/Users/Fabio/data/20180817 U2OS cenpf peric edu formfix/'
    gui = ExplorationGui(
        hfname=b_path + 'Capture 3 - Position 0 [64] Montage.Project Maximum Z_XY1534504412_Z0_T0_C0.tif',
        tfname=b_path + 'Capture 3 - Position 0 [64] Montage.Project Maximum Z_XY1534504412_Z0_T0_C1.tif',
        pfname=b_path + 'Capture 3 - Position 0 [64] Montage.Project Maximum Z_XY1534504412_Z0_T0_C2.tif',
        efname=b_path + 'Capture 3 - Position 0 [64] Montage.Project Maximum Z_XY1534504412_Z0_T0_C3.tif'
    )

    from pycallgraph import PyCallGraph
    from pycallgraph.output import GraphvizOutput

    with PyCallGraph(output=GraphvizOutput()):
        gui.show()
        code = app.exec_()
    sys.exit(code)
