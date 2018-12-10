import logging

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import skimage.draw as draw
from PIL.ImageQt import ImageQt
from PyQt4 import QtCore, uic
from PyQt4.QtCore import *
from PyQt4.QtCore import QThread
from PyQt4.QtGui import QPixmap, QWidget
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.ticker import EngFormatter
from shapely.geometry.polygon import Polygon
from matplotlib.figure import SubplotParams

from . import utils
import measurements as m
import position

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('hhlab')
sp = SubplotParams(left=0., bottom=0., right=1., top=1.)
mydpi = 72


class RenderImagesThread(QThread):
    def __init__(self, hoechst, edu, pericentrin, tubulin,
                 largeQLabel, smallQLabel):
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
        self.id = -1
        self.nucleus = None
        self.cell = None
        self.centrosomes = None
        self.lqlbl = largeQLabel
        self.sqlbl = smallQLabel

    def __del__(self):
        self.wait()

    def setIndividual(self, id, sample):
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
        self.nucleus = sample['nucleus']
        self.cell = sample['cell']
        self.centrosomes = sample['centrosomes']

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

        self.qimg_edu = ImageQt(utils.utils.canvas_to_pil(canvas))

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
        # ax.imshow(self.edu, cmap='gray')

        if self.cell is not None:
            x, y = self.cell.exterior.xy
            ax.plot(x, y, color='green', linewidth=1, solid_capstyle='round', zorder=1)

        if self.centrosomes is not None:
            c1, c2 = self.centrosomes
            if c1 is not None:
                c = plt.Circle((c1.x, c1.y), radius=5, facecolor='none', edgecolor='r',
                               linewidth=1, zorder=5)
                ax.add_artist(c)
            if c2 is not None:
                c = plt.Circle((c2.x, c2.y), radius=5, facecolor='none', edgecolor='b',
                               linewidth=1, zorder=5)
                ax.add_artist(c)

        ax.imshow(self.hoechst, cmap='gray')
        self.qimg_hoechst = ImageQt(utils.utils.canvas_to_pil(canvas))

        ax.imshow(self.pericentrin, cmap='gray')
        self.qimg_peri = ImageQt(utils.canvas_to_pil(canvas))

        ax.imshow(self.tubulin, cmap='gray')
        self.qimg_tubulin = ImageQt(utils.canvas_to_pil(canvas))


class ExplorationGui(QWidget):
    def __init__(self):

        self.hoechst = None
        self.edu = None
        self.pericentrin = None
        self.tubulin = None
        self.resolution = 1550.3e-4

        QWidget.__init__(self)
        uic.loadUi('gui_selection.ui', self)
        for l in [self.lblEduMin, self.lblEduMax, self.lblEduAvg, self.lblTubMin, self.lblTubMax, self.lblTubAvg]:
            l.setStyleSheet('color: grey')

        self.current_sample_id = 0
        self.centrosome_dropped = False
        self.samples = None

        self.prevButton.pressed.connect(self.on_prev_button)
        self.nextButton.pressed.connect(self.on_next_button)
        self.plotButton.pressed.connect(self.plot_everything_debug)

    def process_images(self):
        if self.hoechst is None or self.edu is None or self.pericentrin is None or self.tubulin is None:
            raise Exception('empty images on some channels.')

        logger.info('applying nuclei algorithm')
        r = 6  # [um]
        imgseg, props = m.nuclei_segmentation(self.hoechst, radius=r * self.resolution)

        if len(props) == 0:
            logger.info('found no nuclear features on the hoechst image.')
            return

        # self.nuclei_features = m.nuclei_features(imgseg, area_thresh=(r * self.resolution) ** 2 * np.pi)
        self.nuclei = m.nuclei_features(imgseg)
        for i, n in enumerate(self.nuclei):
            n['id'] = i

        logger.info('applying cell boundary algorithm')
        self.cells, _ = m.cell_boundary(self.tubulin, self.hoechst)

        self.samples, self.df = position.measure(self.hoechst, self.pericentrin, self.edu, self.nuclei, self.cells,
                                        self.resolution)

        self.renderingThread = RenderImagesThread(self.hoechst, self.edu, self.pericentrin, self.tubulin,
                                                  self.imgCell, self.imgEdu)
        self.renderingThread.finished.connect(self.render_images)
        self.renderingThread.terminated.connect(self.trigger_render)
        self.renderingMutex = QMutex()
        self.render_cell()

    def render_cell(self):
        logger.info('render_cell')
        logger.debug('self.current_sample_id: %d' % self.current_sample_id)
        if len(self.samples) == 0:
            raise Exception('no samples to show!')

        sample = [s for s in self.samples if s['id'] == self.current_sample_id][0]
        cell = sample['cell']
        nucleus = sample['nucleus']

        self.renderingMutex.lock()
        if self.renderingThread.isRunning():
            logger.debug('terminating rendering thread')
            self.renderingThread.quit()
        else:
            self.renderingThread.setIndividual(self.current_sample_id, sample)
            self.trigger_render()
        self.renderingMutex.unlock()

        c, r = nucleus.boundary.xy
        r_n, c_n = draw.polygon(r, c)
        logger.debug('feat id: %d' % sample['id'])

        self.mplEduHist.hide()
        self.imgEdu.clear()
        self.imgCell.clear()
        self.imgPericentrin.clear()
        self.imgTubulin.clear()

        self.lblEduMin.setText('min ' + m.eng_string(self.edu[r_n, c_n].min(), format='%0.1f', si=True))
        self.lblEduMax.setText('max ' + m.eng_string(self.edu[r_n, c_n].max(), format='%0.1f', si=True))
        self.lblEduAvg.setText('avg ' + m.eng_string(self.edu[r_n, c_n].mean(), format='%0.1f', si=True))
        self.lblTubMin.setText('min ' + m.eng_string(self.tubulin[r_n, c_n].min(), format='%0.1f', si=True))
        self.lblTubMax.setText('max ' + m.eng_string(self.tubulin[r_n, c_n].max(), format='%0.1f', si=True))
        self.lblTubAvg.setText('avg ' + m.eng_string(self.tubulin[r_n, c_n].mean(), format='%0.1f', si=True))
        self.lblCentr_n.setText('%d' % len([s for s in sample['centrosomes'] if s is not None]))
        self.lblId.setText('id %d' % sample['id'])
        # self.lblOK.setText('OK' if valid else 'no OK')

        fig = Figure((self.imgCentrCloseup.width() / mydpi, self.imgCentrCloseup.height() / mydpi), subplotpars=sp,
                     dpi=mydpi)
        canvas = FigureCanvas(fig)
        ax = fig.gca()
        ax.set_aspect('equal')
        ax.set_axis_off()
        l, b, w, h = fig.bbox.bounds

        c1, c2 = sample['centrosomes']
        self.lblDist_nc.setText('%0.2f' % nucleus.centroid.distance(c1))
        self.lblDist_nb.setText('%0.2f' % nucleus.exterior.distance(c1))
        self.lblDist_cc.setText('%0.2f' % cell.centroid.distance(c1))
        self.lblDist_cb.setText('%0.2f' % cell.exterior.distance(c1))

        ax.imshow(self.pericentrin, cmap='gray')
        if c1 is not None:
            c = plt.Circle((c1.x, c1.y), radius=5, facecolor='none', edgecolor='r',
                           linestyle='--', linewidth=1, zorder=5)
            ax.add_artist(c)
        if c2 is not None:
            c = plt.Circle((c2.x, c2.y), radius=5, facecolor='none', edgecolor='b',
                           linestyle='--', linewidth=1, zorder=5)
            ax.add_artist(c)

        ax.set_xlim(c1.x - w / 8, c1.x + w / 8)
        ax.set_ylim(c1.y - h / 8, c1.y + h / 8)

        qimg_closeup = ImageQt(utils.canvas_to_pil(canvas))
        self.imgCentrCloseup.setPixmap(QPixmap.fromImage(qimg_closeup))
        self.update()

    @QtCore.pyqtSlot()
    def trigger_render(self):
        self.renderingThread.start()

    @QtCore.pyqtSlot()
    def render_images(self):
        if self.current_sample_id != self.renderingThread.id:
            logger.info('rendering thread finished but for a different id! self %d thread %d' % (
                self.current_sample_id, self.renderingThread.id))

            self.render_cell()
            return

        if self.renderingMutex.tryLock():
            self.imgEdu.setPixmap(QPixmap.fromImage(self.renderingThread.qimg_edu))
            self.imgCell.setPixmap(QPixmap.fromImage(self.renderingThread.qimg_hoechst))
            self.imgPericentrin.setPixmap(QPixmap.fromImage(self.renderingThread.qimg_peri))
            self.imgTubulin.setPixmap(QPixmap.fromImage(self.renderingThread.qimg_tubulin))

            sample = [s for s in self.samples if s['id'] == self.current_sample_id][0]
            nucleus = sample['nucleus']
            c, r = nucleus.boundary.xy
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
        for nuc_feat in self.nuclei:
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
                                'c1_d_cell_bound': 'cell boundary', })
        dd = pd.melt(df, id_vars=['dna'])
        dd = dd.rename(columns={'value': 'distance'})

        g = sns.FacetGrid(dd, col="variable", col_wrap=2)
        # g = g.map(plt.scatter, "dna", "distance", edgecolor="w")
        g = g.map(sns.regplot, "dna", "distance")
        g.add_legend()

    @QtCore.pyqtSlot()
    def on_prev_button(self):
        logger.info('on_prev_button')
        # self.prevButton.setEnabled(False)
        self.current_sample_id = (self.current_sample_id - 1) % len(self.samples)
        self.render_cell()
        # self.prevButton.setEnabled(True)

    @QtCore.pyqtSlot()
    def on_next_button(self):
        logger.info('on_next_button')
        # self.nextButton.setEnabled(False)
        self.current_sample_id = (self.current_sample_id + 1) % len(self.samples)
        self.render_cell()
        # self.nextButton.setEnabled(True)
