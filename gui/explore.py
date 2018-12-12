import logging

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import skimage.draw as draw
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.ticker import EngFormatter
from shapely.geometry.polygon import Polygon
from matplotlib.figure import SubplotParams

try:
    from PIL.ImageQt import ImageQt
    from PyQt4 import QtCore, uic
    from PyQt4.QtCore import *
    from PyQt4.QtCore import QThread
    from PyQt4.QtGui import QPixmap, QWidget

    LOAD_GUI = True
except ImportError:  # if calling from batch and system doesn't have GUI capabilities
    from abc import ABC

    QThread = ABC
    LOAD_GUI = False
    pass

from . import utils
from . import SUSSEX_CORAL_RED, SUSSEX_NAVY_BLUE, convert_to, meter, pix, um
import measurements as m

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('hhlab')
sp = SubplotParams(left=0., bottom=0., right=1., top=1.)
mydpi = 72


class RenderImagesThread(QThread):
    def __init__(self, hoechst, edu, pericentrin, tubulin,
                 largeQLabel, smallQLabel, pix_per_um=1):
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
        if pix_per_um == 1: logger.warning('no image resolution was set.')
        self.pix_per_um = pix_per_um
        self._rendered = False

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
        self._rendered = False

    @staticmethod
    def render(nucleus, cell, centrosomes, width=50, height=50, dpi=72, pix_per_um=1, xlim=None, ylim=None):

        # create a matplotlib axis and plot edu image
        fig = Figure((width / dpi, height / dpi), subplotpars=sp, dpi=dpi)
        canvas = FigureCanvas(fig)
        ax = fig.gca()
        ax.set_aspect('equal')
        ax.set_axis_off()
        l, b, w, h = fig.bbox.bounds

        x, y = nucleus.exterior.xy
        cen = nucleus.centroid
        ax.plot(x, y, color='red', linewidth=1, solid_capstyle='round', zorder=2)
        ax.plot(cen.x, cen.y, color='red', marker='+', linewidth=1, solid_capstyle='round', zorder=2)

        if xlim is not None and ylim is not None:
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            x0, xf = xlim
            y0, yf = ylim
            y0 = max(0, y0) + h / 12

        else:
            ax.set_xlim(cen.x - w / 2, cen.x + w / 2)
            ax.set_ylim(cen.y - h / 2, cen.y + h / 2)
            x0, xf = cen.x - w / 2, cen.x + w / 2
            y0, yf = cen.y - h / 2, cen.y + h / 2
            y0 = max(0, y0) + h / 12

        if cell is not None:
            x, y = cell.exterior.xy
            ax.plot(x, y, color='green', linewidth=1, solid_capstyle='round', zorder=1)
            cen = cell.centroid
            ax.plot(cen.x, cen.y, color='green', marker='+', linewidth=1, solid_capstyle='round', zorder=2)

        if centrosomes is not None:
            c1, c2 = centrosomes
            if c1 is not None:
                c = plt.Circle((c1.x, c1.y), radius=5, facecolor='none', edgecolor=SUSSEX_CORAL_RED,
                               linewidth=3, zorder=5)
                ax.add_artist(c)
            if c2 is not None:
                c = plt.Circle((c2.x, c2.y), radius=5, facecolor='none', edgecolor=SUSSEX_NAVY_BLUE,
                               linewidth=3, zorder=5)
                ax.add_artist(c)

        xw = (xf - x0) / 10
        x0 += xw
        print(x0, y0)
        ax.plot([x0, x0 + 10 * pix_per_um], [y0, y0], c='w', lw=4)
        ax.text(x0 + 1 * pix_per_um, y0 + 1 * pix_per_um, '10 um', color='w')

        return ax, fig, canvas

    def run(self):
        """
        Go over every item in the
        """
        if not self._rendered:
            c1 = self.centrosomes[0]['pt']
            c2 = self.centrosomes[1]['pt']
            self.ax, self.fig, self.canvas = RenderImagesThread.render(self.nucleus, self.cell, [c1, c2],
                                                                       width=self.lqlbl.width(),
                                                                       height=self.lqlbl.height(),
                                                                       dpi=mydpi, pix_per_um=self.pix_per_um)

        self.ax.imshow(self.hoechst, cmap='gray')
        self.qimg_hoechst = ImageQt(utils.canvas_to_pil(self.canvas))

        self.ax.imshow(self.pericentrin, cmap='gray')
        self.qimg_peri = ImageQt(utils.canvas_to_pil(self.canvas))

        self.ax.imshow(self.tubulin, cmap='gray')
        self.qimg_tubulin = ImageQt(utils.canvas_to_pil(self.canvas))

        self.ax.imshow(self.edu, cmap='gray')
        self.fig.set_size_inches((self.sqlbl.width() / mydpi, self.sqlbl.height() / mydpi))
        l, b, w, h = self.fig.bbox.bounds
        cen = self.nucleus.centroid
        self.ax.set_xlim(cen.x - w / 2, cen.x + w / 2)
        self.ax.set_ylim(cen.y - h / 2, cen.y + h / 2)
        self.qimg_edu = ImageQt(utils.canvas_to_pil(self.canvas))

        self._rendered = True


if LOAD_GUI:
    class ExplorationGui(QWidget):
        def __init__(self):
            self.hoechst = None
            self.edu = None
            self.pericentrin = None
            self.tubulin = None
            self.um_per_pix = convert_to(1.8983367649421008E-07 * meter / pix, um / pix).n()
            self.pix_per_um = 1 / self.um_per_pix
            self.pix_per_um = float(self.pix_per_um.args[0])

            QWidget.__init__(self)
            uic.loadUi('gui/gui_selection.ui', self)
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
            imgseg, props = m.nuclei_segmentation(self.hoechst, radius=r * self.pix_per_um)

            if len(props) == 0:
                logger.info('found no nuclear features on the hoechst image.')
                return

            # self.nuclei_features = m.nuclei_features(imgseg, area_thresh=(r * self.resolution) ** 2 * np.pi)
            self.nuclei = m.nuclei_features(imgseg)
            for i, n in enumerate(self.nuclei):
                n['id'] = i

            logger.info('applying cell boundary algorithm')
            self.cells, _ = m.cell_boundary(self.tubulin, self.hoechst)

            self.samples, self.df = m.measure_into_dataframe(self.hoechst, self.pericentrin, self.edu, self.nuclei,
                                                             self.cells,
                                                             self.pix_per_um)

            self.renderingThread = RenderImagesThread(self.hoechst, self.edu, self.pericentrin, self.tubulin,
                                                      self.imgCell, self.imgEdu, pix_per_um=self.pix_per_um)
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

            c1 = sample['centrosomes'][0]['pt']
            c2 = sample['centrosomes'][1]['pt']
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
