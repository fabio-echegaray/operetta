import logging
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import skimage.draw as draw
import skimage.exposure as exposure
from PIL import Image
from PIL.ImageQt import ImageQt
from PyQt4 import QtCore, uic
from PyQt4.QtCore import *
from PyQt4.QtCore import QThread
from PyQt4.QtGui import QApplication, QPixmap, QWidget
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure, SubplotParams
from matplotlib.ticker import EngFormatter
from shapely.geometry.polygon import Polygon

import measurements as m
import operetta as o

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


class RenderImagesThread(QThread):
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


class BrowseGui(QWidget):
    def __init__(self, operetta=None, exploration_gui=None):
        if exploration_gui is None:
            raise Exception('need a gui object to show my things.')

        if operetta is not None:
            logger.info('using operetta file structure.')
            self.op = operetta
            self.gen = self.op.stack_generator()
            row, col, fid = self.gen.__next__()
            logger.debug('fid=%d' % fid)
            self.hoechst, self.tubulin, self.pericentrin, self.edu = self.op.max_projection(row, col, fid)

        self.egui = exploration_gui

        QWidget.__init__(self)
        uic.loadUi('gui_browse.ui', self)
        self.nextButton.pressed.connect(self.on_next_button)
        self.processButton.pressed.connect(self.on_process_button)

        fig = Figure((self.imgHoechst.width() / mydpi, self.imgHoechst.height() / mydpi), subplotpars=sp, dpi=mydpi)
        self.canvas = FigureCanvas(fig)
        self.ax = fig.gca()
        self.ax.set_aspect('equal')
        self.ax.set_axis_off()

        self.render_images()

    @QtCore.pyqtSlot()
    def on_next_button(self):
        logger.info('on_next_button')
        row, col, fid = self.gen.__next__()
        # row, col, fid = 1, 1, 1000
        logger.debug('fid=%d' % fid)
        self.hoechst, self.tubulin, self.pericentrin, self.edu = self.op.max_projection(row, col, fid)
        self.render_images()

    @QtCore.pyqtSlot()
    def on_process_button(self):
        logger.info('on_process_button')

        self.egui.hoechst = self.hoechst
        self.egui.tubulin = self.tubulin
        self.egui.pericentrin = self.pericentrin
        self.egui.edu = self.edu

        self.egui.process_images()

    def render_images(self):
        logger.info('render image')

        self.imgHoechst.clear()
        self.imgEdu.clear()
        self.imgPericentrin.clear()
        self.imgTubulin.clear()

        self.ax.imshow(self.hoechst, cmap='gray')
        qimg = ImageQt(canvas_to_pil(self.canvas))
        self.imgHoechst.setPixmap(QPixmap.fromImage(qimg))

        self.ax.imshow(self.edu, cmap='gray')
        qimg = ImageQt(canvas_to_pil(self.canvas))
        self.imgEdu.setPixmap(QPixmap.fromImage(qimg))

        self.ax.imshow(self.tubulin, cmap='gray')
        qimg = ImageQt(canvas_to_pil(self.canvas))
        self.imgTubulin.setPixmap(QPixmap.fromImage(qimg))

        self.ax.imshow(self.pericentrin, cmap='gray')
        qimg = ImageQt(canvas_to_pil(self.canvas))
        self.imgPericentrin.setPixmap(QPixmap.fromImage(qimg))

        self.update()


class ExplorationGui(QWidget):
    def __init__(self):

        self.hoechst = None
        self.edu = None
        self.pericentrin = None
        self.tubulin = None
        # self.resolution = 4.5
        self.resolution = 1550.3

        QWidget.__init__(self)
        uic.loadUi('gui_selection.ui', self)
        for l in [self.lblEduMin, self.lblEduMax, self.lblEduAvg, self.lblTubMin, self.lblTubMax, self.lblTubAvg]:
            l.setStyleSheet('color: grey')

        self.current_nuclei_id = 0
        self.centrosome_dropped = False

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
        self.nuclei_features = m.nuclei_features(imgseg)

        logger.info('applying centrosome algorithm')
        self.pericentrin = exposure.equalize_adapthist(self.pericentrin, clip_limit=0.03)
        self.centrosomes = m.centrosomes(self.pericentrin, max_sigma=self.resolution * 0.2)
        logger.debug('centrosomes {:s}'.format(str(self.centrosomes)))

        logger.info('applying cell boundary algorithm')
        self.cells, _ = m.cell_boundary(self.tubulin, self.hoechst)

        self.renderingThread = RenderImagesThread(self.hoechst, self.edu, self.pericentrin, self.tubulin, self.cells,
                                                  self.nuclei_features, self.centrosomes, self.imgCell, self.imgEdu)
        self.renderingThread.finished.connect(self.render_images)
        self.renderingThread.terminated.connect(self.trigger_render)
        self.renderingMutex = QMutex()

        self.build_df()
        self.render_cell()

    def build_df(self):
        self.df = pd.DataFrame()
        for nuc in self.nuclei_features:
            valid, cell, nuclei, cntrsmes = m.get_nuclei_features(self.hoechst, nuc['boundary'], self.cells,
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

        valid, cell, nucleus, cntrsmes = m.get_nuclei_features(self.hoechst, nuc_bnd, self.cells, self.nuclei_features,
                                                               self.centrosomes)
        logger.debug('{} {} {} {}'.format(valid, cell, nucleus, cntrsmes))

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

        self.lblEduMin.setText('min ' + m.eng_string(self.edu[r_n, c_n].min(), format='%0.1f', si=True))
        self.lblEduMax.setText('max ' + m.eng_string(self.edu[r_n, c_n].max(), format='%0.1f', si=True))
        self.lblEduAvg.setText('avg ' + m.eng_string(self.edu[r_n, c_n].mean(), format='%0.1f', si=True))
        self.lblTubMin.setText('min ' + m.eng_string(self.tubulin[r_n, c_n].min(), format='%0.1f', si=True))
        self.lblTubMax.setText('max ' + m.eng_string(self.tubulin[r_n, c_n].max(), format='%0.1f', si=True))
        self.lblTubAvg.setText('avg ' + m.eng_string(self.tubulin[r_n, c_n].mean(), format='%0.1f', si=True))
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
        self.current_nuclei_id = (self.current_nuclei_id - 1) % len(self.nuclei_features)
        self.render_cell()
        # self.prevButton.setEnabled(True)

    @QtCore.pyqtSlot()
    def on_next_button(self):
        logger.info('on_next_button')
        # self.nextButton.setEnabled(False)
        self.current_nuclei_id = (self.current_nuclei_id + 1) % len(self.nuclei_features)
        self.render_cell()
        # self.nextButton.setEnabled(True)


if __name__ == '__main__':
    b_path = '/Volumes/Unbreakable/data/operetta/u2os__2018-10-26T17_55_11-Measurement 4/Images'

    operetta = o.Montage(b_path)

    # logger.info('applying nuclei algorithm')
    # outdf = pd.DataFrame()
    # for row, col, fid in operetta.stack_generator():
    #     logger.info('%d %d %d' % (row, col, fid))
    #     hoechst, tubulin, pericentrin, edu = operetta.max_projection(row, col, fid)
    #     r = 6  # [um]
    #     resolution = 1550.3
    #     imgseg, props = m.nuclei_segmentation(hoechst, radius=r * resolution)
    #     # nuclei_features = m.nuclei_features(imgseg, area_thresh=(r * resolution) ** 2 * np.pi)
    #     if len(props) > 0:
    #         # interested = props[(props['eccentricity'] > 0.6)].index
    #         outdf = outdf.append(props)
    # pd.to_pickle(outdf, 'out/nuclei.pandas')

    # outdf = pd.read_pickle('out/nuclei.pandas')
    # outdf = outdf[outdf['area'] < 1e4]
    # sns.scatterplot('area', 'mean_intensity', data=outdf)
    # # plt.hist(outdf['mean_intensity'], bins=100,log=True)
    # # plt.hist(outdf['area'], bins=100, log=True)
    # plt.show()

    base_path = os.path.abspath('%s' % os.getcwd())
    logging.info('Qt version:' + QT_VERSION_STR)
    logging.info('PyQt version:' + PYQT_VERSION_STR)
    logging.info('Working dir:' + os.getcwd())
    logging.info('Base dir:' + base_path)
    os.chdir(base_path)

    app = QApplication(sys.argv)

    egui = ExplorationGui()
    bgui = BrowseGui(operetta=operetta, exploration_gui=egui)
    # from pycallgraph import PyCallGraph
    # from pycallgraph.output import GraphvizOutput
    #
    # with PyCallGraph(output=GraphvizOutput()):
    bgui.show()
    egui.show()
    code = app.exec_()
    sys.exit(code)
