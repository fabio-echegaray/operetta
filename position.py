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
from shapely.geometry import Point
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

        self.build_df()

        self.renderingThread = RenderImagesThread(self.hoechst, self.edu, self.pericentrin, self.tubulin,
                                                  self.imgCell, self.imgEdu)
        self.renderingThread.finished.connect(self.render_images)
        self.renderingThread.terminated.connect(self.trigger_render)
        self.renderingMutex = QMutex()
        self.render_cell()

    def build_df(self):
        self.df = pd.DataFrame()
        self.samples = list()

        for nuclei in self.nuclei:
            x0, y0, xf, yf = [int(u) for u in nuclei['boundary'].bounds]

            # search for closest cell boundary based on centroids
            cells = list()
            for cl in self.cells:
                cells.append({'id': cl['id'],
                              'boundary': cl['boundary'],
                              'd': cl['boundary'].centroid.distance(nuclei['boundary'].centroid)})
            cells = sorted(cells, key=lambda k: k['d'])

            if m.is_valid_sample(self.hoechst, cells[0]['boundary'], nuclei['boundary'], self.nuclei):
                pericentrin_crop = self.pericentrin[y0:yf, x0:xf]
                logger.info('applying centrosome algorithm for nuclei %d' % nuclei['id'])
                # self.pericentrin = exposure.equalize_adapthist(pcrop, clip_limit=0.03)
                cntr = m.centrosomes(pericentrin_crop, max_sigma=self.resolution * 0.5)
                cntr[:, 0] += x0
                cntr[:, 1] += y0
                cntrsmes = list()
                for k, c in enumerate(cntr):
                    pt = Point(c[0], c[1])
                    pti = m.integral_over_surface(self.pericentrin, pt.buffer(self.resolution * 1))
                    cntrsmes.append({'id': k, 'pt': pt, 'i': pti})
                    cntrsmes = sorted(cntrsmes, key=lambda k: k['i'], reverse=True)

                logger.debug('centrosomes {:s}'.format(str(cntrsmes)))

                edu_int = m.integral_over_surface(self.edu, nuclei['boundary'])
                dna_int = m.integral_over_surface(self.hoechst, nuclei['boundary'])

                twocntr = len(cntrsmes) >= 2 and cntrsmes[1]['i'] > 900
                c1 = cntrsmes[0]['pt']
                c2 = cntrsmes[1]['pt'] if twocntr else None

                nucb = nuclei['boundary']
                cllb = cells[0]['boundary']
                d = pd.DataFrame(data={'id': [nuclei['id']],
                                       'dna_int': [dna_int],
                                       'edu_int': [edu_int],
                                       'centrosomes': [len(cntrsmes)],
                                       'c1_d_nuc_centr': [nucb.centroid.distance(c1)],
                                       'c2_d_nuc_centr': [nucb.centroid.distance(c2) if twocntr else np.nan],
                                       'c1_d_nuc_bound': [nucb.exterior.distance(c1)],
                                       'c2_d_nuc_bound': [nucb.exterior.distance(c2) if twocntr else np.nan],
                                       'c1_d_cell_centr': [cllb.centroid.distance(c1)],
                                       'c2_d_cell_centr': [cllb.centroid.distance(c2) if twocntr else np.nan],
                                       'c1_d_cell_bound': [cllb.exterior.distance(c1)],
                                       'c2_d_cell_bound': [cllb.exterior.distance(c2) if twocntr else np.nan],
                                       })
                self.df = self.df.append(d, ignore_index=True)
                self.samples.append({'id': nuclei['id'], 'cell': cells[0]['boundary'], 'nucleus': nuclei['boundary'],
                                     'centrosomes': [c1, c2]})
        print(self.df)

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

        qimg_closeup = ImageQt(canvas_to_pil(canvas))
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
        print(self.current_sample_id)
        self.render_cell()
        # self.nextButton.setEnabled(True)


if __name__ == '__main__':
    b_path = '/Volumes/Kidbeat/data/centr-dist(u2os)__2018-11-27T18_08_10-Measurement 1/Images'

    operetta = o.Montage(b_path)

    # for row, col, fid in operetta.stack_generator():
    #     logger.info('%d %d %d' % (row, col, fid))
    #     operetta.save_render(row, col, fid,max_width=300)

    # logger.info('applying nuclei algorithm')
    # outdf = pd.DataFrame()
    # for row, col, fid in operetta.stack_generator():
    #     logger.info('%d %d %d' % (row, col, fid))
    #     hoechst, tubulin, pericentrin, edu = operetta.max_projection(row, col, fid)
    #     r = 30  # [um]
    #     resolution = 1550.3e-4
    #     imgseg, props = m.nuclei_segmentation(hoechst, radius=r * resolution)
    #     # nuclei_features = m.nuclei_features(imgseg, area_thresh=(r * resolution) ** 2 * np.pi)
    #     operetta.add_mesurement(row, col, fid, 'nuclei found', len(np.unique(imgseg)))
    #     if len(props) > 0:
    #         # interested = props[(props['eccentricity'] > 0.6)].index
    #         outdf = outdf.append(props)
    # pd.to_pickle(outdf, 'out/nuclei.pandas')
    # operetta.files.to_csv('out/operetta.csv')
    #
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
