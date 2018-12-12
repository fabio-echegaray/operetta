import logging

from PIL.ImageQt import ImageQt
from PyQt4 import QtCore, uic
from PyQt4.QtGui import QPixmap, QWidget
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.figure import SubplotParams

from . import utils

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('hhlab')
sp = SubplotParams(left=0., bottom=0., right=1., top=1.)
mydpi = 72


class BrowseGui(QWidget):
    def __init__(self, operetta=None, exploration_gui=None):
        if exploration_gui is None:
            raise Exception('need a gui object to show my things.')

        if operetta is not None:
            logger.info('using operetta file structure.')
            self.op = operetta
            self.gen = self.op.stack_generator()
            row, col, fid = self.gen.__next__()
            # row, col, fid = 1, 5, 23
            logger.debug('fid=%d' % fid)
            self.hoechst, self.tubulin, self.pericentrin, self.edu = self.op.max_projection(row, col, fid)

        self.egui = exploration_gui

        QWidget.__init__(self)
        uic.loadUi('gui/gui_browse.ui', self)
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
        qimg = ImageQt(utils.canvas_to_pil(self.canvas))
        self.imgHoechst.setPixmap(QPixmap.fromImage(qimg))

        self.ax.imshow(self.edu, cmap='gray')
        qimg = ImageQt(utils.canvas_to_pil(self.canvas))
        self.imgEdu.setPixmap(QPixmap.fromImage(qimg))

        self.ax.imshow(self.tubulin, cmap='gray')
        qimg = ImageQt(utils.canvas_to_pil(self.canvas))
        self.imgTubulin.setPixmap(QPixmap.fromImage(qimg))

        self.ax.imshow(self.pericentrin, cmap='gray')
        qimg = ImageQt(utils.canvas_to_pil(self.canvas))
        self.imgPericentrin.setPixmap(QPixmap.fromImage(qimg))

        self.update()
