import logging
import os

import shapely.wkt
import shapely.wkt
import shapely.wkt
import numpy as np
import pandas as pd
from skimage import color
from skimage import exposure
import shapely.wkt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

from operetta import Montage
import operetta as o
from gui.utils import canvas_to_pil
import plots as p

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('hhlab')


class FourChannels(Montage):
    def __init__(self, samples_path, base_path):
        logger.info('Initializing FourChannels object with pandas file from folder %s' % samples_path)
        self.samples = pd.read_pickle(samples_path)
        if np.any([i not in self.samples for i in ['row', 'col', 'fid']]):
            is_row, is_col, is_fid = [i in self.samples for i in ['row', 'col', 'fid']]
            raise o.NoSamplesError('key columns not in dataframe row=%s col=%s fid=%s' % (is_row, is_col, is_fid))
        self.base_path = base_path
        self.images_path = os.path.join(base_path, 'Images')
        self._super_initialized = False

    def _init_parent(self):
        if not self._super_initialized:
            logger.info('Initializing parent of operetta.Dataframe...')
            super().__init__(self.base_path)
            self._super_initialized = True

    def stack_generator(self):
        samples = self.samples.groupby(['row', 'col', 'fid']).size().reset_index()
        l = len(samples)
        for k, r in samples.iterrows():
            logger.info('stack generator: retrieving %d of %d - row=%d col=%d fid=%d' %
                        (k, l, r['row'], r['col'], r['fid']))
            yield r['row'], r['col'], r['fid']

    @staticmethod
    def render_filename(row, basepath):
        # if len(row) > 1:
        #     raise Exception('only 1 row allowed')
        name = 'r%d-c%d-f%d-i%d.jpg' % (row['row'], row['col'], row['fid'], row['id'])
        if os.path.basename(basepath) == 'render':
            path = os.path.abspath(os.path.join(basepath, name))
        else:
            path = os.path.abspath(os.path.join(basepath, 'render', name))
        return name, path

    def save_render(self, row, col, fid, path=None, max_width=50):
        hoechst_raw, tubulin_raw, pericentrin_raw, edu_raw = self.max_projection(row, col, fid)

        hoechst_gray = exposure.equalize_hist(hoechst_raw)
        tubulin_gray = exposure.equalize_hist(tubulin_raw)
        # pericentrin_gray = exposure.equalize_hist(pericentrin_gray)
        # edu_gray = exposure.equalize_hist(edu_gray)

        hoechst = color.gray2rgb(hoechst_gray)
        tubulin = color.gray2rgb(tubulin_gray)
        # pericentrin = color.gray2rgb(pericentrin_gray)
        # edu = color.gray2rgb(edu_gray)

        alexa_488 = [.29, 1., 0]
        alexa_594 = [1., .61, 0]
        alexa_647 = [.83, .28, .28]
        hoechst_33342 = [0, .57, 1.]
        # out = hoechst * hoechst_33342 * 0.25 + \
        #       tubulin * alexa_488 * 0.25 + \
        #       pericentrin * alexa_594 * 0.25 + \
        #       edu * alexa_647 * 0.25

        out = hoechst * hoechst_33342 * 0.4 + tubulin * alexa_488 * 0.4

        fig_general = Figure((max_width * 4 / 150, max_width * 4 / 150), dpi=150)
        canvas_g = FigureCanvas(fig_general)
        axg = fig_general.gca()
        fig_closeup = Figure((max_width / 150, max_width / 150), dpi=150)
        canvas_c = FigureCanvas(fig_closeup)
        axc = fig_general.gca()

        if path is None:
            basepath = os.path.dirname(self.dir)
            basepath = os.path.abspath(os.path.join(basepath, 'render'))
        else:
            basepath = os.path.abspath(path)

        s = self.samples
        dfi = s[(s['row'] == row) & (s['col'] == col) & (s['fid'] == fid)]
        axg.cla()
        for ix, smp in dfi.groupby('id'):
            nucleus = shapely.wkt.loads(smp['nucleus'].values[0])
            cell = shapely.wkt.loads(smp['cell'].values[0])

            c1 = shapely.wkt.loads(smp['c1'].values[0]) if smp['c1'].values[0] is not None else None
            c2 = shapely.wkt.loads(smp['c2'].values[0]) if smp['c2'].values[0] is not None else None
            p.render_cell(nucleus, cell, [c1, c2], ax=axg)

            # render and save closeup image
            minx, miny, maxx, maxy = cell.bounds
            w, h = tubulin_gray.shape
            axc.cla()
            axc.imshow(out, extent=[0, w / self.pix_per_um, h / self.pix_per_um, 0])
            p.render_cell(nucleus, cell, [c1, c2], ax=axg)
            # ax.text(nucleus.centroid.x * self.pix_per_um, nucleus.centroid.y * self.pix_per_um, ix, color='w')
            axg.set_xlim([minx - 20, maxx + 20])
            axg.set_ylim([miny - 20, maxy + 20])

            pil = canvas_to_pil(canvas_c)
            name = 'r%d-c%d-f%d-i%d.jpg' % (row, col, fid, ix)
            fpath = os.path.abspath(os.path.join(basepath, name))
            pil.save(o.ensure_dir(fpath))

        axg.plot([5, 5 + 10], [5, 5], c='w', lw=4)
        axg.text(5 + 1, 5 + 1.5, '10 um', color='w')

        w, h = tubulin_gray.shape
        axg.imshow(out, extent=[0, w / self.pix_per_um, h / self.pix_per_um, 0])
        axg.set_xlim([0, w / self.pix_per_um])
        axg.set_ylim([0, h / self.pix_per_um])
        axg.set_axis_off()
        fig_general.tight_layout()
        pil = canvas_to_pil(canvas_g)
        name = 'r%d-c%d-f%d.jpg' % (row, col, fid)
        fpath = os.path.abspath(os.path.join(basepath, name))
        pil.save(o.ensure_dir(fpath))

    def max_projection(self, row, col, fid):
        self._init_parent()
        return super().max_projection(row, col, fid)
