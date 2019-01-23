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
import measurements as m
from gui.explore import RenderImagesThread
import operetta as o
from gui.utils import canvas_to_pil

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('hhlab')


class FourChannels(Montage):
    def __init__(self, samples_path, base_path):
        self.samples = pd.read_pickle(samples_path)
        if np.any([i not in self.samples for i in ['row', 'col', 'fid']]):
            is_row, is_col, is_fid = [i not in self.samples for i in ['row', 'col', 'fid']]
            raise Exception('key columns not in provided dataframe row=%s col=%s fid=%s' % (is_row, is_col, is_fid))
        self.base_path = base_path
        self.images_path = os.path.join(base_path, 'Images')
        self._super_initialized = False

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
        path = os.path.abspath(os.path.join(basepath, 'render/%s' % (name)))
        return name, path

    def save_render(self, row, col, fid, path=None, max_width=50):
        hoechst_raw, tubulin_raw, pericentrin_raw, edu_raw = self.max_projection(row, col, fid)

        tubulin_gray = exposure.equalize_hist(tubulin_raw)
        # pericentrin_gray = exposure.equalize_hist(pericentrin_gray)
        # edu_gray = exposure.equalize_hist(edu_gray)

        # hoechst = color.gray2rgb(hoechst_gray)
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

        # out = hoechst * hoechst_33342 * 0.2 + tubulin * alexa_488 * 0.4
        out = tubulin * alexa_488 * 0.4

        fig = Figure((3, 3), dpi=150)
        canvas = FigureCanvas(fig)
        ax = fig.gca()

        if path is None:
            basepath = os.path.dirname(self.dir)
            basepath = os.path.abspath(os.path.join(basepath, 'render'))
        else:
            basepath = os.path.abspath(path)

        s = self.samples
        dfi = s[(s['row'] == row) & (s['col'] == col) & (s['fid'] == fid)]
        for ix, smp in dfi.groupby('id'):
            nucleus = shapely.wkt.loads(smp['nucleus'].values[0])
            cell = shapely.wkt.loads(smp['cell'].values[0])
            if m.is_valid_sample(tubulin_raw, cell, nucleus):
                ax.cla()
                c1 = shapely.wkt.loads(smp['c1'].values[0]) if smp['c1'].values[0] is not None else None
                c2 = shapely.wkt.loads(smp['c2'].values[0]) if smp['c2'].values[0] is not None else None
                minx, miny, maxx, maxy = cell.bounds
                w, h = tubulin_gray.shape
                ax.imshow(out, extent=[0, w / self.pix_per_um, h / self.pix_per_um, 0])
                RenderImagesThread.render(ax, nucleus, cell, [c1, c2],
                                          xlim=[minx - 20, maxx + 20], ylim=[miny - 20, maxy + 20])

                pil = canvas_to_pil(canvas)
                fpath = os.path.join(basepath, 'r%d-c%d-f%d-i%d.jpg' % (row, col, fid, ix))
                pil.save(o.ensure_dir(fpath))

    def max_projection(self, row, col, fid):
        if not self._super_initialized:
            logger.info('Initializing parent of operetta.Dataframe...')
            super().__init__(self.base_path)
            self._super_initialized = True
        return super().max_projection(row, col, fid)
