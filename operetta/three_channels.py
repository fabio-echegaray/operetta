import logging
import os

import shapely.wkt
import numpy as np
import pandas as pd
from skimage import color
from skimage import exposure
import shapely.wkt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

import operetta as o
from gui.utils import canvas_to_pil

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('hhlab')

# FIXME: this class is not working after major refactoring of the FourChannels class
class ThreeChannels(o.Montage):
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
        name = 'r%d-c%d-f%d-i%d.jpg' % (row['row'], row['col'], row['fid'], row['id'])
        path = os.path.abspath(os.path.join(basepath, 'render/%s' % (name)))
        return name, path

    # def save_render(self, row, col, fid, max_width=50):
    #     dapi, phosphoRb, edu = self.max_projection(row, col, fid)
    #
    #     dapi = exposure.equalize_hist(dapi)
    #     phosphoRb = exposure.equalize_hist(phosphoRb)
    #     dapi = transform.resize(dapi, (max_width, max_width))
    #     phosphoRb = transform.resize(phosphoRb, (max_width, max_width))
    #
    #     dapi = color.gray2rgb(dapi)
    #     phosphoRb = color.gray2rgb(phosphoRb)
    #
    #     alexa_488 = [.29, 1., 0]
    #     alexa_594 = [1., .61, 0]
    #     alexa_647 = [.83, .28, .28]
    #     hoechst_33342 = [0, .57, 1.]
    #     out = dapi * hoechst_33342 + phosphoRb * alexa_594 + edu * alexa_647
    #
    #     basepath = os.path.dirname(self.dir)
    #     path = os.path.abspath(os.path.join(basepath, 'render/%d-%d-%d.jpg' % (fid, row, col)))
    #     io.imsave(path, out)

    def save_render(self, row, col, fid, path=None, max_width=50):
        dapi_raw, phosphoRb_raw, edu_raw = self.max_projection(row, col, fid)

        phrb_gray = exposure.equalize_hist(phosphoRb_raw)
        edu_gray = exposure.equalize_hist(edu_raw)

        dapi = color.gray2rgb(dapi_raw)
        phrb = color.gray2rgb(phrb_gray)
        edu = color.gray2rgb(edu_gray)

        alexa_488 = [.29, 1., 0]
        alexa_594 = [1., .61, 0]
        alexa_647 = [.83, .28, .28]
        hoechst_33342 = [0, .57, 1.]
        out = dapi * hoechst_33342 * 0.1 + phrb * alexa_488 * 0.1 + edu * alexa_647 * 0.1

        s = self.samples
        dfi = s[(s['row'] == row) & (s['col'] == col) & (s['fid'] == fid)]

        fig = Figure((3, 3), dpi=150)
        canvas = FigureCanvas(fig)
        ax = fig.gca()

        if path is None:
            basepath = os.path.dirname(self.images_path)
            basepath = os.path.abspath(os.path.join(basepath, 'render'))
        else:
            basepath = os.path.abspath(path)

        ax.cla()
        # ax.imshow(out, extent=[0, w / self.pix_per_um, h / self.pix_per_um, 0])
        for id, _dfi in dfi.set_index('id').iterrows():
            nucleus = shapely.wkt.loads(_dfi['nucleus'])

            ax.set_aspect('equal')
            ax.set_axis_off()
            l, b, w, h = ax.get_figure().bbox.bounds
            # ax.set_xlim(cen.x - w / 2, cen.x + w / 2)
            # ax.set_ylim(cen.y - h / 2, cen.y + h / 2)
            # x0, xf = cen.x - w / 2, cen.x + w / 2
            # y0, yf = cen.y - h / 2, cen.y + h / 2
            # x0 = max(0, x0) + 5 * self.pix_per_um
            # y0 = max(0, y0) + 5 * self.pix_per_um

            x, y = nucleus.exterior.xy
            cen = nucleus.centroid
            ax.plot(x, y, color='red', linewidth=1, solid_capstyle='round', zorder=2)
            ax.plot(cen.x, cen.y, color='red', marker='+', linewidth=1, solid_capstyle='round', zorder=2)

            # ax.plot([x0, x0 + 10 * pix_per_um], [y0, y0], c='w', lw=4)
            # ax.text(x0 + 1 * pix_per_um, y0 + 1.5 * pix_per_um, '10 um', color='w')

        w, h = phrb_gray.shape
        ax.imshow(dapi_raw, extent=[0, w / self.pix_per_um, h / self.pix_per_um, 0])
        pil = canvas_to_pil(canvas)
        fpath = os.path.join(basepath, 'dapi-r%d-c%d-f%d.jpg' % (row, col, fid))
        pil.save(o.ensure_dir(fpath))

        ax.imshow(phosphoRb_raw, extent=[0, w / self.pix_per_um, h / self.pix_per_um, 0])
        pil = canvas_to_pil(canvas)
        fpath = os.path.join(basepath, 'phospho-r%d-c%d-f%d.jpg' % (row, col, fid))
        pil.save(o.ensure_dir(fpath))

    def max_projection(self, row, col, fid):
        if not self._super_initialized:
            logger.info('Initializing parent of operetta.Dataframe...')
            super().__init__(self.base_path)
            self._super_initialized = True
        return super().max_projection(row, col, fid)
