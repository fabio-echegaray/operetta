import logging
import os
import re

import numpy as np
import pandas as pd
from skimage import color
from skimage import transform
from skimage import exposure
from skimage import io
import shapely.wkt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

import measurements as m
from gui.utils import canvas_to_pil
from gui.explore import RenderImagesThread
from gui import convert_to, meter, pix, um

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('hhlab')


def ensure_dir(file_path):
    file_path = os.path.abspath(file_path)
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    return file_path


class Montage:
    def __init__(self, folder):
        l = list()
        self.dir = os.path.join(folder, 'Images')
        #  build a list of dicts for every image file in the directory
        for root, directories, filenames in os.walk(folder):
            for filename in filenames:
                ext = filename.split('.')[-1]
                if ext == 'tiff':
                    # joinf = os.path.join(root, filename)
                    row, col, f, p, ch, sk, fk, fl = [int(g) for g in re.search(
                        'r([0-9]+)c([0-9]+)f([0-9]+)p([0-9]+)-ch([0-9]+)sk([0-9]+)fk([0-9]+)fl([0-9]+).tiff',
                        filename).groups()]
                    i = {'row': row, 'col': col, 'f': f, 'p': p, 'ch': ch, 'sk': sk, 'fk': fk, 'fl': fl}
                    l.append(i)
        self.files = pd.DataFrame(l)

    @staticmethod
    def filename(row):
        if row.index.size > 1: raise Exception('only 1 row accepted')
        return 'r%02dc%02df%02dp%02d-ch%dsk%dfk%dfl%d.tiff' % tuple(
            row[['row', 'col', 'f', 'p', 'ch', 'sk', 'fk', 'fl']].values.tolist()[0])

    @property
    def rows(self):
        return self.files['row'].unique()

    @property
    def columns(self):
        return self.files['col'].unique()

    @property
    def channels(self):
        return self.files['ch'].unique()

    @property
    def z_positions(self):
        return self.files['p'].unique()

    def stack_generator(self):
        group = self.files.groupby(['row', 'col', 'f'])
        l = len(group)
        for k, (ig, g) in enumerate(group):
            logger.info('stack generator: retrieving %d of %d - row=%d col=%d fid=%d' % (k, l, ig[0], ig[1], ig[2]))
            yield ig

    def add_mesurement(self, row, col, f, name, value):
        self.files.loc[(self.files['row'] == row) & (self.files['col'] == col) & (self.files['f'] == f), name] = value

    def save_render(self, row, col, fid, max_width=50):
        hoechst, tubulin, pericentrin, edu = self.max_projection(row, col, fid)

        hoechst = exposure.equalize_hist(hoechst)
        tubulin = exposure.equalize_hist(tubulin)
        pericentrin = exposure.equalize_hist(pericentrin)
        hoechst = transform.resize(hoechst, (max_width, max_width))
        tubulin = transform.resize(tubulin, (max_width, max_width))
        pericentrin = transform.resize(pericentrin, (max_width, max_width))

        hoechst = color.gray2rgb(hoechst)
        tubulin = color.gray2rgb(tubulin)
        pericentrin = color.gray2rgb(pericentrin)

        alexa_488 = [.29, 1., 0]
        alexa_594 = [1., .61, 0]
        alexa_647 = [.83, .28, .28]
        hoechst_33342 = [0, .57, 1.]
        out = hoechst * hoechst_33342 + tubulin * alexa_488 + pericentrin * alexa_594 + edu * alexa_647

        basepath = os.path.dirname(self.dir)
        path = os.path.abspath(os.path.join(basepath, 'render/%d-%d-%d.jpg' % (fid, row, col)))
        io.imsave(path, out)

    def max_projection(self, row, col, f):
        group = self.files[(self.files['row'] == row) & (self.files['col'] == col) & (self.files['f'] == f)]
        channels = list()
        for ic, c in group.groupby('ch'):  # iterate over channels
            files = list()
            for iz, z in c.groupby('p'):  # iterate over z positions
                if z.index.size == 1:
                    files.append(self.filename(z))
            first_file = os.path.join(self.dir, files.pop())
            max = io.imread(first_file)
            for f in files:
                fname = os.path.join(self.dir, f)
                img = io.imread(fname)
                max = np.maximum(max, img)
            channels.append(max)
        return channels


class Dataframe(Montage):
    def __init__(self, samples_path, base_path):
        self.samples = pd.read_pickle(samples_path)
        if np.any([i not in self.samples for i in ['row', 'col', 'fid']]):
            is_row, is_col, is_fid = [i not in self.samples for i in ['row', 'col', 'fid']]
            raise Exception('key columns not in provided dataframe row=%s col=%s fid=%s' % (is_row, is_col, is_fid))
        self.base_path = base_path
        self.images_path = os.path.join(base_path, 'Images')
        self._super_initialized = False
        self.um_per_pix = convert_to(1.8983367649421008E-07 * meter / pix, um / pix).n()
        self.pix_per_um = 1 / self.um_per_pix
        self.pix_per_um = float(self.pix_per_um.args[0])

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

    def save_render(self, row, col, fid, id, path=None, max_width=50):
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

        s = self.samples
        dfi = s[(s['row'] == row) & (s['col'] == col) & (s['fid'] == fid) & (s['id'] == id)]

        fig = Figure((3, 3), dpi=150)
        canvas = FigureCanvas(fig)
        ax = fig.gca()

        if path is None:
            basepath = os.path.dirname(self.dir)
            basepath = os.path.abspath(os.path.join(basepath, 'render'))
        else:
            basepath = os.path.abspath(path)

        nucleus = shapely.wkt.loads(dfi['nucleus'].values[0])
        cell = shapely.wkt.loads(dfi['cell'].values[0])
        if m.is_valid_sample(tubulin_raw, cell, nucleus):
            ax.cla()
            c1 = shapely.wkt.loads(dfi['c1'].values[0]) if dfi['c1'].values[0] is not None else None
            c2 = shapely.wkt.loads(dfi['c2'].values[0]) if dfi['c2'].values[0] is not None else None
            minx, miny, maxx, maxy = cell.bounds
            w, h = tubulin_gray.shape
            ax.imshow(out, extent=[0, w / self.pix_per_um, h / self.pix_per_um, 0])
            RenderImagesThread.render(ax, nucleus, cell, [c1, c2],
                                      xlim=[minx - 20, maxx + 20], ylim=[miny - 20, maxy + 20])

            pil = canvas_to_pil(canvas)
            fpath = os.path.join(basepath, 'r%d-c%d-f%d-i%d.jpg' % (row, col, fid, id))
            pil.save(ensure_dir(fpath))

    def max_projection(self, row, col, fid):
        if not self._super_initialized:
            logger.info('Initializing parent of operetta.Dataframe...')
            super().__init__(self.base_path)
            self._super_initialized = True
        return super().max_projection(row, col, fid)
