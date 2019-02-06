import os
import re

import numpy as np
import pandas as pd
from skimage import io

from .exceptions import ImagesFolderNotFound
from . import logger
from gui import convert_to, meter, pix, um


def ensure_dir(file_path):
    file_path = os.path.abspath(file_path)
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    return file_path


class Montage:
    def __init__(self, folder, row=None, col=None, condition_name=None):
        self.base_path = folder
        self.images_path = os.path.join(folder, 'Images')

        self.name = condition_name

        csv_path = os.path.join(folder, 'out', 'operetta.csv')
        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                'File operetta.csv is missing in the folder structure. Generate the csv file first.')
        else:
            f = pd.read_csv(csv_path)
        self.files = f

        if row is not None:
            logger.debug('original rows: %s' % self.rows())
            logger.debug('restricting rows to %s' % row)
            f = f[f['row'] == row]
            logger.debug('rows: %s' % f['row'].unique())
        if col is not None:
            logger.debug('original columns: %s' % self.columns())
            logger.debug('restricting columns to %s' % col)
            f = f[f['col'] == col]
            logger.debug('columns: %s' % f['col'].unique())
        self.files_gr = self.files.groupby(['row', 'col', 'fid']).size().reset_index()

        self.um_per_pix = convert_to(1.8983367649421008E-07 * meter / pix, um / pix).n()
        self.pix_per_um = 1 / self.um_per_pix
        self.pix_per_um = float(self.pix_per_um.args[0])
        self.um_per_pix = float(self.um_per_pix.args[0])

        if not os.path.exists(self.images_path):
            raise ImagesFolderNotFound('Images folder is not in the structure.')

    @staticmethod
    def filename(row):
        if row.index.size > 1: raise Exception('only 1 row accepted.')
        return 'r%02dc%02df%02dp%02d-ch%dsk%dfk%dfl%d.tiff' % tuple(
            row[['row', 'col', 'fid', 'p', 'ch', 'sk', 'fk', 'fl']].values.tolist()[0])

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
        l = len(self.files_gr)
        for ig, g in self.files_gr.iterrows():
            # if not (row == 1 and col == 9 and fid == 72): continue
            logger.info('stack generator:  %d of %d - row=%d col=%d fid=%d' % (ig + 1, l, g['row'], g['col'], g['fid']))
            yield g['row'], g['col'], g['fid']

    def add_mesurement(self, row, col, f, name, value):
        self.files.loc[(self.files['row'] == row) & (self.files['col'] == col) & (self.files['fid'] == f), name] = value

    def max_projection(self, row, col, f):
        group = self.files[(self.files['row'] == row) & (self.files['col'] == col) & (self.files['fid'] == f)]
        channels = list()
        for ic, c in group.groupby('ch'):  # iterate over channels
            files = list()
            for iz, z in c.groupby('p'):  # iterate over z positions
                if z.index.size == 1:
                    files.append(self.filename(z))
            first_file = os.path.join(self.images_path, files.pop())
            try:
                max = io.imread(first_file)
                for f in files:
                    fname = os.path.join(self.images_path, f)
                    img = io.imread(fname)
                    max = np.maximum(max, img)
                channels.append(max)
            except Exception as e:
                logger.error(e)

        return channels

    def measure(self, row, col, f):
        pass

    def save_path(self, filename, subdir=''):
        exp = self.name if self.name is not None else ''
        return ensure_dir(os.path.join(self.base_path, 'out', exp, subdir, filename))

    @staticmethod
    def generate_images_structure(base_folder):
        #  build a list of dicts for every image file in the directory
        l = list()
        images_path = os.path.join(base_folder, 'Images')
        for root, directories, filenames in os.walk(images_path):
            for filename in filenames:
                ext = filename.split('.')[-1]
                if ext == 'tiff':
                    _row, _col, f, p, ch, sk, fk, fl = [int(g) for g in re.search(
                        'r([0-9]+)c([0-9]+)f([0-9]+)p([0-9]+)-ch([0-9]+)sk([0-9]+)fk([0-9]+)fl([0-9]+).tiff',
                        filename).groups()]
                    i = {'row': _row, 'col': _col, 'fid': f, 'p': p, 'ch': ch, 'sk': sk, 'fk': fk, 'fl': fl}
                    l.append(i)
        f = pd.DataFrame(l)

        return f
