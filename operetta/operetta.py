import logging
import os
import sys
import re

import numpy as np
import pandas as pd
from skimage import io

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
    def __init__(self, folder, row=None, col=None, name=None):
        l = list()
        self.folder = folder
        self.dir = os.path.join(folder, 'Images')
        if not os.path.exists(self.dir):
            raise FileNotFoundError('Images folder is not in the structure.')

        #  build a list of dicts for every image file in the directory
        for root, directories, filenames in os.walk(folder):
            for filename in filenames:
                ext = filename.split('.')[-1]
                if ext == 'tiff':
                    # joinf = os.path.join(root, filename)
                    _row, _col, f, p, ch, sk, fk, fl = [int(g) for g in re.search(
                        'r([0-9]+)c([0-9]+)f([0-9]+)p([0-9]+)-ch([0-9]+)sk([0-9]+)fk([0-9]+)fl([0-9]+).tiff',
                        filename).groups()]
                    i = {'row': _row, 'col': _col, 'f': f, 'p': p, 'ch': ch, 'sk': sk, 'fk': fk, 'fl': fl}
                    l.append(i)
        f = pd.DataFrame(l)
        logger.info('original rows: %s' % f['row'].unique())
        logger.info('original columns: %s' % f['col'].unique())
        if row is not None:
            logger.debug('restricting rows to %s' % row)
            f = f[f['row'] == row]
        if col is not None:
            logger.debug('restricting columns to %s' % col)
            f = f[f['col'] == col]
        logger.info('rows: %s' % f['row'].unique())
        logger.info('columns: %s' % f['col'].unique())
        self.files = f
        self.name = name

        self.um_per_pix = convert_to(1.8983367649421008E-07 * meter / pix, um / pix).n()
        self.pix_per_um = 1 / self.um_per_pix
        self.pix_per_um = float(self.pix_per_um.args[0])

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
            row, col, fid = ig
            # if not (row == 1 and col == 9 and fid == 72): continue
            logger.info('stack generator: retrieving %d of %d - row=%d col=%d fid=%d' % (k, l, row, col, fid))
            yield ig

    def add_mesurement(self, row, col, f, name, value):
        self.files.loc[(self.files['row'] == row) & (self.files['col'] == col) & (self.files['f'] == f), name] = value

    def max_projection(self, row, col, f):
        group = self.files[(self.files['row'] == row) & (self.files['col'] == col) & (self.files['f'] == f)]
        channels = list()
        for ic, c in group.groupby('ch'):  # iterate over channels
            files = list()
            for iz, z in c.groupby('p'):  # iterate over z positions
                if z.index.size == 1:
                    files.append(self.filename(z))
            first_file = os.path.join(self.dir, files.pop())
            try:
                max = io.imread(first_file)
                for f in files:
                    fname = os.path.join(self.dir, f)
                    img = io.imread(fname)
                    max = np.maximum(max, img)
                channels.append(max)
            except Exception as e:
                logger.error(e)

        return channels

    def save_path(self, file=None):
        if not self.files.empty:
            if self.name is not None:
                exp = self.name
            else:
                exp = ''
            return ensure_dir(os.path.join(self.folder, 'out', exp, file))
        else:
            return None

    def save(self):
        self.files.to_csv(os.path.join(self.save_path(), 'operetta.csv'))
