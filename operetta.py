import os
import re

import numpy as np
import pandas as pd
from skimage import io


class Montage:
    def __init__(self, dir):
        l = list()
        self.dir = dir
        #  build a list of dicts for every image file in the directory
        for root, directories, filenames in os.walk(dir):
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
        for ig, g in group:
            yield ig

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
