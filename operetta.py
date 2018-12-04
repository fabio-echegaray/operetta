import os
import re

import numpy as np
import pandas as pd
from skimage import color
from skimage import transform
from skimage import exposure
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
        l = len(group)
        for k, (ig, g) in enumerate(group):
            print('stack generator: retrieving %d of %d' % (k, l))
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

        red_multiplier = [1, 0, 0]
        green_multiplier = [0, 1, 0]
        blue_multiplier = [0, 0, 1]
        out = hoechst * blue_multiplier * 0.6 + tubulin * green_multiplier * 0.2 + pericentrin * red_multiplier * 0.2

        io.imsave('/Volumes/Kidbeat/data/'
                  'centr-dist(u2os)__2018-11-27T18_08_10-Measurement 1/'
                  'render/%d-%d-%d.jpg' % (fid, row, col),
                  out)

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
