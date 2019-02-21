import os
import xml.etree.ElementTree
import json
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
        self.assay_layout_path = os.path.join(folder, 'Assaylayout')
        self.ffc_path = os.path.join(folder, 'FFC_Profile')

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

        self.flatfield_profiles = None
        if os.path.exists(self.ffc_path):
            self.flat_field_profile()

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

    def max_projection(self, *args):
        if len(args) == 1 and isinstance(args[0], int):
            id = args[0]
            r = self.files_gr.ix[id - 1]
            row, col, fid = r['row'], r['col'], r['fid']
            logger.debug('retrieving image id=%d row=%d col=%d fid=%d' % (id, row, col, fid))
            return self._max_projection_row_col_fid(row, col, fid)

        elif len(args) == 3 and np.all([np.issubdtype(a, np.integer) for a in args]):
            row, col, fid = args[0], args[1], args[2]
            return self._max_projection_row_col_fid(row, col, fid)

    def _max_projection_row_col_fid(self, row, col, f):
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

                channels.append(self._flat_field_correct(max, ic))
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

    def flat_field_profile(self):
        for root, directories, filenames in os.walk(self.ffc_path):
            for filename in filenames:
                ext = filename.split('.')[-1]
                if ext == 'xml':
                    e = xml.etree.ElementTree.parse(os.path.join(root, filename)).getroot()
                    map = [i for i in e if i.tag[-3:] == 'Map'][0]

                    self.flatfield_profiles = list()
                    # regex1 = re.compile(r"{([a-zA-Z]+): ")
                    # regex2 = re.compile(r", ([a-zA-Z]+): ")
                    regex = re.compile(r"([a-zA-Z]+[ ]?[0-9]*)")
                    for p in map:
                        # info = {'ChannelId': p.attrib['ChannelID']}
                        for ffp in p:
                            txt = ffp.text
                            # txt = regex1.sub(r'{"\1": ', txt)
                            # txt = regex2.sub(r', "\1": ', txt)
                            txt = regex.sub(r'"\1"', txt)
                            txt = txt.replace('"Acapella":2013', '"Acapella:2013"')
                            self.flatfield_profiles.append(json.loads(txt))

                    for profile in self.flatfield_profiles:
                        for plane in ['Background', 'Foreground']:
                            if profile[plane]['Profile']['Type'] != 'Polynomial': continue
                            dims = profile[plane]['Profile']['Dims']
                            ori = profile[plane]['Profile']['Origin']
                            sca = profile[plane]['Profile']['Scale']
                            # ones = np.multiply(dims, sca)
                            # x = (np.arange(0, dims[0], 1) - ori[0]) * sca[0]
                            # y = (np.arange(0, dims[1], 1) - ori[1]) * sca[1]
                            x = np.linspace(-0.5, 0.5, num=dims[0])
                            y = np.linspace(-0.5, 0.5, num=dims[1])
                            xx, yy = np.meshgrid(x, y)
                            z = np.zeros(dims, dtype=np.float32)

                            coeficients = profile[plane]['Profile']['Coefficients']
                            for c in coeficients:
                                n = np.polyval(c, xx) + np.polyval(c, yy)
                                z = n + z
                            profile[plane]['Profile']['Image'] = z

        logger.info("flat field correction calculated from coefficients")

    def _flat_field_correct(self, img, channel):
        if self.flatfield_profiles is not None:
            for profile in self.flatfield_profiles:
                if profile['Channel'] != channel: continue
                br = profile['Foreground']['Profile']['Image']
                dk = profile['Background']['Profile']['Image']

                br_dk = br - dk
                mean_bd = np.mean(br_dk)
                gain = br_dk / mean_bd
                img = np.divide((img - dk), gain).astype(np.int16)

        return img
