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

        ensure_dir(os.path.join(folder, 'out', 'render', 'nil'))
        self.render_path = os.path.join(folder, 'out', 'render')

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

        self.um_per_pix = None
        self.pix_per_um = None
        self._layout = None

        self.flatfield_profiles = None
        if os.path.exists(self.ffc_path):
            self.flat_field_profile()

        if not os.path.exists(self.images_path):
            raise ImagesFolderNotFound('Images folder is not in the structure.')

    @staticmethod
    def filename(row):
        if row.index.size > 1: raise Exception('only 1 row accepted.')
        return row['filename'].values[0]

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
        um_per_pix = group['um_per_pix'].unique()
        pix_per_um = group['pix_per_um'].unique()
        assert len(um_per_pix) == 1 and len(pix_per_um) == 1, 'different resolutions for projection'
        self.um_per_pix = um_per_pix[0]
        self.pix_per_um = pix_per_um[0]

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
        images_path = os.path.join(base_folder, 'Images')
        ns = {'d': 'http://www.perkinelmer.com/PEHH/HarmonyV5'}
        e = xml.etree.ElementTree.parse(os.path.join(images_path, 'Index.idx.xml')).getroot()
        ims = e.find('d:Images', ns).findall('d:Image', ns)
        list_of_dicts_of_img = [{x.tag.replace('{%s}' % ns['d'], ''): x.text for x in im} for im in ims]
        f = (
            pd.DataFrame(list_of_dicts_of_img)
                .rename(index=str,
                        columns={"Row": "row", "Col": "col", "FieldID": "fid", "PlaneID": "p", "TimepointID": "tid",
                                 "ChannelID": "ch", "FlimID": "fl", "URL": "filename", })
                .drop(['AbsPositionZ', 'AbsTime', 'AcquisitionType', 'BinningX', 'BinningY', 'CameraType',
                       'ChannelType', 'ExposureTime', 'IlluminationType', 'ImageSizeX', 'ImageSizeY', 'ImageType',
                       'MainEmissionWavelength', 'MainExcitationWavelength', 'MaxIntensity', 'MeasurementTimeOffset',
                       'OrientationMatrix', 'PositionX', 'PositionY', 'PositionZ', 'State'], axis=1)
        )

        assert (f['ImageResolutionX'] == f['ImageResolutionY']).all(), 'some pixels are not square'

        f.loc[:, 'ImageResolutionX'] = f['ImageResolutionX'].apply(pd.to_numeric, errors='ignore')
        f.loc[:, 'um_per_pix'] = f['ImageResolutionX'].apply(
            lambda rx: convert_to(rx * meter / pix, um / pix).n().args[0])
        f.loc[:, 'pix_per_um'] = 1 / f['um_per_pix']
        f.drop(['ImageResolutionX', 'ImageResolutionY'], axis=1, inplace=True)
        return f[['row', 'col', 'fid', 'p', 'ch', 'tid', 'fl', 'ChannelName', 'um_per_pix', 'pix_per_um',
                  'ObjectiveMagnification', 'ObjectiveNA', 'filename', 'id', ]]

    @property
    def layout(self):
        """ Locates the assay layout xml file in the folder structure, and constructs a pandas dataframe from it. """
        if self._layout is not None: return self._layout

        _xmld = "{http://www.perkinelmer.com/PEHH/HarmonyV5}"
        ns = {'d': 'http://www.perkinelmer.com/PEHH/HarmonyV5'}
        for root, directories, filenames in os.walk(self.assay_layout_path):
            for filename in filenames:
                ext = filename.split('.')[-1]
                if ext == 'xml':
                    e = xml.etree.ElementTree.parse(os.path.join(root, filename)).getroot()

                    cols = int(e.find('d:PlateCols', ns).text)
                    rows = int(e.find('d:PlateRows', ns).text)
                    df = (  # this creates cartesian product of cols & rows
                        pd.DataFrame({'row': range(1, rows + 1)}).assign(key=1)
                            .merge(pd.DataFrame({'col': range(1, cols + 1)}).assign(key=1), on='key')
                            .drop('key', axis=1)
                    )

                    for ly in e.findall('d:Layer', ns):
                        ly_name = ly.find('d:Name', ns).text
                        # ly_type = ly.find('d:ValueType', ns).text
                        for w in ly.findall('d:Well', ns):
                            row = int(w.find('d:Row', ns).text)
                            col = int(w.find('d:Col', ns).text)
                            val = w.find('d:Value', ns).text
                            df.loc[(df['row'] == row) & (df['col'] == col), ly_name] = val

                    self._layout = df.apply(pd.to_numeric, errors='ignore')
                    break

        logger.info("layout succesfully extracted from xml file.")
        return self._layout

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

        logger.info("flat field correction calculated from coefficients.")

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
