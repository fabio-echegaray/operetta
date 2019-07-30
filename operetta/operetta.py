import os
import xml.etree.ElementTree
import json
import re
import matplotlib

matplotlib.use("agg")
import numpy as np
import pandas as pd
from skimage import io
import skimage.exposure as exposure

from .exceptions import ImagesFolderNotFound
from . import logger
from gui import convert_to, meter, pix, um


def ensure_dir(file_path):
    file_path = os.path.abspath(file_path)
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
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
        generate_structure = False
        if not os.path.exists(csv_path):
            logger.warning("File operetta.csv is missing in the folder structure, generating it now.\r\n"
                           "\tA new folder with the name 'out' will be created. You can safely delete this\r\n"
                           "\tfolder if you don't want any of the analisys output from this tool.\r\n")
            generate_structure = True
            self._generate_directory_structure()
        else:
            self.files = pd.read_csv(csv_path)

        if row is not None:
            logger.debug('original rows: %s' % self.rows())
            logger.debug('restricting rows to %s' % row)
            self.files = self.files[self.files['row'] == row]
            logger.debug('rows: %s' % self.files['row'].unique())
        if col is not None:
            logger.debug('original columns: %s' % self.columns())
            logger.debug('restricting columns to %s' % col)
            self.files = self.files[self.files['col'] == col]
            logger.debug('columns: %s' % self.files['col'].unique())
        self.files_gr = self.files.groupby(['row', 'col', 'fid']).size().reset_index()
        logger.info("%d image stacks available." % len(self.files_gr))

        self._layout = None
        self.flatfield_profiles = None

        if generate_structure and os.path.exists(self.images_path):
            self._generate_sample_image()
        else:
            raise ImagesFolderNotFound('Images folder is not in the structure.')

    @staticmethod
    def filename(row):
        assert row.index.size == 1, 'only 1 row accepted'
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

    @property
    def um_per_pix(self):
        um_per_pix = self.files['um_per_pix'].unique()
        assert len(um_per_pix) == 1, 'different resolutions for dataset'
        return float(um_per_pix[0])

    @property
    def pix_per_um(self):
        pix_per_um = self.files['pix_per_um'].unique()
        assert len(pix_per_um) == 1, 'different resolutions for dataset'
        return float(pix_per_um[0])

    def stack_generator(self):
        l = len(self.files_gr)
        for ig, g in self.files_gr.iterrows():
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

        elif len(args) == 3 and np.all([np.issubdtype(type(a), np.integer) for a in args]):
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
                max_img = io.imread(first_file)
                for f in files:
                    fname = os.path.join(self.images_path, f)
                    img = io.imread(fname)
                    max_img = np.maximum(max_img, img)

                channels.append(max_img)
            except Exception as e:
                logger.error(e)

        assert len(channels) > 0, 'no images out of max projection'
        return channels

    def image(self, row=0, col=0, fid=0, zpos=0):
        f = self.files[(self.files['row'] == row) & (self.files['col'] == col) &
                       (self.files['fid'] == fid) & (self.files['p'] == zpos)]

        channels = list()
        for ic, c in f.groupby('ch'):  # iterate over channels
            try:
                img = io.imread(os.path.join(self.images_path, self.filename(c)))
                channels.append(img)
            except Exception as e:
                logger.error(e)

        assert len(channels) > 0, 'no images out extracted'
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
        for key in ['row', 'col', 'fid', 'ch', 'tid']:
            f.loc[:, key] = f[key].astype('int32')
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

    def _generate_sample_image(self):
        from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
        from matplotlib.figure import Figure
        from gui.utils import canvas_to_pil

        logger.debug("Generating sample image...")
        cfg_ch = self.files.groupby(['ch', 'ChannelName']).size().reset_index().drop(0, axis=1)
        max_ch = cfg_ch['ch'].max()
        images = self.max_projection(np.random.randint(len(self.files_gr)))

        width, height = images[0].shape
        w_um, h_um = [s * self.um_per_pix for s in images[0].shape]
        fig = Figure((width * 4 / 150, width * 4 / 150), dpi=150)
        canvas_g = FigureCanvas(fig)

        for i in range(4):
            images = self.max_projection(np.random.randint(len(self.files_gr)))
            images = [exposure.equalize_hist(im) for im in images]

            for im, ch, title in zip(images, cfg_ch['ch'], cfg_ch['ChannelName']):
                sp = i * max_ch + ch
                ax = fig.add_subplot(4, max_ch, sp)
                ax.imshow(im, extent=[0, w_um, h_um, 0], cmap='gray')
                ax.set_xlim([0, w_um])
                ax.set_ylim([0, h_um])
                ax.set_axis_off()
                ax.set_title(title)

        pil = canvas_to_pil(canvas_g)
        fpath = os.path.abspath(os.path.join(self.base_path, 'out', 'sample.jpg'))
        pil.save(ensure_dir(fpath))

    def flat_field_profile(self):
        for root, directories, filenames in os.walk(self.ffc_path):
            for filename in filenames:
                ext = filename.split('.')[-1]
                if ext == 'xml':
                    e = xml.etree.ElementTree.parse(os.path.join(root, filename)).getroot()
                    map = [i for i in e if i.tag[-3:] == 'Map'][0]

                    self.flatfield_profiles = list()
                    regex = re.compile(r"([a-zA-Z]+[ ]?[0-9]*)")
                    for p in map:
                        for ffp in p:
                            txt = ffp.text
                            txt = regex.sub(r'"\1"', txt)
                            txt = txt.replace('"Acapella":2013', '"Acapella:2013"')
                            self.flatfield_profiles.append(json.loads(txt))

                    for profile in self.flatfield_profiles:
                        for plane in ['Background', 'Foreground']:
                            if not plane in profile: continue
                            if profile[plane]['Profile']['Type'] != 'Polynomial': continue
                            dims = profile[plane]['Profile']['Dims']
                            # ori = profile[plane]['Profile']['Origin']
                            # sca = profile[plane]['Profile']['Scale']
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
        if self.flatfield_profiles is None and os.path.exists(self.ffc_path):
            self.flat_field_profile()

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

    def _generate_directory_structure(self):
        self.files = self.generate_images_structure(self.base_path)
        op_csv = ensure_dir(os.path.join(self.base_path, 'out', 'operetta.csv'))
        self.files.to_csv(op_csv, index=False)
