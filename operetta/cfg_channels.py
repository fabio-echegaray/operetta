import os
import ast
import configparser

from skimage import color
from skimage import exposure
from skimage import draw
import shapely.wkt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from shapely.geometry.point import Point
from shapely import affinity
from shapely.wkt import dumps
from shapely.geometry import LineString, MultiLineString, Polygon

from plots import utils as p
import measurements as m
from .exceptions import ImagesFolderNotFound, NoSamplesError
from . import Montage, ensure_dir, logger
from gui.utils import canvas_to_pil

_hist_edges_arr_ops = {"precision": 2, "separator": ",", "suppress_small": True,
                       "formatter": {"float": lambda x: "%0.2f" % x}}


class ConfiguredChannels(Montage):

    def __init__(self, base_path, row=None, col=None, condition_name=None):
        logger.info('Initializing FourChannels object with pandas file from folder %s' % base_path)
        try:
            super().__init__(base_path, row=row, col=col, condition_name=condition_name)
        except ImagesFolderNotFound:
            logger.warning("no images folder found.")
            self.images_path = os.path.join(base_path, 'Images')

        self._samples = None
        self._cfg = None
        self._row = None
        self._col = None
        self._fid = None
        self._zp = None
        self._mdf = pd.DataFrame()  # here we store all measurements

    def samples_are_well_formed(self, raise_exception=False):
        if np.any([i not in self.samples for i in ['row', 'col', 'fid']]):
            is_row, is_col, is_fid = [i in self.samples for i in ['row', 'col', 'fid']]
            _txt = 'key columns not in dataframe row=%s col=%s fid=%s' % (is_row, is_col, is_fid)
            if raise_exception:
                raise NoSamplesError(_txt)
            else:
                logger.warning(_txt)

            return False
        return True

    @property
    def samples(self):
        if self._samples is not None: return self._samples
        pd_path = os.path.join(self.base_path, 'out', 'nuclei.pandas')
        if os.path.exists(pd_path):
            self._samples = pd.read_pickle(pd_path)
            if self.samples_are_well_formed(raise_exception=True): pass
        else:
            logger.warning('no pandas file found.')
        return self._samples

    @staticmethod
    def filename_of_render(row, basepath, ext='jpg'):
        name = 'r%d-c%d-i%d.%s' % (row['row'], row['col'], row['fid'], ext)
        if os.path.basename(basepath) == 'render':
            path = os.path.abspath(os.path.join(basepath, name))
        else:
            path = os.path.abspath(os.path.join(basepath, 'render', name))
        return name, path

    @staticmethod
    def _load_cfg(path):
        with open(path, 'r') as configfile:
            logger.info('loading configuration from %s' % path)
            config = configparser.ConfigParser()
            config.read_file(configfile)

            section = 'General'
            if config.has_section(section):
                channels = config.getint(section, 'channels')

                ch = list()
                for c in range(1, channels + 1):
                    section = 'Channel %d' % c
                    if config.has_section(section):
                        ch.append({
                            'number': config.getint(section, 'number'),
                            'channel name': config.get(section, 'channel name'),
                            'z_stack_aggregation': config.get(section, 'z_stack_aggregation',
                                                              fallback='do a max projection'),
                            'tag': config.get(section, 'tag'),
                            'pipeline': ast.literal_eval(config.get(section, 'pipeline')),
                            'rng_thickness': config.getfloat(section, 'rng_thickness', fallback=0),
                            'n_lines': config.getint(section, 'n_lines', fallback=3),
                            'hist_bins': config.getint(section, 'hist_bins', fallback=10),
                            'hist_min': config.getint(section, 'hist_min', fallback=0),
                            'hist_max': config.getint(section, 'hist_max', fallback=np.iinfo(np.uint16).max),
                            'hist_log': config.getboolean(section, 'hist_log', fallback=False),
                            'render': config.getboolean(section, 'render'),
                            'render intensity': config.getfloat(section, 'render intensity'),
                            'flatfield': config.getboolean(section, 'flat field correction')
                        })
                return {'channels': ch}

    def _save_cfg(self, path):
        with open(path, 'w') as configfile:
            config = configparser.RawConfigParser(allow_no_value=True)
            config.add_section('Information')
            config.set('Information', '#')
            config.set('Information', '# allowed functions for the pipeline are: nucleus, cell, histogram, ')
            config.set('Information', '#        intensity_in_nucleus, ring_around_nucleus, particle_in_cytoplasm,')
            config.set('Information', '#        line_intensity_ring, histogram_of_nucleus, histogram_of_ring.')
            config.set('Information', '#')
            config.set('Information', '#')
            config.set('Information', '# Accepted parameters:')
            config.set('Information', '#     for z_stack_aggregation on every Channel: Aggregation op to do on ')
            config.set('Information', '#     the z-stack. Can be one of the following:')
            config.set('Information', '#         use each image individually')
            config.set('Information', '#         do a max projection')
            config.set('Information', '#')
            config.set('Information', '#     for ring_around_nucleus:')
            config.set('Information', '#         rng_thickness: Thickness of the ring, in um.')
            config.set('Information', '#')
            config.set('Information', '#     for histogram:')
            config.set('Information', '#         hist_bins: Number of bins to use.')
            config.set('Information', '#         hist_min: Lower boundary for bins.')
            config.set('Information', '#         hist_max: Upper boundary for bins.')
            config.set('Information', '#         hist_log: Makes bin to look equally sized on a log scale.')
            config.set('Information', '#')
            config.set('Information', '#     for line_intensity_ring:')
            config.set('Information', '#         n_lines: Number of lines to measure.')
            config.set('Information', '#')
            config.set('Information', '#')

            config.add_section('General')
            config.set('General', 'Version', 'v0.2')
            config.set('General', 'channels', len(self.channels))

            for i, c in enumerate(sorted(self.channels)):
                section = 'Channel %d' % c
                config.add_section(section)
                config.set(section, 'number', c - 1)
                config.set(section, 'channel name', self.files.loc[self.files['ch'] == c, 'ChannelName'].iloc[0])
                config.set(section, 'tag', 'default')
                config.set(section, 'pipeline', [])
                config.set(section, 'z_stack_aggregation', 'do a max projection')
                config.set(section, 'rng_thickness', 3)
                config.set(section, 'n_lines', 3)
                config.set(section, 'hist_bins', 100)
                config.set(section, 'hist_min', 0)
                config.set(section, 'hist_max', np.iinfo(np.uint16).max)
                config.set(section, 'hist_log', True)
                config.set(section, 'render', False)
                config.set(section, 'render intensity', 0.1)
                config.set(section, 'flat field correction', True)

            config.write(configfile)

    @property
    def configuration(self):
        """ Reads (or creates if non existent) the configuration file for the dataset """
        if self._cfg is not None: return self._cfg
        cfg_path = os.path.join(self.base_path, 'out', 'operetta.cfg')
        if not os.path.exists(cfg_path):
            self._save_cfg(cfg_path)
        self._cfg = self._load_cfg(cfg_path)
        return self._cfg

    def save_render(self, *args, **kwargs):
        assert self.samples is not None, 'pandas samples file is needed to use this function'
        if len(args) == 1 and isinstance(args[0], int):
            _id = args[0]
            if not self.samples_are_well_formed():
                logger.warning("couldn't render because there was either an incorrect measurement or none a all")
                return
            r = self.files_gr.ix[_id - 1]
            row, col, fid = r['row'], r['col'], r['fid']
            logger.debug('rendering id=%d row=%d col=%d fid=%d' % (_id, row, col, fid))

        elif len(args) == 3 and np.all([np.issubdtype(a, np.integer) for a in args]):
            row, col, fid = args[0], args[1], args[2]
        else:
            logger.warning("there's nothing to render")
            return

        s = self.samples
        dfi = s[(s['row'] == row) & (s['col'] == col) & (s['fid'] == fid)]

        # create path where the images are going to be stored
        if 'path' not in kwargs:
            basepath = os.path.dirname(self.images_path)
            basepath = os.path.abspath(os.path.join(basepath, 'out', 'render'))
        else:
            basepath = os.path.abspath(kwargs['path'])

        cfg_ch = self.configuration['channels']

        # canvas for overall image
        max_width = kwargs['max_width'] if 'max_width' in kwargs else None
        fig_general = Figure((max_width * 4 / 150, max_width * 4 / 150), dpi=150)
        canvas_g = FigureCanvas(fig_general)
        # canvas for closeup image
        fig_closeup = Figure((max_width / 150, max_width / 150), dpi=150)
        canvas_c = FigureCanvas(fig_closeup)

        for _ip, dpos in dfi.groupby('p'):
            # get images for render
            if _ip in ["max"]:  # ,"min","avg"]:
                images = self.max_projection(row, col, fid)
            else:
                images = self.image(row, col, fid, _ip)

            # ----------------------
            #      BACKGROUND IMAGE
            # ----------------------
            background = np.zeros(images[0].shape + (3,), dtype=np.float64)
            for ch in cfg_ch:
                _img = images[ch['number']]
                if ch['flatfield']:
                    _img = self._flat_field_correct(_img, ch['number'])
                _img = exposure.equalize_hist(_img)
                _img = color.gray2rgb(_img)

                for operation in ch['pipeline']:
                    if operation == 'nucleus' and ch['render']:
                        background += _img * p.colors.hoechst_33342 * ch['render intensity']
                    if operation == 'cell' and ch['render']:
                        background += _img * p.colors.alexa_488 * ch['render intensity']
                    if operation == 'intensity_in_nucleus' and ch['render']:
                        background += _img * p.colors.alexa_647 * ch['render intensity']
                    if operation == 'ring_around_nucleus' and ch['render']:
                        background += _img * p.colors.alexa_647 * ch['render intensity']
                    if operation == 'particle_in_cytoplasm' and ch['render']:
                        background += _img * p.colors.alexa_594 * ch['render intensity']

            # ----------------------
            #      RENDER GENERAL
            # ----------------------
            self._render_image(fig_general.gca(), dpos, background)
            name = 'r%d-c%d-f%d-p%s.jpg' % (row, col, fid, _ip)
            fpath = os.path.abspath(os.path.join(basepath, name))
            pil = canvas_to_pil(canvas_g)
            pil.save(ensure_dir(fpath))

            # ----------------------
            #      RENDER CLOSEUPS
            # ----------------------
            for ix, smp in dpos.groupby('id'):
                self._render_image_closeup(fig_closeup.gca(), smp, background)
                name = 'r%d-c%d-f%d-p%s-i%d.pdf' % (row, col, fid, _ip, ix)
                fpath = os.path.abspath(os.path.join(basepath, name))
                fig_closeup.savefig(fpath)

    def _render_image(self, axg, df_row, bkg_img):
        w_um, h_um, _ = [s * self.um_per_pix for s in bkg_img.shape]
        frame = Polygon([(0, 0), (0, w_um), (h_um, w_um), (h_um, 0)])
        frx, fry = frame.exterior.xy
        axg.plot(frx, fry, color='red', linewidth=2, solid_capstyle='round', zorder=10)

        for ix, smp in df_row.groupby('id'):
            nucleus = shapely.wkt.loads(smp['nucleus'].iloc[0])
            cell = shapely.wkt.loads(smp['cell'].iloc[0]) if 'cell' in smp and not smp['cell'].isna().iloc[0] else None

            if 'c1' in smp and 'c2' in smp:
                c1 = shapely.wkt.loads(smp['c1'].iloc[0])
                c2 = shapely.wkt.loads(smp['c2'].iloc[0])
                centr = [c1, c2]
            else:
                centr = None

            p.render_cell(nucleus, cell, centr, base_zorder=20, ax=axg)
            if 'ring' in smp and not smp['ring'].isna().iloc[0]:
                ring = shapely.wkt.loads(smp['ring'].iloc[0])
                if ring.area > 0:
                    p.render_polygon(ring, zorder=10, ax=axg)
            axg.text(nucleus.centroid.x + 2, nucleus.centroid.y - 1, ix, color='red', zorder=50)

        axg.plot([5, 5 + 10], [5, 5], c='w', lw=4)
        axg.text(5 + 1, 5 + 1.5, '10 um', color='w')

        axg.imshow(bkg_img, extent=[0, w_um, h_um, 0])
        axg.set_xlim([0, w_um])
        axg.set_ylim([0, h_um])
        axg.set_axis_on()

    def _render_image_closeup(self, axc, smp, bkg_img):
        assert len(smp) == 1, 'only one sample allowed'
        w_um, h_um, _ = [s * self.um_per_pix for s in bkg_img.shape]
        axc.cla()
        axc.set_facecolor('xkcd:salmon')

        nucleus = shapely.wkt.loads(smp['nucleus'].iloc[0])
        cell = shapely.wkt.loads(smp['cell'].iloc[0]) if 'cell' in smp and not smp['cell'].isna().iloc[0] else None

        if 'c1' in smp and 'c2' in smp:
            c1 = shapely.wkt.loads(smp['c1'].iloc[0])
            c2 = shapely.wkt.loads(smp['c2'].iloc[0])
            centr = [c1, c2]
        else:
            centr = None

        # render and save closeup image
        p.render_cell(nucleus, cell, centr, base_zorder=20, ax=axc)
        if 'ring' in smp and not smp['ring'].isna().iloc[0]:
            ring = shapely.wkt.loads(smp['ring'].iloc[0])
            if ring.area > 0:
                p.render_polygon(ring, zorder=10, ax=axc)

            #  draw measured lines
            if "act_int_lines" in smp and not np.isnan(smp['act_int_lines'].iloc[0]):
                n_lines = int(smp['act_int_lines'].iloc[0])
                angle_delta = 2 * np.pi / n_lines
                minx, miny, maxx, maxy = nucleus.bounds
                radius = max(maxx - minx, maxy - miny)
                for angle in [angle_delta * i for i in range(n_lines)]:
                    ray = LineString([nucleus.centroid,
                                      (nucleus.centroid.x + radius * np.cos(angle),
                                       nucleus.centroid.y + radius * np.sin(angle))])
                    r_seg = ray.intersection(nucleus)
                    if r_seg.is_empty:
                        continue
                    if type(r_seg) == MultiLineString:
                        r_seg = r_seg[0]
                    pt = Point(r_seg.coords[-1])

                    for pt0, pt1 in m.pairwise(nucleus.exterior.coords):
                        # if pt.touches(LineString([pt0, pt1])):
                        if Point(pt).distance(LineString([pt0, pt1])) < 1e-6:
                            # compute normal vector
                            dx = pt1[0] - pt0[0]
                            dy = pt1[1] - pt0[1]
                            # touching point of the polygon line segment
                            px, py = pt.x, pt.y
                            # normalize normal vector
                            mag = np.sqrt(dx ** 2 + dy ** 2)
                            dx, dy = dx / mag, dy / mag
                            # get length from ring thickness
                            length = 3

                            # axc.plot(px, py, dx, c='green', marker='o', markersize=2)
                            axc.plot([px, px - dy * length], [py, py + dx * length], c='magenta', lw=1)

                    # draw line from nucleus centroid to point
                    pt_x, pt_y = r_seg.xy
                    axc.plot(list(pt_x), list(pt_y), c='gray', lw=0.5)

                #  draw normals of nucleus polygon
                dx_v, dy_v = np.diff(nucleus.exterior.xy)
                for dx, dy, (px, py) in zip(dx_v, dy_v, nucleus.exterior.coords):
                    x = px + dx / 2
                    y = py + dy / 2
                    # normalize normal vector
                    mag = np.sqrt(dx ** 2 + dy ** 2)
                    dx, dy = dx / mag, dy / mag
                    length = 1.5

                    # axc.plot(x, y, c='yellow', marker='o', markersize=1)
                    axc.arrow(x, y, -dy * length, dx * length, color='blue', head_width=0.5, ls='-', lw=0.1)

        _m = 30
        x0, xf = nucleus.centroid.x - _m, nucleus.centroid.x + _m
        y0, yf = nucleus.centroid.y - _m, nucleus.centroid.y + _m
        if x0 < 0: xf -= x0
        if y0 < 0: yf -= y0
        axc.imshow(bkg_img, extent=[0, w_um, h_um, 0])
        axc.set_xlim(x0, xf)
        axc.set_ylim(y0, yf)
        axc.plot([x0 + 5, x0 + 15], [y0 + 5, y0 + 5], c='w', lw=4)
        axc.text(x0 + 5, y0 + 7, '10 um', color='w', zorder=50)
        if "cluster" in smp:
            axc.text(nucleus.centroid.x, nucleus.centroid.y, smp["cluster"].iloc[0], color='w', zorder=10)

    def _exec_op(self, img, op, cfg_row):
        img = img[cfg_row['number'].iloc[0]]
        assert len(img) > 0, 'no image for measurements'
        if cfg_row['flatfield'].iloc[0]:
            img = self._flat_field_correct(img, cfg_row['number'].iloc[0])
        if not self._mdf.empty:
            self._ix = ((self._mdf['row'] == self._row) &
                        (self._mdf['col'] == self._col) &
                        (self._mdf['fid'] == self._fid) &
                        (self._mdf['p'] == self._zp))
        else:
            self._ix = self._mdf.index
        op(img, cfg=cfg_row)

    def _stack_operation(self, row, col, fid, configuration_row, op):
        assert len(configuration_row) == 1, "only one configuration row allowed"
        z_op = configuration_row['z_stack_aggregation'].iloc[0]
        self._row = row
        self._col = col
        self._fid = fid

        # get stack processing from configuration file and apply it to an image
        if z_op == 'use each image individually':
            logger.info("Processing each image in the stack individually.")
            gr = self.files[(self.files['row'] == row) & (self.files['col'] == col) & (self.files['fid'] == fid)]
            for zp, z in gr.groupby('p'):  # iterate over z positions
                logger.debug("Z stack %s" % zp)
                self._zp = zp
                img = self.image(row, col, fid, zp)
                self._exec_op(img, op, configuration_row)
        elif z_op == 'do a max projection':
            logger.info("Performing a max projection on the stack previous processing.")
            self._zp = "max"
            img = self.max_projection(row, col, fid)
            self._exec_op(img, op, configuration_row)
        else:
            logger.warning("Z_stack aggregation is bad configured. Nothing done for this operation.")
        pass

    def measure(self, *args):
        if len(args) == 1 and isinstance(args[0], int):
            _id = args[0]
            r = self.files_gr.ix[_id - 1]
            row, col, fid = r['row'], r['col'], r['fid']
            logger.debug('measuring id=%d row=%d col=%d fid=%d' % (_id, row, col, fid))

        elif len(args) == 3 and np.all([np.issubdtype(a, np.integer) for a in args]):
            row, col, fid = args[0], args[1], args[2]
        else:
            logger.warning('nothing to save on this image')
            return pd.DataFrame()

        df = self._measure_row_col_fid(row, col, fid)
        name = 'r%d-c%d-f%d.csv' % (row, col, fid)
        df.to_csv(self.save_path(name, subdir='pandas'), index=False)
        if self._samples is None: self._samples = df
        return df

    def _measure_row_col_fid(self, row, col, fid):
        cfg_ch = self.configuration['channels']
        cfg = pd.DataFrame(cfg_ch)
        self._mdf = pd.DataFrame()
        # NOTE: all the extraction process and validation is done on the pixel space, and the final dataframe is in [um]

        # --------------------
        #     Find nuclei
        # --------------------
        s = cfg[cfg['z_stack_aggregation'].apply(lambda l: l == 'use each image individually')]
        zstack_idv = len(s) > 0
        c = cfg[cfg['pipeline'].apply(lambda l: 'nucleus' in l)]
        assert len(c) > 0, 'nuclei finding step needed in configuration'
        assert len(c) < 2, 'only one nuclei finding step per batch allowed'
        if zstack_idv:
            c.loc[:, 'z_stack_aggregation'] = 'use each image individually'
        self._stack_operation(row, col, fid, c, self._measure_nuclei)

        # --------------------
        #     Find cells
        # --------------------
        self.cells_measured = False
        c = cfg[cfg['pipeline'].apply(lambda l: 'cell' in l)]
        assert len(c) < 2, 'only one cell finding step per batch allowed'
        if len(c) == 1:
            self._stack_operation(row, col, fid, c, self._measure_cells)

        # --------------------
        #     Intensity in nucleus
        # --------------------
        c = cfg[cfg['pipeline'].apply(lambda l: 'intensity_in_nucleus' in l)]
        for _, cf in c.iterrows():
            self._stack_operation(row, col, fid, c, self._measure_intensity_in_nucleus)

        # --------------------
        #     Ring around nucleus
        # --------------------
        c = cfg[cfg['pipeline'].apply(lambda l: 'ring_around_nucleus' in l)]
        assert len(c) < 2, 'only one ring measurement step per batch allowed at the moment'
        for _, cf in c.iterrows():
            self._stack_operation(row, col, fid, c, self._measure_ring_intensity_around_nucleus)

        # --------------------
        #     Particle in cytoplasm
        # --------------------
        c = cfg[cfg['pipeline'].apply(lambda l: 'particle_in_cytoplasm' in l)]
        # assert len(c) == 0, 'cell finding step needed in configuration'
        assert len(c) < 2, 'only one particle_in_cytoplasm step per batch allowed'
        if len(c) == 1:
            assert self.cells_measured, 'particle_in_cytoplasm needs cell data'
            self._stack_operation(row, col, fid, c, self._measure_particle_in_cytoplasm)

        # --------------------
        #     Histogram
        # --------------------
        c = cfg[cfg['pipeline'].apply(lambda l: 'histogram' in l)]
        for _, cf in c.iterrows():
            self._stack_operation(row, col, fid, c, self._measure_histogram)

        c = cfg[cfg['pipeline'].apply(lambda l: 'histogram_of_nucleus' in l)]
        for _, cf in c.iterrows():
            self._stack_operation(row, col, fid, c, self._measure_histogram_on_nucleus)

        c = cfg[cfg['pipeline'].apply(lambda l: 'histogram_of_ring' in l)]
        for _, cf in c.iterrows():
            self._stack_operation(row, col, fid, c, self._measure_histogram_on_ring)

        # --------------------
        #     Line intensity
        # --------------------
        c = cfg[cfg['pipeline'].apply(lambda l: 'line_intensity_ring' in l)]
        for _, cf in c.iterrows():
            self._stack_operation(row, col, fid, c, self._measure_line_intensity)

        # --------------------
        #     Re label id column if analizing every image in the z-stack
        # --------------------
        if zstack_idv:
            self._relabel_id()

        return self._mdf

    def _measure_nuclei(self, nuclei_img, cfg, r=10):
        imgseg, nuclei = m.nuclei_segmentation(nuclei_img, radius=r * self.pix_per_um)
        nuclei = m.exclude_contained(nuclei)

        if len(nuclei) == 0:
            logger.warning("Couldn't find nuclei in image.")
            return

        for nucleus in nuclei:
            nucl_bnd = (nucleus['boundary']
                        .buffer(self.pix_per_um, join_style=1)
                        .buffer(-self.pix_per_um, join_style=1)
                        .simplify(self.pix_per_um / 2, preserve_topology=True)
                        )
            if nucl_bnd.is_empty: continue
            logger.debug("previous simplify %d, after %d" %
                         (len(nucleus['boundary'].exterior.coords), len(nucl_bnd.exterior.coords)))
            if nucl_bnd.area < np.pi * (3 * self.pix_per_um) ** 2: continue
            dna_int = m.integral_over_surface(nuclei_img, nucl_bnd)

            # convert everything to um space for dataframe construction
            n_bum = affinity.scale(nucl_bnd, xfact=self.um_per_pix, yfact=self.um_per_pix, origin=(0, 0, 0))

            # TODO: Add units support
            d = pd.DataFrame(data={
                'id': [nucleus['id']],
                'row': [self._row],
                'col': [self._col],
                'fid': [self._fid],
                'p': [self._zp],
                '%s_int' % cfg['tag'].iloc[0]: [int(dna_int)],
                '%s_dens' % cfg['tag'].iloc[0]: [int(dna_int / nucl_bnd.area)],
                'nucleus': dumps(n_bum, rounding_precision=4),
                'nuc_pix': dumps(nucl_bnd, rounding_precision=1),
            })
            self._mdf = self._mdf.append(d, ignore_index=True, sort=False)
        logger.debug("%d nuclei found in image" % len(nuclei))

        return self._mdf.index

    def _measure_cells(self, cell_img, cfg):
        assert self._ix.any(), "no rows in the filtered dataframe"
        # generate an image based on nuclei found previously
        nuclei_img = np.zeros(cell_img.shape, dtype=np.bool)
        nuclei = list()
        for _id, nuc in self._mdf.set_index("id").loc[self._ix, "nuc_pix"].iteritems():
            _nimg = m.generate_mask_from(nuc, cell_img.shape)
            nuclei_img = nuclei_img | _nimg
            nuclei.append({"id": _id, "boundary": shapely.wkt.loads(nuc)})
        cells, cells_mask = m.cell_boundary(cell_img, nuclei_img)

        #  filter polygons contained in others
        cells = m.exclude_contained(cells)
        logger.debug("%d cells found in image" % len(cells))

        width, height = cell_img.shape
        frame = Polygon([(0, 0), (0, height), (width, height), (width, 0)])
        touching_fr = too_big = no_nuclei = two_nuclei = 0
        # iterate through all cells
        for cl in cells:
            logger.debug("processing cell id %d" % cl['id'])
            cell_bnd = (cl['boundary']
                        .buffer(self.pix_per_um, join_style=1)
                        .buffer(-self.pix_per_um, join_style=1)
                        .simplify(self.pix_per_um / 2, preserve_topology=True)
                        )
            logger.debug("previous simplify %d, after %d" %
                         (len(cl['boundary'].exterior.coords), len(cell_bnd.exterior.coords)))

            for _id, nucleus in self._mdf.loc[self._ix, "nuc_pix"]:
                valid_sample, reason = m.is_valid_sample(frame, cell_bnd, nucleus, nuclei)
                if reason == m.REJECTION_TOUCHING_FRAME: touching_fr += 1
                if reason == m.REJECTION_NO_NUCLEUS: no_nuclei += 1
                if reason == m.REJECTION_TWO_NUCLEI: two_nuclei += 1
                if reason == m.REJECTION_CELL_TOO_BIG: too_big += 1
                if valid_sample:
                    tubulin_int = m.integral_over_surface(cell_img, cell_bnd)
                    tub_density = tubulin_int / cell_bnd.area
                    if tub_density < 150:
                        logger.warning(
                            "Sample rejected after validation because it had a low tubulin density.")
                        logger.debug('tubulin density in cell: %0.2f, intensity %0.2f, area %0.2f' % (
                            tub_density, tubulin_int, cell_bnd.area))
                        continue

                    # convert everything to um space for dataframe construction
                    c_bum = affinity.scale(cell_bnd, xfact=self.um_per_pix, yfact=self.um_per_pix, origin=(0, 0, 0))

                    # TODO: Add units support
                    ix = self._ix & (self._mdf['id'] == nucleus['id'])
                    self._mdf.loc[ix, 'tubulin_int'] = int(tubulin_int)
                    self._mdf.loc[ix, 'tubulin_dens'] = int(tubulin_int / cell_bnd.area)
                    self._mdf.loc[ix, 'cell'] = dumps(c_bum, rounding_precision=4)
                    self._mdf.loc[ix, 'cell_pix'] = dumps(cell_bnd, rounding_precision=1)

        logger.info("%d samples rejected because they were touching the frame" % touching_fr)
        logger.info("%d samples rejected because cell didn't have a nucleus" % no_nuclei)
        logger.info("%d samples rejected because cell had more than two nuclei" % two_nuclei)
        logger.info("%d samples rejected because cell area was too big" % too_big)
        if (~self._mdf.loc[self._ix, "cell"].isna()).any():
            self.cells_measured = True

    def _measure_intensity_in_nucleus(self, image, cfg):
        assert self._ix.any(), "no rows in the filtered dataframe"

        nuclei = list()
        for _id, nuc in self._mdf.loc[self._ix, "nuc_pix"].set_index("id").iteritems():
            nuclei.append({"id": _id, "boundary": shapely.wkt.loads(nuc)})

        for ix, row in self._mdf[self._ix].iterrows():
            _id = row["id"]
            nucl_bnd = shapely.wkt.loads(row["nuc_pix"])

            logger.debug("intensity_in_nucleus for nucleus id %d" % _id)
            signal_int = m.integral_over_surface(image, nucl_bnd)
            signal_density = signal_int / nucl_bnd.area

            # TODO: scale intensity from pixels^2 to um^2
            self._mdf.loc[ix, '%s_int' % cfg['tag'].iloc[0]] = int(signal_int)
            self._mdf.loc[ix, '%s_dens' % cfg['tag'].iloc[0]] = int(signal_density)

    def _measure_ring_intensity_around_nucleus(self, image, cfg):
        assert self._ix.any(), "no rows in the filtered dataframe"
        for ix, row in self._mdf[self._ix].iterrows():
            nucl_bnd = shapely.wkt.loads(row["nuc_pix"])
            thickness = float(cfg['rng_thickness'])
            thickness *= self.pix_per_um
            rng_bnd = (nucl_bnd
                       .buffer(thickness)
                       .difference(nucl_bnd)
                       .simplify(self.pix_per_um / 2, preserve_topology=True)
                       )
            if rng_bnd.area > 0:
                rng_int = m.integral_over_surface(image, rng_bnd)
                if np.isnan(rng_int): continue
                rng_density = rng_int / rng_bnd.area
            else:
                logger.warning("Ring polygon with no area!\r\nThickness of ring set to %.2f [pix]" % thickness)
                continue

            logger.debug("ring_around_nucleus on tag '%s' for nucleus id %d = %s" % (
                cfg['tag'].iloc[0], row['id'], m.eng_string(rng_int, si=True, format='%.2f')))
            rng_um = affinity.scale(rng_bnd, xfact=self.um_per_pix, yfact=self.um_per_pix, origin=(0, 0, 0))

            # TODO: scale intensity from pixels^2 to um^2
            self._mdf.loc[ix, 'ring'] = dumps(rng_um, rounding_precision=4)
            self._mdf.loc[ix, 'ring_pix'] = dumps(rng_bnd, rounding_precision=1)
            self._mdf.loc[ix, '%s_rng_int' % cfg['tag'].iloc[0]] = int(rng_int)
            self._mdf.loc[ix, '%s_rng_dens' % cfg['tag'].iloc[0]] = int(rng_density)
            # logger.debug("\r\n" + str(self._mdf[ix]))

    def _measure_particle_in_cytoplasm(self, image, cfg):
        assert self._ix.any(), "no rows in the filtered dataframe"
        width, height = image.shape
        frame = Polygon([(0, 0), (0, height), (width, height), (width, 0)])

        nuclei = list()
        for _id, nuc in self._mdf.set_index("id").loc[self._ix, "nuc_pix"].iteritems():
            nuclei.append({"id": _id, "boundary": shapely.wkt.loads(nuc)})

        for ix, row in self._mdf[self._ix].iterrows():
            _id = row["id"]
            nucl_bnd = shapely.wkt.loads(row["nuc_pix"])
            cell_bnd = shapely.wkt.loads(row["cell_pix"])
            x0, y0, xf, yf = [int(u) for u in nucl_bnd.bounds]

            valid_sample, reason = m.is_valid_sample(frame, cell_bnd, nucl_bnd, nuclei)
            if not valid_sample: continue
            logger.debug("particle_in_cytoplasm for cell id %d" % _id)

            centr_crop = image[y0:yf, x0:xf]
            logger.info('applying centrosome algorithm for nuclei %d' % _id)

            # load boundaries of um space for dataframe construction
            n_bum = shapely.wkt.loads(row["nucleus"])
            c_bum = shapely.wkt.loads(row["cell"])

            cntr = m.centrosomes(centr_crop, min_size=0.2 * self.pix_per_um,
                                 max_size=0.5 * self.pix_per_um,
                                 threshold=0.01)
            cntr[:, 0] += x0
            cntr[:, 1] += y0
            cntrsmes = list()
            for k, c in enumerate(cntr):
                pt = Point(c[0], c[1])
                pti = m.integral_over_surface(image, pt.buffer(1 * self.pix_per_um))
                cntrsmes.append(
                    {'id': k, 'pt': Point(c[0] / self.pix_per_um, c[1] / self.pix_per_um), 'i': pti})
                cntrsmes = sorted(cntrsmes, key=lambda ki: ki['i'], reverse=True)

            logger.debug('found {:d} centrosomes'.format(len(cntrsmes)))

            twocntr = len(cntrsmes) >= 2
            c1 = cntrsmes[0] if len(cntrsmes) > 0 else None
            c2 = cntrsmes[1] if twocntr else None

            lc = 2 if c2 is not None else 1 if c1 is not None else np.nan
            # TODO: Add units support
            self._mdf.loc[ix, 'centrosomes'] = lc
            self._mdf.loc[ix, 'c1'] = c1['pt'].wkt if c1 is not None else None
            self._mdf.loc[ix, 'c2'] = c2['pt'].wkt if c2 is not None else None
            self._mdf.loc[ix, 'c1_int'] = c1['i'] if c1 is not None else np.nan
            self._mdf.loc[ix, 'c2_int'] = c2['i'] if c2 is not None else np.nan
            self._mdf.loc[ix, 'c1_d_nuc_centr'] = n_bum.centroid.distance(
                c1['pt']) if c1 is not None else np.nan
            self._mdf.loc[ix, 'c2_d_nuc_centr'] = n_bum.centroid.distance(c2['pt']) if twocntr else np.nan
            self._mdf.loc[ix, 'c1_d_nuc_bound'] = n_bum.exterior.distance(
                c1['pt']) if c1 is not None else np.nan
            self._mdf.loc[ix, 'c2_d_nuc_bound'] = n_bum.exterior.distance(c2['pt']) if twocntr else np.nan
            self._mdf.loc[ix, 'c1_d_cell_centr'] = c_bum.centroid.distance(
                c1['pt']) if c1 is not None else np.nan
            self._mdf.loc[ix, 'c2_d_cell_centr'] = c_bum.centroid.distance(c2['pt']) if twocntr else np.nan
            self._mdf.loc[ix, 'c1_d_cell_bound'] = c_bum.exterior.distance(
                c1['pt']) if c1 is not None else np.nan
            self._mdf.loc[ix, 'c2_d_cell_bound'] = c_bum.exterior.distance(c2['pt']) if twocntr else np.nan
            self._mdf.loc[ix, 'nuc_centr_d_cell_centr'] = n_bum.centroid.distance(c_bum.centroid)
            self._mdf.loc[ix, 'c1_d_c2'] = c1['pt'].distance(c2['pt']) if twocntr else np.nan

        # if len(self._mdf) > 0:
        #     # Compute SNR: Step 1. Calculate standard deviation of background
        #     logger.debug('computing std dev of background for SNR calculation')
        #     std_pericentrin = np.std(_img[cells_mask])
        #     # u_pericentrin = np.mean(_img[cells_mask])
        #     self._mdf['snr_c1'] = self._mdf['c1_int'].apply(lambda i: i / std_pericentrin if not np.isnan(i) else np.nan)
        #     self._mdf['snr_c2'] = self._mdf['c2_int'].apply(lambda i: i / std_pericentrin if not np.isnan(i) else np.nan)

    def _measure_histogram(self, image, cfg):
        n_bins = cfg['hist_bins'].iloc[0]
        min = cfg['hist_min'].iloc[0]
        max = cfg['hist_max'].iloc[0]
        tag = cfg['tag'].iloc[0]
        log_bins = cfg['hist_log'].iloc[0]
        if log_bins:
            bins = np.logspace(np.log10(1 if min == 0 else min), np.log10(max), n_bins)
        else:
            bins = np.linspace(min, max, n_bins)
        histogram, edges = np.histogram(image.ravel(), bins)

        if self._ix.any():
            self._mdf.loc[self._ix, 'hist_edges'] = np.array2string(edges, **_hist_edges_arr_ops)
            self._mdf.loc[self._ix, '%s_hist' % tag] = np.array2string(histogram.astype(int), separator=',')
        else:
            d = pd.DataFrame(data={
                'id': [None],
                'row': [self._row],
                'col': [self._col],
                'fid': [self._fid],
                'p': [self._zp],
                'hist_edges': [np.array2string(edges, **_hist_edges_arr_ops)],
                '%s_hist' % tag: [np.array2string(histogram.astype(int), separator=',')],
            })
            self._mdf = self._mdf.append(d, ignore_index=True, sort=False)

    def _measure_histogram_on_nucleus(self, image, cfg):
        assert self._ix.any(), "no rows in the filtered dataframe"
        assert "nuc_pix" in self._mdf[self._ix], "nucleus detection step needed"

        n_bins = cfg['hist_bins'].iloc[0]
        min = cfg['hist_min'].iloc[0]
        max = cfg['hist_max'].iloc[0]
        tag = cfg['tag'].iloc[0]
        log_bins = cfg['hist_log'].iloc[0]
        if log_bins:
            bins = np.logspace(np.log10(1 if min == 0 else min), np.log10(max), n_bins)
        else:
            bins = np.linspace(min, max, n_bins)

        for ix, row in self._mdf[self._ix].iterrows():
            _id = row["id"]
            nucl_bnd = shapely.wkt.loads(row["nuc_pix"])
            logger.debug("histogram_on_nucleus for nucleus id %d" % _id)

            histogram, edges = m.histogram_of_surface(image, nucl_bnd, bins)

            self._mdf.loc[self._ix, 'hist_edges'] = np.array2string(edges, **_hist_edges_arr_ops)
            self._mdf.loc[self._ix, '%s_nuc_hist' % tag] = np.array2string(histogram.astype(int), separator=',')

    def _measure_histogram_on_ring(self, image, cfg):
        assert self._ix.any(), "no rows in the filtered dataframe"
        assert "ring_pix" in self._mdf[self._ix], "ring detection step needed"

        n_bins = cfg['hist_bins'].iloc[0]
        min = cfg['hist_min'].iloc[0]
        max = cfg['hist_max'].iloc[0]
        tag = cfg['tag'].iloc[0]
        log_bins = cfg['hist_log'].iloc[0]
        if log_bins:
            bins = np.logspace(np.log10(1 if min == 0 else min), np.log10(max), n_bins)
        else:
            bins = np.linspace(min, max, n_bins)

        for ix, row in self._mdf[self._ix].iterrows():
            _id = row["id"]
            rng_bnd = shapely.wkt.loads(row["ring_pix"])
            logger.debug("histogram_on_nucleus for nucleus id %d" % _id)

            histogram, edges = m.histogram_of_surface(image, rng_bnd, bins)

            self._mdf.loc[self._ix, 'hist_edges'] = np.array2string(edges, **_hist_edges_arr_ops)
            self._mdf.loc[self._ix, '%s_rng_hist' % tag] = np.array2string(histogram.astype(int), separator=',')

    def _measure_line_intensity(self, image, cfg):
        assert self._ix.any(), "no rows in the filtered dataframe"
        assert "ring_pix" in self._mdf[self._ix], "ring detection step needed"

        width, height = [s for s in image.shape]
        tag = cfg['tag'].iloc[0]
        n_lines = cfg['n_lines'].iloc[0]
        rng_thick = cfg['rng_thickness'].iloc[0]
        rng_thick *= self.pix_per_um
        angle_delta = 2 * np.pi / n_lines
        frame = Polygon([(0, 0), (0, width), (height, width), (height, 0)]).buffer(-rng_thick * self.pix_per_um)

        for ix, row in self._mdf[self._ix].iterrows():
            _id = row["id"]
            nucleus = shapely.wkt.loads(row["nuc_pix"])
            logger.debug("line_intensity for nucleus id %d" % _id)

            minx, miny, maxx, maxy = nucleus.bounds
            radius = max(maxx - minx, maxy - miny)
            for k, angle in enumerate([angle_delta * i for i in range(n_lines)]):
                ray = LineString([nucleus.centroid,
                                  (nucleus.centroid.x + radius * np.cos(angle),
                                   nucleus.centroid.y + radius * np.sin(angle))])
                r_seg = ray.intersection(nucleus)
                # print(r_seg, ray)
                if r_seg.is_empty:
                    continue
                if type(r_seg) == MultiLineString:
                    r_seg = r_seg[0]
                    # print(r_seg)
                pt = Point(r_seg.coords[-1])
                if not frame.contains(pt):
                    continue

                for pt0, pt1 in m.pairwise(nucleus.exterior.coords):
                    # if pt.touches(LineString([pt0, pt1])):
                    if Point(pt).distance(LineString([pt0, pt1])) < 1e-6:
                        # compute normal vector
                        dx = pt1[0] - pt0[0]
                        dy = pt1[1] - pt0[1]
                        # touching point of the polygon line segment
                        px, py = pt.x, pt.y
                        # normalize normal vector
                        mag = np.sqrt(dx ** 2 + dy ** 2)
                        dx, dy = dx / mag, dy / mag

                        r0, c0, r1, c1 = np.array([px, py, px - dy * rng_thick, py + dx * rng_thick]).astype(int)
                        lin = LineString([(r0, c0), (r1, c1)])
                        rr, cc = draw.line(r0, c0, r1, c1)

                        # TODO: Add units support
                        ix = self._ix & (self._mdf['id'] == _id)
                        self._mdf.loc[ix, '%s_line_%02d' % (tag, k)] = np.array2string(image[cc, rr], separator=',')
                        # self._mdf.loc[ix, '%s_line_%02d_sum' % (tag, k)] = m.integral_over_line(image, lin).astype(int)
                        self._mdf.loc[ix, '%s_int_lines' % tag] = int(n_lines)

                        # FIXME: not measuring nucleus
                        # lin = r_seg
                        r0, c0, r1, c1 = np.array(r_seg).flatten().astype(int)
                        rr, cc = draw.line(r0, c0, r1, c1)
                        self._mdf.loc[ix, '%s_nuc_line_%02d' % (tag, k)] = np.array2string(image[cc, rr], separator=',')
                        # self._mdf.loc[ix, '%s_nuc_line_%02d_sum' % (tag, k)] = m.integral_over_line(image, lin).astype(int)

    def _relabel_id(self):
        assert self._ix.any(), "no rows in the filtered dataframe"

        def _fn(df):
            z = df["p"].unique()
            for k, (_ix, nuc) in enumerate(df.loc[df["p"] == min(z), "nuc_pix"].iteritems()):  # for every nuclei
                nucleus = shapely.wkt.loads(nuc).buffer(2 * self.pix_per_um)
                in_nuc_ix = df["nuc_pix"].apply(lambda v: nucleus.contains(shapely.wkt.loads(v)))
                df.loc[in_nuc_ix, "zid"] = k

            return df

        group = ["row", "col", "fid"]
        self._mdf = self._mdf.groupby(group).apply(_fn).reset_index(drop=True)
