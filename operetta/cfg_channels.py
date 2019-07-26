import os
import ast
import configparser

from skimage import color
from skimage import exposure
import shapely.wkt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from shapely.geometry.point import Point
from shapely import affinity
from shapely.geometry.polygon import Polygon

import plots as p
import measurements as m
from .exceptions import ImagesFolderNotFound, NoSamplesError
from . import Montage, ensure_dir, logger
from gui.utils import canvas_to_pil


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

    def _load_cfg(self, path):
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
                        'tag': config.get(section, 'tag'),
                        'pipeline': ast.literal_eval(config.get(section, 'pipeline')),
                        'rng_thickness': config.getfloat(section, 'rng_thickness', fallback=0),
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
            config.set('Information', '# allowed functions for the pipeline are: nucleus, cell, '
                                      'intensity_in_nucleus, ring_around_nucleus, particle_in_cytoplasm')
            config.set('Information', '#')
            config.set('Information', '# Accepted parameters:')
            config.set('Information', '#     for ring_around_nucleus:')
            config.set('Information', '#         rng_thickness: Thickness of the ring, in um.')
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

        path = kwargs['path'] if 'path' in kwargs else None
        max_width = kwargs['max_width'] if 'max_width' in kwargs else None
        self._save_render_row_col_fid(row, col, fid, path=path, max_width=max_width)

    def _save_render_row_col_fid(self, row, col, fid, path=None, max_width=50):
        assert self.samples is not None, 'pandas samples file is needed to use this function'
        s = self.samples
        dfi = s[(s['row'] == row) & (s['col'] == col) & (s['fid'] == fid)]
        # assert len(dfi) == 1, 'more than one sample'

        # create path where the images are going to be stored
        if path is None:
            basepath = os.path.dirname(self.images_path)
            basepath = os.path.abspath(os.path.join(basepath, 'out', 'render'))
        else:
            basepath = os.path.abspath(path)

        # get images for render
        cfg_ch = self.configuration['channels']
        images = self.max_projection(row, col, fid)

        width, height = images[0].shape
        w_um, h_um = [s * self.um_per_pix for s in images[0].shape]
        frame = Polygon([(0, 0), (0, width), (height, width), (height, 0)])
        frx, fry = frame.exterior.xy
        out = np.zeros(images[0].shape + (3,), dtype=np.float64)

        # ----------------------
        #      OVERALL IMAGE
        # ----------------------
        fig_general = Figure((max_width * 4 / 150, max_width * 4 / 150), dpi=150)
        canvas_g = FigureCanvas(fig_general)
        axg = fig_general.gca()
        axg.plot(frx, fry, color='red', linewidth=2, solid_capstyle='round', zorder=10)
        for ch in cfg_ch:
            _img = images[ch['number']]
            if ch['flatfield']:
                _img = self._flat_field_correct(_img, ch['number'])
            _img = exposure.equalize_hist(_img)
            _img = color.gray2rgb(_img)

            for operation in ch['pipeline']:
                if operation == 'nucleus' and ch['render']:
                    out += _img * p.colors.hoechst_33342 * ch['render intensity']
                if operation == 'cell' and ch['render']:
                    out += _img * p.colors.alexa_488 * ch['render intensity']
                if operation == 'intensity_in_nucleus' and ch['render']:
                    out += _img * p.colors.alexa_647 * ch['render intensity']
                if operation == 'ring_around_nucleus' and ch['render']:
                    out += _img * p.colors.alexa_647 * ch['render intensity']
                if operation == 'particle_in_cytoplasm' and ch['render']:
                    out += _img * p.colors.alexa_594 * ch['render intensity']

        for ix, smp in dfi.groupby('id'):
            nucleus = shapely.wkt.loads(smp['nucleus'].iloc[0])
            cell = shapely.wkt.loads(smp['cell'].iloc[0]) if 'cell' in smp and not smp['cell'].isna().iloc[0] else None

            if 'c1' in smp and 'c2' in smp:
                c1 = shapely.wkt.loads(smp['c1'].iloc[0])
                c2 = shapely.wkt.loads(smp['c2'].iloc[0])
                centr = [c1, c2]
            else:
                centr = None

            p.render_cell(nucleus, cell, centr, base_zorder=20, ax=axg)
            if 'ring' in smp:
                ring = shapely.wkt.loads(smp['ring'].iloc[0])
                if ring.area == 0: continue
                p.render_polygon(ring, zorder=10, ax=axg)
            axg.text(nucleus.centroid.x + 2, nucleus.centroid.y - 1, ix, color='red', zorder=50)

        axg.plot([5, 5 + 10], [5, 5], c='w', lw=4)
        axg.text(5 + 1, 5 + 1.5, '10 um', color='w')

        axg.imshow(out, extent=[0, w_um, h_um, 0])
        axg.set_xlim([0, w_um])
        axg.set_ylim([0, h_um])
        axg.set_axis_on()

        pil = canvas_to_pil(canvas_g)
        name = 'r%d-c%d-f%d.jpg' % (row, col, fid)
        fpath = os.path.abspath(os.path.join(basepath, name))
        pil.save(ensure_dir(fpath))

        # ----------------------
        #     CLOSEUP RENDER
        # ----------------------
        fig_closeup = Figure((max_width / 150, max_width / 150), dpi=150)
        canvas_c = FigureCanvas(fig_closeup)
        axc = fig_closeup.gca()
        axc.set_facecolor('xkcd:salmon')
        for ix, smp in dfi.groupby('id'):
            nucleus = shapely.wkt.loads(smp['nucleus'].values[0])
            cell = shapely.wkt.loads(smp['cell'].iloc[0]) if 'cell' in smp and not smp['cell'].isna().iloc[0] else None

            if 'c1' in smp and 'c2' in smp:
                c1 = shapely.wkt.loads(smp['c1'].iloc[0])
                c2 = shapely.wkt.loads(smp['c2'].iloc[0])
                centr = [c1, c2]
            else:
                centr = None

            # render and save closeup image
            axc.cla()
            p.render_cell(nucleus, cell, centr, base_zorder=20, ax=axc)
            if 'ring' in smp:
                ring = shapely.wkt.loads(smp['ring'].iloc[0])
                if ring.area == 0: continue
                p.render_polygon(ring, zorder=10, ax=axc)
            _m = 30
            x0, xf = nucleus.centroid.x - _m, nucleus.centroid.x + _m
            y0, yf = nucleus.centroid.y - _m, nucleus.centroid.y + _m
            if x0 < 0: xf -= x0
            if y0 < 0: yf -= y0
            axc.imshow(out, extent=[0, w_um, h_um, 0])
            axc.set_xlim(x0, xf)
            axc.set_ylim(y0, yf)
            axc.plot([x0 + 5, x0 + 15], [y0 + 5, y0 + 5], c='w', lw=4)
            axc.text(x0 + 5, y0 + 7, '10 um', color='w', zorder=50)
            if "cluster" in smp:
                axc.text(nucleus.centroid.x, nucleus.centroid.y, smp["cluster"].iloc[0], color='w', zorder=10)

            pil = canvas_to_pil(canvas_c)
            name = 'r%d-c%d-f%d-i%d.jpg' % (row, col, fid, ix)
            fpath = os.path.abspath(os.path.join(basepath, name))
            pil.save(ensure_dir(fpath))

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
        images = self.max_projection(row, col, fid)
        assert len(images) > 0, 'no images for measurements'

        cfg = pd.DataFrame(cfg_ch)
        # NOTE: all the extraction process and validation is done in the pixel space
        df = pd.DataFrame()
        # --------------------
        #     Find nuclei
        # --------------------
        c = cfg[cfg['pipeline'].apply(lambda l: 'nucleus' in l)]
        assert len(c) > 0, 'nuclei finding step needed in configuration'
        assert len(c) < 2, 'only one nuclei finding step per batch allowed'

        _img = images[c['number'].iloc[0]]
        if c['flatfield'].iloc[0]:
            _img = self._flat_field_correct(_img, c['number'].iloc[0])

        r = 10  # [um]
        nuclei_img = _img.copy()
        imgseg, nuclei = m.nuclei_segmentation(nuclei_img, radius=r * self.pix_per_um)
        nuclei = m.exclude_contained(nuclei)
        n_nuclei = len(nuclei)

        if n_nuclei == 0:
            logger.warning("couldn't find nuclei in image")
            return pd.DataFrame()
        self.add_mesurement(row, col, fid, 'nuclei found', n_nuclei)

        for nucleus in nuclei:
            nucl_bnd = nucleus['boundary']
            dna_int = m.integral_over_surface(nuclei_img, nucl_bnd)

            # convert everything to um space for dataframe construction
            n_bum = affinity.scale(nucl_bnd, xfact=self.um_per_pix, yfact=self.um_per_pix, origin=(0, 0, 0))

            # TODO: Add units support
            d = pd.DataFrame(data={
                'id': [nucleus['id']],
                'dna_int': [dna_int],
                'dna_dens': [dna_int / nucl_bnd.area],
                'nucleus': n_bum.wkt,
            })
            df = df.append(d, ignore_index=True, sort=False)
        logger.debug("%d nuclei found in image" % n_nuclei)

        # --------------------
        #     Find cells
        # --------------------
        cell_step = False
        c = cfg[cfg['pipeline'].apply(lambda l: 'cell' in l)]
        assert len(c) < 2, 'only one cell finding step per batch allowed'

        if len(c) == 1:
            _img = images[c['number'].iloc[0]]
            if c['flatfield'].iloc[0]:
                _img = self._flat_field_correct(_img, c['number'].iloc[0])

            cell_img = _img.copy()
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
                cell_bnd = cl['boundary']
                for nucleus in nuclei:
                    nucl_bnd = nucleus['boundary']

                    valid_sample, reason = m.is_valid_sample(frame, cell_bnd, nucl_bnd, nuclei)
                    if reason == m.REJECTION_TOUCHING_FRAME: touching_fr += 1
                    if reason == m.REJECTION_NO_NUCLEUS: no_nuclei += 1
                    if reason == m.REJECTION_TWO_NUCLEI: two_nuclei += 1
                    if reason == m.REJECTION_CELL_TOO_BIG: too_big += 1
                    if valid_sample:
                        tubulin_int = m.integral_over_surface(cell_img, cell_bnd)
                        tub_density = tubulin_int / cell_bnd.area
                        if tub_density < 150:
                            logger.warning(
                                "sample rejected after validation because it had a low tubulin density")
                            logger.debug('tubulin density in cell: %0.2f, intensity %0.2f, area %0.2f' % (
                                tub_density, tubulin_int, cell_bnd.area))
                            continue

                        # convert everything to um space for dataframe construction
                        c_bum = affinity.scale(cell_bnd, xfact=self.um_per_pix, yfact=self.um_per_pix, origin=(0, 0, 0))

                        # TODO: Add units support
                        ix = df['id'] == nucleus['id']
                        df.loc[ix, 'tubulin_int'] = tubulin_int
                        df.loc[ix, 'tubulin_dens'] = tubulin_int / cell_bnd.area
                        df.loc[ix, 'cell'] = c_bum.wkt

            logger.info("%d samples rejected because they were touching the frame" % touching_fr)
            logger.info("%d samples rejected because cell didn't have a nucleus" % no_nuclei)
            logger.info("%d samples rejected because cell had more than two nuclei" % two_nuclei)
            logger.info("%d samples rejected because cell area was too big" % too_big)
            if len(df) > 0:
                cell_step = True

        # --------------------
        #     Intensity in nucleus
        # --------------------
        c = cfg[cfg['pipeline'].apply(lambda l: 'intensity_in_nucleus' in l)]
        # assert len(c) == 0, 'cell finding step needed in configuration'
        # assert len(c) > 1, 'only one cell finding step per batch allowed'

        for _, cf in c.iterrows():
            assert cell_step, 'intensity_in_nucleus currently needs cell data'
            _img = images[c['number'].iloc[0]]
            if c['flatfield'].iloc[0]:
                _img = self._flat_field_correct(_img, c['number'].iloc[0])

            for cl in cells:
                logger.debug("intensity_in_nucleus for cell id %d" % cl['id'])
                cell_bnd = cl['boundary']
                for nucleus in nuclei:
                    nucl_bnd = nucleus['boundary']

                    valid_sample, reason = m.is_valid_sample(frame, cell_bnd, nucl_bnd, nuclei)
                    if not valid_sample: continue
                    signal_int = m.integral_over_surface(_img, nucl_bnd)
                    signal_density = signal_int / nucl_bnd.area

                    # TODO: scale intensity from pixels^2 to um^2
                    ix = df['id'] == nucleus['id']
                    df.loc[ix, '%s_int' % cf['tag']] = signal_int
                    df.loc[ix, '%s_dens' % cf['tag']] = signal_density

        # --------------------
        #     Ring around nucleus
        # --------------------
        c = cfg[cfg['pipeline'].apply(lambda l: 'ring_around_nucleus' in l)]
        assert len(c) < 2, 'only one ring measurement step per batch allowed at the moment'

        for _, cf in c.iterrows():
            _img = images[cf['number']]
            if cf['flatfield']:
                _img = self._flat_field_correct(_img, cf['number'])

            for nucleus in nuclei:
                nucl_bnd = nucleus['boundary']
                thickness = float(cf['rng_thickness']) if not np.isnan(cf['rng_thickness']) in cf else 3.0
                thickness *= self.pix_per_um
                rng_bnd = nucl_bnd.buffer(thickness).difference(nucl_bnd)
                if rng_bnd.area > 0:
                    rng_int = m.integral_over_surface(_img, rng_bnd)
                    rng_density = rng_int / rng_bnd.area
                else:
                    logger.warning("Ring polygon with no area!\r\nThickness of ring set to %.2f [pix]" % thickness)
                    rng_int = 0
                    rng_density = 0

                logger.debug("ring_around_nucleus on tag '%s' for nucleus id %d of %s" % (
                    cf['tag'], nucleus['id'], m.eng_string(rng_int, si=True, format='%.2f')))
                rng_um = affinity.scale(rng_bnd, xfact=self.um_per_pix, yfact=self.um_per_pix, origin=(0, 0, 0))

                # TODO: scale intensity from pixels^2 to um^2
                ix = df['id'] == nucleus['id']
                df.loc[ix, 'ring'] = rng_um.wkt
                df.loc[ix, '%s_rng_int' % cf['tag']] = rng_int
                df.loc[ix, '%s_rng_dens' % cf['tag']] = rng_density

        # --------------------
        #     Particle in cytoplasm
        # --------------------
        c = cfg[cfg['pipeline'].apply(lambda l: 'particle_in_cytoplasm' in l)]
        # assert len(c) == 0, 'cell finding step needed in configuration'
        assert len(c) < 2, 'only one particle_in_cytoplasm step per batch allowed'

        if len(c) == 1:
            assert cell_step, 'particle_in_cytoplasm needs cell data'
            _img = images[c['number'].iloc[0]]
            if c['flatfield'].iloc[0]:
                _img = self._flat_field_correct(_img, c['number'].iloc[0])

            for cl in cells:
                logger.debug("particle_in_cytoplasm for cell id %d" % cl['id'])
                cell_bnd = cl['boundary']
                for nucleus in nuclei:
                    nucl_bnd = nucleus['boundary']
                    x0, y0, xf, yf = [int(u) for u in nucleus['boundary'].bounds]

                    valid_sample, reason = m.is_valid_sample(frame, cell_bnd, nucl_bnd, nuclei)
                    if not valid_sample: continue

                    centr_crop = _img[y0:yf, x0:xf]
                    logger.info('applying centrosome algorithm for nuclei %d' % nucleus['id'])

                    # convert everything to um space for dataframe construction
                    n_bum = affinity.scale(nucl_bnd, xfact=self.um_per_pix, yfact=self.um_per_pix, origin=(0, 0, 0))
                    c_bum = affinity.scale(cell_bnd, xfact=self.um_per_pix, yfact=self.um_per_pix, origin=(0, 0, 0))

                    cntr = m.centrosomes(centr_crop, min_size=0.2 * self.pix_per_um,
                                         max_size=0.5 * self.pix_per_um,
                                         threshold=0.01)
                    cntr[:, 0] += x0
                    cntr[:, 1] += y0
                    cntrsmes = list()
                    for k, c in enumerate(cntr):
                        pt = Point(c[0], c[1])
                        pti = m.integral_over_surface(_img, pt.buffer(1 * self.pix_per_um))
                        cntrsmes.append(
                            {'id': k, 'pt': Point(c[0] / self.pix_per_um, c[1] / self.pix_per_um), 'i': pti})
                        cntrsmes = sorted(cntrsmes, key=lambda ki: ki['i'], reverse=True)

                    logger.debug('found {:d} centrosomes'.format(len(cntrsmes)))

                    twocntr = len(cntrsmes) >= 2
                    c1 = cntrsmes[0] if len(cntrsmes) > 0 else None
                    c2 = cntrsmes[1] if twocntr else None

                    lc = 2 if c2 is not None else 1 if c1 is not None else np.nan
                    # TODO: Add units support
                    ix = df['id'] == nucleus['id']
                    df.loc[ix, 'centrosomes'] = lc
                    df.loc[ix, 'c1'] = c1['pt'].wkt if c1 is not None else None
                    df.loc[ix, 'c2'] = c2['pt'].wkt if c2 is not None else None
                    df.loc[ix, 'c1_int'] = c1['i'] if c1 is not None else np.nan
                    df.loc[ix, 'c2_int'] = c2['i'] if c2 is not None else np.nan
                    df.loc[ix, 'c1_d_nuc_centr'] = n_bum.centroid.distance(
                        c1['pt']) if c1 is not None else np.nan
                    df.loc[ix, 'c2_d_nuc_centr'] = n_bum.centroid.distance(c2['pt']) if twocntr else np.nan
                    df.loc[ix, 'c1_d_nuc_bound'] = n_bum.exterior.distance(
                        c1['pt']) if c1 is not None else np.nan
                    df.loc[ix, 'c2_d_nuc_bound'] = n_bum.exterior.distance(c2['pt']) if twocntr else np.nan
                    df.loc[ix, 'c1_d_cell_centr'] = c_bum.centroid.distance(
                        c1['pt']) if c1 is not None else np.nan
                    df.loc[ix, 'c2_d_cell_centr'] = c_bum.centroid.distance(c2['pt']) if twocntr else np.nan
                    df.loc[ix, 'c1_d_cell_bound'] = c_bum.exterior.distance(
                        c1['pt']) if c1 is not None else np.nan
                    df.loc[ix, 'c2_d_cell_bound'] = c_bum.exterior.distance(c2['pt']) if twocntr else np.nan
                    df.loc[ix, 'nuc_centr_d_cell_centr'] = n_bum.centroid.distance(c_bum.centroid)
                    df.loc[ix, 'c1_d_c2'] = c1['pt'].distance(c2['pt']) if twocntr else np.nan

            if len(df) > 0:
                # Compute SNR: Step 1. Calculate standard deviation of background
                logger.debug('computing std dev of background for SNR calculation')
                std_pericentrin = np.std(_img[cells_mask])
                # u_pericentrin = np.mean(_img[cells_mask])
                df['snr_c1'] = df['c1_int'].apply(lambda i: i / std_pericentrin if not np.isnan(i) else np.nan)
                df['snr_c2'] = df['c2_int'].apply(lambda i: i / std_pericentrin if not np.isnan(i) else np.nan)

        if len(df) > 0:
            df['fid'] = fid
            df['row'] = row
            df['col'] = col

        return df
