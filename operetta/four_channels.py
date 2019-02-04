import os

import shapely.wkt
import numpy as np
import pandas as pd
from skimage import color
from skimage import exposure
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


class FourChannels(Montage):
    def __init__(self, base_path, row=None, col=None, condition_name=None):
        logger.info('Initializing FourChannels object with pandas file from folder %s' % base_path)
        try:
            super().__init__(base_path, row=row, col=col, condition_name=condition_name)
        except ImagesFolderNotFound:
            self.images_path = os.path.join(base_path, 'Images')

        pd_path = os.path.join(base_path, 'out', 'nuclei.pandas')
        if os.path.exists(pd_path):
            self.samples = pd.read_pickle(pd_path)

            if np.any([i not in self.samples for i in ['row', 'col', 'fid']]):
                is_row, is_col, is_fid = [i in self.samples for i in ['row', 'col', 'fid']]
                raise NoSamplesError('key columns not in dataframe row=%s col=%s fid=%s' % (is_row, is_col, is_fid))
        else:
            logger.warning('no pandas file found.')
            self.samples = None

    @staticmethod
    def filename_of_render(row, basepath):
        if len(row) > 1:
            raise Exception('only 1 row allowed')
        name = 'r%d-c%d-f%d-i%d.jpg' % (row['row'], row['col'], row['fid'], row['id'])
        if os.path.basename(basepath) == 'render':
            path = os.path.abspath(os.path.join(basepath, name))
        else:
            path = os.path.abspath(os.path.join(basepath, 'render', name))
        return name, path

    def save_render(self, row, col, fid, path=None, max_width=50):
        if self.samples is None:
            raise NoSamplesError('pandas samples file is needed to use this function.')

        hoechst_raw, tubulin_raw, pericentrin_raw, edu_raw = self.max_projection(row, col, fid)

        hoechst_gray = exposure.equalize_hist(hoechst_raw)
        # tubulin_gray = exposure.equalize_hist(tubulin_raw)
        pericentrin_gray = exposure.equalize_hist(pericentrin_raw)
        # edu_gray = exposure.equalize_hist(edu_gray)

        hoechst = color.gray2rgb(hoechst_gray)
        # tubulin = color.gray2rgb(tubulin_gray)
        pericentrin = color.gray2rgb(pericentrin_gray)
        # edu = color.gray2rgb(edu_gray)

        alexa_488 = [.29, 1., 0]
        alexa_594 = [1., .61, 0]
        alexa_647 = [.83, .28, .28]
        hoechst_33342 = [0, .57, 1.]
        # out = hoechst * hoechst_33342 * 0.25 + \
        #       tubulin * alexa_488 * 0.25 + \
        #       pericentrin * alexa_594 * 0.25 + \
        #       edu * alexa_647 * 0.25

        # out = hoechst * hoechst_33342 * 0.4 + tubulin * alexa_488 * 0.4
        out = hoechst * hoechst_33342 * 0.5 + pericentrin * alexa_594 * 0.5
        # out = pericentrin * alexa_594

        fig_general = Figure((max_width * 4 / 150, max_width * 4 / 150), dpi=150)
        canvas_g = FigureCanvas(fig_general)
        axg = fig_general.gca()
        fig_closeup = Figure((max_width / 150, max_width / 150), dpi=150)
        canvas_c = FigureCanvas(fig_closeup)
        axc = fig_closeup.gca()

        if path is None:
            basepath = os.path.dirname(self.images_path)
            basepath = os.path.abspath(os.path.join(basepath, 'render'))
        else:
            basepath = os.path.abspath(path)

        s = self.samples
        dfi = s[(s['row'] == row) & (s['col'] == col) & (s['fid'] == fid)]
        axg.cla()
        for ix, smp in dfi.groupby('id'):
            nucleus = shapely.wkt.loads(smp['nucleus'].values[0])
            cell = shapely.wkt.loads(smp['cell'].values[0])

            c1 = shapely.wkt.loads(smp['c1'].values[0]) if smp['c1'].values[0] is not None else None
            c2 = shapely.wkt.loads(smp['c2'].values[0]) if smp['c2'].values[0] is not None else None
            p.render_cell(nucleus, cell, [c1, c2], ax=axg)

            # render and save closeup image
            axc.cla()
            p.render_cell(nucleus, cell, [c1, c2], ax=axc)
            w, h = hoechst_raw.shape
            axc.imshow(out, extent=[0, w / self.pix_per_um, h / self.pix_per_um, 0])
            minx, miny, maxx, maxy = cell.bounds
            w, h = maxx - minx, maxy - miny
            axc.set_xlim(cell.centroid.x - w / 2, cell.centroid.x + w / 2)
            axc.set_ylim(cell.centroid.y - h / 2, cell.centroid.y + h / 2)
            x0, xf = cell.centroid.x - w / 2, cell.centroid.x + w / 2
            y0, yf = cell.centroid.y - h / 2, cell.centroid.y + h / 2
            axc.set_xlim(x0, xf)
            axc.set_ylim(y0, yf)
            axc.plot([x0, x0 + 10], [y0 + 0.5, y0 + 0.5], c='w', lw=4)
            axc.text(x0 + 1, y0 + 0.6 * self.pix_per_um, '10 um', color='w')
            if "cluster" in smp:
                axc.text(nucleus.centroid.x, nucleus.centroid.y, smp["cluster"], color='w', zorder=10)

            pil = canvas_to_pil(canvas_c)
            name = 'r%d-c%d-f%d-i%d.jpg' % (row, col, fid, ix)
            fpath = os.path.abspath(os.path.join(basepath, name))
            pil.save(ensure_dir(fpath))

        axg.plot([5, 5 + 10], [5, 5], c='w', lw=4)
        axg.text(5 + 1, 5 + 1.5, '10 um', color='w')

        w, h = hoechst_raw.shape
        axg.imshow(out, extent=[0, w / self.pix_per_um, h / self.pix_per_um, 0])
        axg.set_xlim([0, w / self.pix_per_um])
        axg.set_ylim([0, h / self.pix_per_um])
        axg.set_axis_off()
        fig_general.tight_layout()
        pil = canvas_to_pil(canvas_g)
        name = 'r%d-c%d-f%d.jpg' % (row, col, fid)
        fpath = os.path.abspath(os.path.join(basepath, name))
        pil.save(ensure_dir(fpath))

    def is_valid_sample(self, width, height, cell_polygon, nuclei_polygon, nuclei_list=None):
        # check that neither nucleus or cell boundary touch the ends of the frame
        frame = Polygon([(0, 0), (0, width), (height, width), (height, 0)])

        nuclei_polygon = affinity.scale(nuclei_polygon, xfact=self.pix_per_um, yfact=self.pix_per_um, origin=(0, 0, 0))
        cell_polygon = affinity.scale(cell_polygon, xfact=self.pix_per_um, yfact=self.pix_per_um, origin=(0, 0, 0))

        # FIXME: not working
        if frame.touches(nuclei_polygon.buffer(2)):
            logger.debug('sample rejected because it was touching the frame')
            return False
        if not cell_polygon.contains(nuclei_polygon):
            # logger.debug("sample rejected because it didn't contain a nucleus")
            return False

        # make sure that there's only one nucleus inside cell
        if nuclei_list is not None:
            n_nuc = 0
            for ncl in nuclei_list:
                nuclei = ncl['boundary']
                if cell_polygon.contains(nuclei):
                    n_nuc += 1
            if n_nuc != 1:
                return False

        return True

    @staticmethod
    def is_valid_measured_row(row):
        if row['centrosomes'] < 1: return False
        if row['tubulin_dens'] < 0.5e3: return False
        if np.isnan(row['c1_int']): return False
        if row['c2_int'] / row['c1_int'] < 0.6: return False
        if shapely.wkt.loads(row['nucleus']).area < 5 ** 2 * np.pi: return False

        return True

    def measure(self, *args):
        if len(args) == 1 and isinstance(args[0], int):
            id = args[0]
            r = self.files_gr.ix[id - 1]
            row, col, fid = r['row'], r['col'], r['fid']
            logger.debug('measuring id=%d row=%d col=%d fid=%d' % (id, row, col, fid))

        elif len(args) == 3 and np.all([np.issubdtype(a, np.integer) for a in args]):
            row, col, fid = args[0], args[1], args[2]
        else:
            logger.warning('nothing to save on this image')
            return

        df = self._measure_row_col_fid(row, col, fid)
        name = 'r%d-c%d-f%d.csv' % (row, col, fid)
        df.to_csv(self.save_path(name, subdir='pandas'), index=False)

    def _measure_row_col_fid(self, row, col, fid):
        hoechst, tubulin, pericentrin, edu = self.max_projection(row, col, fid)
        r = 10  # [um]
        pix_per_um = self.pix_per_um

        # --------------------
        #     Find nuclei
        # --------------------
        imgseg, nuclei = m.nuclei_segmentation(hoechst, radius=r * pix_per_um)
        n_nuclei = len(np.unique(imgseg)) - 1  # background is labeled as 0 and is not counted in

        if n_nuclei == 0:
            logger.warning("couldn't find nuclei in image")
            return pd.DataFrame()
        self.add_mesurement(row, col, fid, 'nuclei found', n_nuclei)

        # --------------------
        #     Find cells
        # --------------------
        cells, _ = m.cell_boundary(tubulin, hoechst)
        # TODO: compute SNR of measurements

        df = pd.DataFrame()
        for nucleus in nuclei:
            x0, y0, xf, yf = [int(u) for u in nucleus['boundary'].bounds]

            # search for closest cell boundary based on centroids
            clls = list()
            for cl in cells:
                clls.append({'id': cl['id'],
                             'boundary': cl['boundary'],
                             'd': cl['boundary'].centroid.distance(nucleus['boundary'].centroid)})
            clls = sorted(clls, key=lambda k: k['d'])

            nucl_bnd = nucleus['boundary']
            cell_bnd = clls[0]['boundary']
            w, h = tubulin.shape
            if self.is_valid_sample(w, h, cell_bnd, nucl_bnd, nuclei):
                tubulin_int = m.integral_over_surface(tubulin, cell_bnd)
                tub_density = tubulin_int / cell_bnd.area
                if tub_density < 250:
                    logger.warning("sample rejected after validation because it had a low tubulin density")
                    logger.debug('tubulin density in cell: %0.2f, intensity %0.2f, area %0.2f' % (
                        tub_density, tubulin_int, cell_bnd.area))
                    continue

                pericentrin_crop = pericentrin[y0:yf, x0:xf]
                logger.info('applying centrosome algorithm for nuclei %d' % nucleus['id'])

                cntr = m.centrosomes(pericentrin_crop, max_sigma=pix_per_um * 0.5)
                cntr[:, 0] += x0
                cntr[:, 1] += y0
                cntrsmes = list()
                for k, c in enumerate(cntr):
                    pt = Point(c[0], c[1])
                    pti = m.integral_over_surface(pericentrin, pt.buffer(pix_per_um * 5))
                    cntrsmes.append({'id': k, 'pt': Point(c[0] / pix_per_um, c[1] / pix_per_um), 'i': pti})
                    cntrsmes = sorted(cntrsmes, key=lambda k: k['i'], reverse=True)

                # logger.debug('centrosomes {:s}'.format(str(cntrsmes)))
                logger.debug('found {:d} centrosomes'.format(len(cntrsmes)))

                twocntr = len(cntrsmes) >= 2
                c1 = cntrsmes[0] if len(cntrsmes) > 0 else None
                c2 = cntrsmes[1] if twocntr else None

                edu_int = m.integral_over_surface(edu, nucl_bnd)
                dna_int = m.integral_over_surface(hoechst, nucl_bnd)

                nucl_bndum = affinity.scale(nucl_bnd, xfact=self.um_per_pix, yfact=self.um_per_pix, origin=(0, 0, 0))
                cell_bndum = affinity.scale(cell_bnd, xfact=self.um_per_pix, yfact=self.um_per_pix, origin=(0, 0, 0))

                lc = 2 if c2 is not None else 1 if c1 is not None else np.nan
                # TODO: Add units support
                d = pd.DataFrame(data={
                    'id': [nucleus['id']],
                    'dna_int': [dna_int],
                    'edu_int': [edu_int],
                    'tubulin_int': [tubulin_int],
                    'dna_dens': [dna_int / nucl_bnd.area],
                    'tubulin_dens': [tubulin_int / cell_bnd.area],
                    'centrosomes': [lc],
                    'c1_int': [c1['i'] if c1 is not None else np.nan],
                    'c2_int': [c2['i'] if c2 is not None else np.nan],
                    'c1_d_nuc_centr': [nucl_bndum.centroid.distance(c1['pt']) if c1 is not None else np.nan],
                    'c2_d_nuc_centr': [nucl_bndum.centroid.distance(c2['pt']) if twocntr else np.nan],
                    'c1_d_nuc_bound': [nucl_bndum.exterior.distance(c1['pt']) if c1 is not None else np.nan],
                    'c2_d_nuc_bound': [nucl_bndum.exterior.distance(c2['pt']) if twocntr else np.nan],
                    'c1_d_cell_centr': [cell_bndum.centroid.distance(c1['pt']) if c1 is not None else np.nan],
                    'c2_d_cell_centr': [cell_bndum.centroid.distance(c2['pt']) if twocntr else np.nan],
                    'c1_d_cell_bound': [cell_bndum.exterior.distance(c1['pt']) if c1 is not None else np.nan],
                    'c2_d_cell_bound': [cell_bndum.exterior.distance(c2['pt']) if twocntr else np.nan],
                    'nuc_centr_d_cell_centr': [nucl_bndum.centroid.distance(cell_bndum.centroid)],
                    'c1_d_c2': [c1['pt'].distance(c2['pt']) if twocntr else np.nan],
                    'cell': cell_bndum.wkt,
                    'nucleus': nucl_bndum.wkt,
                    'c1': c1['pt'].wkt if c1 is not None else None,
                    'c2': c2['pt'].wkt if c2 is not None else None,
                })
                df = df.append(d, ignore_index=True, sort=False)

        if len(df) > 0:
            df['fid'] = fid
            df['row'] = row
            df['col'] = col

        return df
