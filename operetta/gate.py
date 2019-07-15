import os
import configparser
import ast

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from shapely.geometry import Point
import shapely.wkt
from matplotlib.patches import Ellipse
import numpy as np

from gui.draggable import DraggableEightNote
import operetta as o
import plots as p
from . import logger
from .cfg_channels import ConfiguredChannels


class CellCycle():
    def __init__(self, operetta: ConfiguredChannels, is_valid_fn=None):
        assert type(operetta) == ConfiguredChannels, 'need a ConfiguredChannels object.'
        self._cf = operetta
        assert not self._cf.samples.empty, 'samples in operetta out folder are empty.'
        self.df = self._cf.samples[
            self._cf.samples.apply(is_valid_fn, axis=1)] if is_valid_fn is not None else self._cf.samples

        self.cfgfile = os.path.join(self._cf.base_path, 'out', 'gate', 'gate.cfg')
        if not os.path.exists(self.cfgfile):
            o.ensure_dir(self.cfgfile)
            self._generate_gate_file()
        self.cfg = self._read_gate_config()

    @staticmethod
    def _store_section(config, df, section, cell_type, cell_count, compound, concentration, ellipse1, ellipse2, circle):
        assert not config.has_section(section), 'section already in gate config file.'
        config.add_section(section)
        # _c = l[l[['Cell Type', 'Cell Count', 'Compound', 'Concentration']].apply(tuple, axis=1).isin(_i)]
        _c = df[(df['Cell Type'] == cell_type) & (df['Cell Count'] == float(cell_count)) & (
                df['Compound'] == compound) & (df['Concentration'] == concentration)]
        config.set(section, 'pandas_rows', sorted(_c["row"].unique()))
        config.set(section, 'pandas_cols', sorted(_c["col"].unique()))
        config.set(section, 'cell_type', cell_type)
        config.set(section, 'cell_count', cell_count)
        config.set(section, 'compound', compound)
        config.set(section, 'concentration', concentration)

        xe1, ye1 = ellipse1.center
        we1, he1, ae1 = ellipse1.width, ellipse1.height, ellipse1.angle
        xe2, ye2 = ellipse2.center
        we2, he2, ae2 = ellipse2.width, ellipse2.height, ellipse2.angle
        xc, yc = circle.center

        config.set(section, 'G1_ellipse_x', xe1)
        config.set(section, 'G1_ellipse_y', ye1)
        config.set(section, 'G1_ellipse_width', we1)
        config.set(section, 'G1_ellipse_height', he1)
        config.set(section, 'G1_ellipse_angle', ae1)

        config.set(section, 'G2_ellipse_x', xe2)
        config.set(section, 'G2_ellipse_y', ye2)
        config.set(section, 'G2_ellipse_width', we2)
        config.set(section, 'G2_ellipse_height', he2)
        config.set(section, 'G2_ellipse_angle', ae2)

        config.set(section, 'height_circle_x', xc)
        config.set(section, 'height_circle_y', yc)
        return config

    def _generate_gate_file(self):
        s = self._cf.samples.groupby(['row', 'col']).size().reset_index().drop(columns=0)
        l = self._cf.layout
        # now we filter all the layout rows we need based on the samples we've got
        # for this, we convert every row,col into a tuple in l, then filter by the existing tuples in s
        l = l[l[['row', 'col']].apply(tuple, axis=1).isin(s.apply(tuple, axis=1))].replace(pd.np.nan, '-')

        with open(self.cfgfile, 'w') as configfile:
            config = configparser.RawConfigParser()
            config.add_section('General')
            config.set('General', 'Version', 'v0.2')
            config.set('General', 'Gating instances', len(l))

            ellipse1 = Ellipse(xy=(2.0, 14.4), width=1, height=0.5, angle=25, fc='g', picker=5)
            ellipse2 = Ellipse(xy=(3.8, 14.9), width=1, height=0.5, angle=25, fc='g', picker=5)
            circle = plt.Circle(xy=(2.5, 16), radius=0.1, fc='r', picker=5)

            for k, _i in enumerate(l[['Cell Type', 'Cell Count', 'Compound', 'Concentration']].apply(tuple, axis=1)):
                cell_type, cell_count, compound, concentration = _i
                section = 'Gating for group %d' % (k + 1)
                config = self._store_section(config, l, section, cell_type, cell_count, compound, concentration,
                                             ellipse1, ellipse2, circle)

            config.write(configfile)

    def _read_gate_config(self):
        if os.path.isfile(self.cfgfile):
            with open(self.cfgfile, 'r') as configfile:
                config = configparser.ConfigParser()
                config.read_file(configfile)

                section = 'General'
                assert config.has_section(section), 'no general section in gate config file.'
                version = config.get(section, 'Version')
                assert version == 'v0.2', 'version == v0.2 required!'
                gn = config.getint(section, 'Gating instances')

                for k in range(1, gn + 1):
                    section = 'Gating for group %d' % k

                    if config.has_section(section):
                        rows = ast.literal_eval(config.get(section, 'pandas_rows'))
                        cols = ast.literal_eval(config.get(section, 'pandas_cols'))
                        cell_type = config.get(section, 'cell_type')
                        cell_count = config.get(section, 'cell_count')
                        compound = config.get(section, 'compound')
                        concentration = config.get(section, 'concentration')
                        xe1 = config.getfloat(section, 'G1_ellipse_x')
                        ye1 = config.getfloat(section, 'G1_ellipse_y')
                        we1 = config.getfloat(section, 'G1_ellipse_width')
                        he1 = config.getfloat(section, 'G1_ellipse_height')
                        ae1 = config.getfloat(section, 'G1_ellipse_angle')

                        xe2 = config.getfloat(section, 'G2_ellipse_x')
                        ye2 = config.getfloat(section, 'G2_ellipse_y')
                        we2 = config.getfloat(section, 'G2_ellipse_width')
                        he2 = config.getfloat(section, 'G2_ellipse_height')
                        ae2 = config.getfloat(section, 'G2_ellipse_angle')

                        xc = config.getfloat(section, 'height_circle_x')
                        yc = config.getfloat(section, 'height_circle_y')

                        ellipse1 = Ellipse(xy=(xe1, ye1), width=we1, height=he1, angle=ae1, fc='g', picker=5)
                        ellipse2 = Ellipse(xy=(xe2, ye2), width=we2, height=he2, angle=ae2, fc='g', picker=5)
                        circle = plt.Circle(xy=(xc, yc), radius=0.1, fc='r', picker=5)

                        yield section, ellipse1, ellipse2, circle, rows, cols, cell_type, cell_count, compound, concentration

    def gate(self):
        df = self.df
        df.loc[:, "area_nucleus"] = df.apply(lambda row: shapely.wkt.loads(row['nucleus']).area, axis=1)
        df.loc[:, "min_dist_to_nuclear_boundary"] = df[['c1_d_nuc_bound', 'c2_d_nuc_bound']].apply(min, axis=1)
        df.loc[:, "min_dist_to_nuclear_center"] = df[['c1_d_nuc_centr', 'c2_d_nuc_centr']].apply(min, axis=1)
        df.loc[:, "min_dist_to_cell_boundary"] = df[['c1_d_cell_bound', 'c2_d_cell_bound']].apply(min, axis=1)
        df.loc[:, "facs"] = df.apply(lambda row: Point(row['dna_int'] / 1e6 / 6, np.log(row['edu_int'])), axis=1)

        config = configparser.RawConfigParser()
        s = self._cf.samples.groupby(['row', 'col']).size().reset_index().drop(columns=0)
        l = self._cf.layout
        l = l[l[['row', 'col']].apply(tuple, axis=1).isin(s.apply(tuple, axis=1))].replace(pd.np.nan, '-')  # see above
        config.add_section('General')
        config.set('General', 'Version', 'v0.2')
        config.set('General', 'Gating instances', len(l))

        for g in self._read_gate_config():
            section, ellipse1, ellipse2, circle, rows, cols, cell_type, cell_count, compound, concentration = g
            logger.info('gating %s|%s|%s|%s...' % (cell_type, cell_count, compound, concentration))

            assert (len(rows) > 0 and len(cols) > 0), 'no rows or cols for this gating.'
            dfg = df[(df["row"].isin(rows)) & (df["col"].isin(cols))]

            dest_path = os.path.join(self._cf.base_path, 'out', 'gate', '%s@%s' % (cell_type, cell_count), compound,
                                     concentration)
            o.ensure_dir(os.path.join(dest_path, 'nil'))

            # gating in a matplotlib window
            fig = plt.figure()
            ax = fig.gca()
            ax.set_aspect('equal')
            # p.facs(dfg, ax=ax, xlim=[1, 8], ylim=[12, 18.5])
            p.facs(dfg, ax=ax)
            den = DraggableEightNote(ax, ellipse1, ellipse2, circle, number_of_sphase_segments=4)
            fig.subplots_adjust(top=0.99, bottom=0.3)
            plt.show()
            extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            fig.savefig('{:s}/gate.png'.format(dest_path), bbox_inches=extent)
            plt.close(fig)

            config = self._store_section(config, l, section, cell_type, cell_count, compound, concentration,
                                         ellipse1, ellipse2, circle)

            # assign a cluster id for every polygon
            if dfg.empty: continue
            for i, poly in enumerate(den.polygons()):
                ix = dfg['facs'].apply(lambda g: g.within(poly))
                dfg.loc[ix, 'cluster'] = i
            dfg = dfg[~dfg["cluster"].isna()]
            if dfg.empty: continue

            rorder = sorted(dfg["cluster"].unique())
            g = sns.FacetGrid(dfg, row="cluster", row_order=rorder, height=1.5, aspect=5)
            g = g.map(sns.distplot, "min_dist_to_nuclear_center", rug=True)
            g.axes[-1][0].set_xlim([-1, 20])
            g.savefig('{:s}/centr-distribution-nucleus-center.pdf'.format(dest_path))
            plt.close(g.fig)

            g = sns.FacetGrid(dfg, row="cluster", row_order=rorder, height=1.5, aspect=5)
            g = g.map(sns.distplot, "min_dist_to_nuclear_boundary", rug=True)
            g.axes[-1][0].set_xlim([-1, 20])
            g.savefig('{:s}/centr-distribution-nucleus-boundary.pdf'.format(dest_path))
            plt.close(g.fig)

            g = sns.FacetGrid(dfg, row="cluster", row_order=rorder, height=1.5, aspect=5)
            g = g.map(sns.distplot, "min_dist_to_cell_boundary", rug=True)
            g.axes[-1][0].set_xlim([-1, 30])
            g.savefig('{:s}/centr-distribution-cell-boundary.pdf'.format(dest_path))
            plt.close(g.fig)

            g = sns.FacetGrid(dfg, row="cluster", row_order=rorder, height=1.5, aspect=5)
            g = g.map(sns.distplot, "nuc_centr_d_cell_centr", rug=True)
            g.axes[-1][0].set_xlim([-1, 20])
            g.savefig('{:s}/centr-distribution-nuc-cell.pdf'.format(dest_path))
            plt.close(g.fig)

            # g = sns.FacetGrid(df, row="cluster", row_order=rorder, height=1.5, aspect=5)
            # g = g.map(sns.distplot, "c1_d_c2", rug=True)
            # g.axes[-1][0].set_xlim([-1, 10])
            # g.savefig('{:s}/centr-distribution-inter-centr.pdf'.format(out_path))
            # plt.close(g.fig)

            g = sns.FacetGrid(dfg, row="cluster", row_order=rorder, height=1.5, aspect=5)
            g = g.map(sns.distplot, "area_nucleus", rug=True)
            g.savefig('{:s}/nucleus-distribution-area.pdf'.format(dest_path))
            plt.close(g.fig)

            # rorder = sorted(dfg["cluster"].unique())
            # dfg = dfg.melt(id_vars=["row", "col", "cluster"])
            # corder = ["c1_d_nuc_centr", "c1_d_nuc_bound", "c1_d_cell_bound", "nuc_centr_d_cell_centr"]
            # dfg = dfg[dfg["variable"].isin(corder)]
            #
            # g = sns.FacetGrid(dfg, row="cluster", row_order=rorder, col="variable", height=1.5, aspect=3)
            # g = g.map_dataframe(_distplot, "value", rug=True)
            # g.savefig('{:s}/distribution-across-cc.pdf'.format(out_path))

        with open(self.cfgfile, 'w') as configfile:
            config.write(configfile)

    def move_images(self):
        df = self.df
        df.loc[:, "facs"] = df.apply(lambda row: Point(row['dna_int'] / 1e6 / 6, np.log(row['edu_int'])), axis=1)
        ax = plt.gca()
        render_path = o.ensure_dir(os.path.join(self._cf.base_path, 'out', 'render'))

        for g in self._read_gate_config():
            ellipse1, ellipse2, circle, rows, cols, cell_type, cell_count, compound, concentration = g
            logger.info('moving images for %s|%s|%s|%s...' % (cell_type, cell_count, compound, concentration))

            if not (len(rows) == 0 or len(cols) == 0):
                df = df[(df["row"].isin(rows)) & (df["col"].isin(cols))]

            den = DraggableEightNote(ax, ellipse1, ellipse2, circle, number_of_sphase_segments=4)
            # assign a cluster id for every polygon
            for i, poly in enumerate(den.polygons()):
                ix = df['facs'].apply(lambda g: g.within(poly))
                df.loc[ix, 'cluster'] = i
            df = df[~df["cluster"].isna()]

            for ix, dfg in df.groupby("cluster"):
                _path = os.path.join(self._cf.base_path, 'out', '%s@%s' % (cell_type, cell_count), compound,
                                     concentration, 'render', str(int(ix)))
                destination_folder = o.ensure_dir(_path)

                for i, r in dfg.iterrows():
                    name, original_path = o.ConfiguredChannels.filename_of_render(r, render_path)
                    destination_path = os.path.join(destination_folder, name)
                    try:
                        logger.info('moving %s to %s' % (name, destination_folder))
                        os.rename(original_path, destination_path)
                    except Exception:
                        logger.warning('no render for %s' % name)
