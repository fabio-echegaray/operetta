import logging
import os
import configparser
import ast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shapely.geometry
from matplotlib.patches import Ellipse

from gui.draggable import DraggableEightNote
import operetta as o

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('hhlab')
logging.getLogger('matplotlib').setLevel(logging.ERROR)


def move_images(df, path, folder):
    render_path = o.ensure_dir(os.path.join(path, 'render'))
    destination_folder = os.path.join(render_path, folder)
    os.makedirs(destination_folder, exist_ok=True)
    for i, r in df.iterrows():
        name, original_path = o.Dataframe.render_filename(r, path)
        destination_path = os.path.join(destination_folder, name)
        try:
            logger.warning('moving %s to %s' % (name, folder))
            os.rename(original_path, destination_path)
        except Exception:
            logger.warning('no render for %s' % destination_path)


def on_key(event):
    print('press', event.key)
    if event.key == 'g':
        gate()


def read_gate_config(fname):
    if os.path.isfile(fname):
        with open(fname, 'r') as configfile:
            config = configparser.ConfigParser()
            config.read_file(configfile)

            # logging.debug('sections found in file ' + str(config.sections()))

            section = 'Pands'
            if config.has_section(section):
                rows = ast.literal_eval(config.get(section, 'rows'))
                cols = ast.literal_eval(config.get(section, 'cols'))
            else:
                rows = []
                cols = []
            section = 'Gating geometries'
            if config.has_section(section):
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
    else:
        ellipse1 = Ellipse(xy=(2.0, 14.4), width=1, height=0.5, angle=25, fc='g', picker=5)
        ellipse2 = Ellipse(xy=(3.8, 14.9), width=1, height=0.5, angle=25, fc='g', picker=5)
        circle = plt.Circle(xy=(2.5, 16), radius=0.1, fc='r', picker=5)
        rows = []
        cols = []

    return ellipse1, ellipse2, circle, rows, cols


def write_gate_config(fname, df, ellipse1, ellipse2, circle):
    with open(fname, 'w') as configfile:
        config = configparser.RawConfigParser()
        config.add_section('General')
        config.set('General', 'Version', 'v0.1')

        section = 'Pandas'
        config.add_section(section)
        config.set(section, 'rows', sorted(df["row"].unique()))
        config.set(section, 'cols', sorted(df["col"].unique()))

        section = 'Gating geometries'
        config.add_section(section)
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

        config.write(configfile)


def gate(df, out_basepath, den):
    if df.empty:
        logger.warning('empty dataset!')
        return

    for i, poly in enumerate(den.polygons()):
        print(i)
        ix = df['geometry'].apply(lambda g: g.within(poly))
        df.loc[ix, 'cluster'] = i
    rorder = sorted(df["cluster"].unique())
    print(rorder)

    g = sns.FacetGrid(df, row="cluster", row_order=rorder, height=1.5, aspect=5)
    g = g.map(sns.distplot, "c1_d_nuc_centr", rug=True)
    g.savefig('{:s}/centr-distribution-nucleus-center.pdf'.format(out_basepath))

    g = sns.FacetGrid(df, row="cluster", row_order=rorder, height=1.5, aspect=5)
    g = g.map(sns.distplot, "c1_d_nuc_bound", rug=True)
    g.savefig('{:s}/centr-distribution-nucleus-boundary.pdf'.format(out_basepath))

    g = sns.FacetGrid(df, row="cluster", row_order=rorder, height=1.5, aspect=5)
    g = g.map(sns.distplot, "c1_d_cell_bound", rug=True)
    g.savefig('{:s}/centr-distribution-cell-boundary.pdf'.format(out_basepath))

    g = sns.FacetGrid(df, row="cluster", row_order=rorder, height=1.5, aspect=5)
    g = g.map(sns.distplot, "c1_d_c2", rug=True)
    g.axes[-1][0].set_xlim([-1, 10])
    g.savefig('{:s}/centr-distribution-inter-centr.pdf'.format(out_basepath))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Process analysis output.')
    parser.add_argument('folder', metavar='F', type=str,
                        help='folder where operetta images reside')
    parser.add_argument('--gate', action='store_true',
                        help='filters groups of data according to cell cycle progression')
    args = parser.parse_args()

    if not os.path.exists(os.path.join(args.folder, 'out')):
        raise Exception('Folder does not have any analysis in. Make sure you run batch.py first.')

    if args.gate:
        for root, directories, filenames in os.walk(os.path.join(args.folder, 'out')):
            for dir in directories:
                pd_path = os.path.join(args.folder, 'out', dir, 'nuclei.pandas')
                df = pd.read_pickle(pd_path)
                df = df[df['tubulin_dens'] > 0.5e3]
                df['c_ratio'] = df['c2_int'] / df['c1_int']
                df = df[df['c_ratio'] > 0.6]

                print(df.groupby(['fid', 'row', 'col', 'id']).size())
                print(len(df.groupby(['fid', 'row', 'col', 'id']).size()))

                df["geometry"] = df.apply(
                    lambda row: shapely.geometry.Point(row['dna_int'] / 1e6 / 6, np.log(row['edu_int'])),
                    axis=1)

                fig = plt.figure()
                ax = fig.gca()
                ax.set_aspect('equal')

                cidkeyboard = ax.figure.canvas.mpl_connect('key_press_event', on_key)

                cfg_path = os.path.join(args.folder, 'out', dir, 'gate.cfg')
                ellipse1, ellipse2, circle, rows, cols = read_gate_config(cfg_path)
                if not (len(rows) == 0 or len(cols) == 0):
                    df = df[(df["row"].isin(rows)) & (df["col"].isin(cols))]
                map = ax.scatter(df['dna_int'] / 1e6 / 6, np.log(df['edu_int']), c=df['c1_d_nuc_bound'], alpha=1)

                den = DraggableEightNote(ax, ellipse1, ellipse2, circle, number_of_sphase_segments=4)
                ax.set_title(dir)
                fig.subplots_adjust(top=0.99, bottom=0.3)

                plt.show()
                logger.info('gating...')
                write_gate_config(cfg_path, df, ellipse1, ellipse2, circle)
                gate(df, os.path.join(args.folder, 'out', dir), den)

    if False:
        import sys
        from PyQt4.QtCore import *
        from PyQt4.QtGui import QApplication
        from gui.explore import ExplorationGui
        from gui.browse import BrowseGui

        b_path = '/Volumes/Kidbeat/data/centr-dist(u2os)__2018-11-27T18_08_10-Measurement 1/'

        base_path = os.path.abspath('%s' % os.getcwd())
        logging.info('Qt version:' + QT_VERSION_STR)
        logging.info('PyQt version:' + PYQT_VERSION_STR)
        logging.info('Working dir:' + os.getcwd())
        logging.info('Base dir:' + base_path)
        os.chdir(base_path)

        app = QApplication(sys.argv)

        egui = ExplorationGui()
        operetta = o.Dataframe('{:s}/nuclei.pandas'.format(out_path), b_path)
        bgui = BrowseGui(operetta=operetta, exploration_gui=egui)
        # from pycallgraph import PyCallGraph
        # from pycallgraph.output import GraphvizOutput
        #
        # with PyCallGraph(output=GraphvizOutput()):
        bgui.show()
        egui.show()
        code = app.exec_()
        sys.exit(code)
