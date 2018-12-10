import logging
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PyQt4.QtCore import *
from PyQt4.QtGui import QApplication
from matplotlib.ticker import EngFormatter
from shapely.geometry import Point

from gui.explore import ExplorationGui
from gui.browse import BrowseGui
import measurements as m
import operetta as o

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('hhlab')


def measure(hoechst, pericentrin, edu, nuclei, cells, resolution):
    out = list()
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

        if m.is_valid_sample(hoechst, clls[0]['boundary'], nucleus['boundary'], nuclei):
            pericentrin_crop = pericentrin[y0:yf, x0:xf]
            logger.info('applying centrosome algorithm for nuclei %d' % nucleus['id'])
            # self.pericentrin = exposure.equalize_adapthist(pcrop, clip_limit=0.03)
            cntr = m.centrosomes(pericentrin_crop, max_sigma=resolution * 0.5)
            cntr[:, 0] += x0
            cntr[:, 1] += y0
            cntrsmes = list()
            for k, c in enumerate(cntr):
                pt = Point(c[0], c[1])
                pti = m.integral_over_surface(pericentrin, pt.buffer(resolution * 1))
                cntrsmes.append({'id': k, 'pt': pt, 'i': pti})
                cntrsmes = sorted(cntrsmes, key=lambda k: k['i'], reverse=True)

            logger.debug('centrosomes {:s}'.format(str(cntrsmes)))

            edu_int = m.integral_over_surface(edu, nucleus['boundary'])
            dna_int = m.integral_over_surface(hoechst, nucleus['boundary'])

            twocntr = len(cntrsmes) >= 2
            c1 = cntrsmes[0] if len(cntrsmes) > 0 else None
            c2 = cntrsmes[1] if twocntr else None

            nucb = nucleus['boundary']
            cllb = clls[0]['boundary']

            lc = 2 if c2 is not None else 1
            d = pd.DataFrame(data={'id': [nucleus['id']],
                                   'dna_int': [dna_int],
                                   'edu_int': [edu_int],
                                   'centrosomes': [lc],
                                   'c1_int': [c1['i'] if c1 is not None else np.nan],
                                   'c2_int': [c2['i'] if c2 is not None else np.nan],
                                   'c1_d_nuc_centr': [nucb.centroid.distance(c1['pt']) if c1 is not None else np.nan],
                                   'c2_d_nuc_centr': [nucb.centroid.distance(c2['pt']) if twocntr else np.nan],
                                   'c1_d_nuc_bound': [nucb.exterior.distance(c1['pt']) if c1 is not None else np.nan],
                                   'c2_d_nuc_bound': [nucb.exterior.distance(c2['pt']) if twocntr else np.nan],
                                   'c1_d_cell_centr': [cllb.centroid.distance(c1['pt']) if c1 is not None else np.nan],
                                   'c2_d_cell_centr': [cllb.centroid.distance(c2['pt']) if twocntr else np.nan],
                                   'c1_d_cell_bound': [cllb.exterior.distance(c1['pt']) if c1 is not None else np.nan],
                                   'c2_d_cell_bound': [cllb.exterior.distance(c2['pt']) if twocntr else np.nan],
                                   'c1_d_c2': [c1['pt'].distance(c2['pt']) if twocntr else np.nan],
                                   'cell': cllb.wkb,
                                   'nucleus': nucb.wkb
                                   })
            df = df.append(d, ignore_index=True, sort=False)

            out.append({'id': nucleus['id'], 'cell': cllb, 'nucleus': nucb,
                        'centrosomes': [c1, c2], 'edu_int': edu_int, 'dna_int': dna_int})
    return out, df


def batch_process_operetta_folder(path):
    operetta = o.Montage(path)
    outdf = pd.DataFrame()
    for row, col, fid in operetta.stack_generator():
        logger.info('%d %d %d' % (row, col, fid))
        #     operetta.save_render(row, col, fid,max_width=300)
        hoechst, tubulin, pericentrin, edu = operetta.max_projection(row, col, fid)
        r = 30  # [um]
        resolution = 1550.3e-4
        imgseg, props = m.nuclei_segmentation(hoechst, radius=r * resolution)
        operetta.add_mesurement(row, col, fid, 'nuclei found', len(np.unique(imgseg)))

        if len(props) > 0:
            outdf = outdf.append(props)

            # self.nuclei_features = m.nuclei_features(imgseg, area_thresh=(r * self.resolution) ** 2 * np.pi)
            nuclei = m.nuclei_features(imgseg)
            for i, n in enumerate(nuclei):
                n['id'] = i

            cells, _ = m.cell_boundary(tubulin, hoechst)

            samples, df = measure(hoechst, pericentrin, edu, nuclei, cells, resolution)
            df['fid'] = fid
            df['row'] = row
            df['col'] = col
            outdf = outdf.append(df, ignore_index=True, sort=False)

    pd.to_pickle(outdf, 'out/nuclei.pandas')
    operetta.files.to_csv('out/operetta.csv')
    return outdf


if __name__ == '__main__':
    b_path = '/Volumes/Kidbeat/data/centr-dist(u2os)__2018-11-27T18_08_10-Measurement 1/Images'
    # df = batch_process_operetta_folder(b_path)
    df = pd.read_pickle('out/nuclei.pandas')
    print(df.groupby(['fid', 'row', 'col', 'id']).size())
    print(len(df.groupby(['fid', 'row', 'col', 'id']).size()))

    # df = df[df['c1_d_nuc_centr'] > 80]
    # df = df[df['c1_int'] > 2000]
    # df['c1_int'] = np.log(df['c1_int'])
    fig = plt.figure(figsize=(8, 8))
    ax = fig.gca()
    # map = ax.scatter(df['dna_int'], df['edu_int'], c=df['c1_int'], alpha=1)
    # cbar = fig.colorbar(map)
    # cbar.set_label('distance [um]', rotation=270)
    sns.scatterplot(x="dna_int", y="edu_int", hue="c1_d_nuc_bound", size="c1_int",
                    alpha=.5, palette="PRGn", data=df, ax=ax)

    ax.set_title('distance of the first centrosome with respect to nuleus centroid through cell cycle')
    ax.set_xlabel('dna [AU]')
    ax.set_ylabel('edu [AU]')
    formatter = EngFormatter(unit='')
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)
    ax.semilogy()
    ax.set_xlim([6e6, 6e7])
    ax.set_ylim([3e5, 1e8])
    fig.savefig('facs.pdf')

    #
    #
    #
    fig = plt.figure(figsize=(5, 5))
    ax = fig.gca()
    sns.scatterplot(x="c1_d_nuc_centr", y="c1_int",
                    alpha=.5, palette="PRGn", data=df, ax=ax)
    ax.set_xlabel('distance [um]')
    ax.set_ylabel('centrosome intensity [AU]')
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)
    ax.semilogy()

    fig.savefig('centr-intensity.pdf')

    df.loc[df['c1_int'] == 65535, 'c1_int'] = 0
    g = sns.jointplot(x="c1_d_nuc_centr", y="c1_int", data=df,
                      dropna=True, alpha=0.5)
    ax = g.ax_joint
    ax.set_ylim([0, 7500])
    g.savefig('centr-intensity-joint.pdf')

    plt.show()

    if True:
        base_path = os.path.abspath('%s' % os.getcwd())
        logging.info('Qt version:' + QT_VERSION_STR)
        logging.info('PyQt version:' + PYQT_VERSION_STR)
        logging.info('Working dir:' + os.getcwd())
        logging.info('Base dir:' + base_path)
        os.chdir(base_path)

        app = QApplication(sys.argv)

        egui = ExplorationGui()
        operetta = o.Montage(b_path)
        bgui = BrowseGui(operetta=operetta, exploration_gui=egui)
        # from pycallgraph import PyCallGraph
        # from pycallgraph.output import GraphvizOutput
        #
        # with PyCallGraph(output=GraphvizOutput()):
        bgui.show()
        egui.show()
        code = app.exec_()
        sys.exit(code)
