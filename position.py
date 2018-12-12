import logging

from sympy import lambdify
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import EngFormatter
import shapely.geometry
from descartes import PolygonPatch
from sympy import symbols
from sympy.physics.mechanics import ReferenceFrame

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('hhlab')


def s_phase_function(t, ref):
    from sympy.codegen.cfunctions import log10

    f = t * ref.x + (log10(t - 2.2) + 17) * ref.y
    Tn = f.diff(t, ref).normalize().simplify()
    Nn = Tn.diff(t, ref).normalize().simplify()
    sq = f.to_matrix(ref).applyfunc(lambda x: x ** 2)
    # length = sympy.sqrt(sq.row(0)+sq.row(1))
    # return lambdify(t, f.to_matrix(N)[1]), lambdify(t, Tn.to_matrix(N)), lambdify(t, Nn.to_matrix(N))
    return lambdify(t, f.to_matrix(ref)[1]), Tn, Nn


if __name__ == '__main__':
    df = pd.read_pickle('out/nuclei.pandas')
    print(df.groupby(['fid', 'row', 'col', 'id']).size())
    print(len(df.groupby(['fid', 'row', 'col', 'id']).size()))

    df["geometry"] = df.apply(lambda row: shapely.geometry.Point(row['dna_int'] / 1e6 / 6, np.log(row['edu_int'])),
                              axis=1)
    df["cluster"] = -1

    fig = plt.figure(figsize=(8, 8))
    ax = fig.gca()
    ax.set_title('distance of the first centrosome with respect to nuleus centroid through cell cycle')
    ax.set_xlabel('dna [AU]')
    ax.set_ylabel('edu [AU]')
    formatter = EngFormatter(unit='')
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)
    # ax.semilogy()
    # ax.set_xlim([6e6, 6e7])
    # ax.set_ylim([3e5, 1e8])
    ax.set_xlim([1, 8])
    ax.set_ylim([12, 18.5])
    ax.set_aspect('equal')
    # phony_log_formatter(ax)

    map = ax.scatter(df['dna_int'] / 1e6 / 6, np.log(df['edu_int']), c=df['c1_d_nuc_bound'], alpha=1)
    # map = ax.scatter(df['dna_int'], df['edu_int'], c=df['c1_int'], alpha=1)
    # cbar = fig.colorbar(map)
    # cbar.set_label('distance [um]', rotation=270)
    # sns.scatterplot(x="dna_int", y="edu_int", hue="c1_d_nuc_bound", size="c1_int",
    #                 alpha=.5, palette="PRGn", data=df, ax=ax)

    # G1 ellipse
    circ = shapely.geometry.Point((2.2, 13.8)).buffer(0.9)
    ell = shapely.affinity.scale(circ, 1, 0.6)
    ellr = shapely.affinity.rotate(ell, 20)

    ix = df['geometry'].apply(lambda g: g.within(ellr))
    df.loc[ix, 'cluster'] = 1

    patch = PolygonPatch(ellr, fc="#FF0000", ec="#999999", alpha=0.5, zorder=2)
    ax.add_patch(patch)

    # S phase rect 1
    poly = shapely.geometry.Polygon([(1.5, 14.4), (3.0, 14.4), (3.0, 15.4), (1.5, 15.4)])
    patch = PolygonPatch(poly, fc="#0000FF", ec="#999999", alpha=0.6, zorder=2)
    ax.add_patch(patch)

    ix = df['geometry'].apply(lambda g: g.within(poly))
    df.loc[ix, 'cluster'] = 2

    # S phase rect 2
    poly = shapely.geometry.Polygon([(3.3, 16.1), (5.5, 16.6), (5.5, 15.1), (3.2, 14.8)])
    patch = PolygonPatch(poly, fc="#0000FF", ec="#999999", alpha=0.6, zorder=2)
    ax.add_patch(patch)

    ix = df['geometry'].apply(lambda g: g.within(poly))
    df.loc[ix, 'cluster'] = 19

    # G2 ellipse
    circ = shapely.geometry.Point((4.35, 14.4)).buffer(1)
    ell = shapely.affinity.scale(circ, 1.4, 0.6)
    ellr = shapely.affinity.rotate(ell, 15)

    ix = df['geometry'].apply(lambda g: g.within(ellr))
    df.loc[ix, 'cluster'] = 20

    patch = PolygonPatch(ellr, fc="#FF0000", ec="#999999", alpha=0.5, zorder=2)
    ax.add_patch(patch)

    x = np.linspace(0.1, 5.5, num=100)
    t = symbols('t', positive=True)
    N = ReferenceFrame('N')
    f, tn, nn = s_phase_function(t, N)

    h = 0.8
    interior = lambdify(t, (nn * h).to_matrix(N))
    exterior = lambdify(t, (nn * -h).to_matrix(N))

    [ox], [oy], _ = interior(x) + np.array([[x], [f(x)], [0]])
    [ix], [iy], _ = exterior(x) + np.array([[x], [f(x)], [0]])
    ax.plot(ox, oy, c='#000AFF')
    ax.plot(x, f(x), c='#FF00FF', zorder=1)
    ax.plot(ix, iy, c='#000AFF')

    n = len(np.arange(2.2, 5.5, step=0.5))
    current_palette = sns.color_palette('bright', n)
    print(current_palette)
    xti = None
    for k, xt in enumerate(np.arange(2.2, 5.5, step=0.7)):
        x_ii = np.where(x <= xt)[0].max() + 1
        xx = x[x_ii]

        xi, yi, _ = interior(xx) + np.array([[xx], [f(xx)], [0]])
        xf, yf, _ = exterior(xx) + np.array([[xx], [f(xx)], [0]])
        ax.scatter(xx, f(xx), s=10, c='k', zorder=10)
        ax.plot([xi, xf], [yi, yf], c='r', zorder=10)

        if xti is not None:
            x_ix = np.where((xti <= x) & (x <= xt))[0]
            xx = x[np.append(x_ix, x_ix.max() + 1)]
            [xi], [yi], _ = interior(xx) + np.array([[xx], [f(xx)], [0]])
            [xf], [yf], _ = exterior(xx) + np.array([[xx], [f(xx)], [0]])
            pointList = list()
            pointList.extend([shapely.geometry.Point(x, y) for x, y in zip(xi, yi)])
            pointList.extend([shapely.geometry.Point(x, y) for x, y in zip(np.flip(xf), np.flip(yf))])

            poly = shapely.geometry.Polygon([(p.x, p.y) for p in pointList])
            patch = PolygonPatch(poly, fc=current_palette[k], ec="#999999", alpha=0.6, zorder=2)
            ax.add_patch(patch)

            ix = df['geometry'].apply(lambda g: g.within(poly))
            df.loc[ix, 'cluster'] = k + 2

        xti = xt
    fig.savefig('facs.pdf')

    print(df.groupby('cluster').size())

    # fig = plt.figure(figsize=(5, 5))
    # ax = fig.gca()
    # print(df.groupby('cluster').size())
    # with sns.color_palette('bright'):
    #     for ii, dfi in df[df['cluster'] > 0].groupby('cluster'):
    #         sns.distplot(dfi['c1_d_nuc_bound'].dropna(), hist=False, rug=True)
    # fig.savefig('centr-distribution.pdf')

    g = sns.FacetGrid(df[df['cluster'] > 0], row="cluster", height=1.5, aspect=5)
    g = g.map(sns.distplot, "c1_d_nuc_bound", rug=True)
    g.savefig('centr-distribution.pdf')

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

    if False:
        import os
        import sys
        from PyQt4.QtCore import *
        from PyQt4.QtGui import QApplication
        from gui.explore import ExplorationGui
        from gui.browse import BrowseGui
        import operetta as o

        b_path = '/Volumes/Kidbeat/data/centr-dist(u2os)__2018-11-27T18_08_10-Measurement 1/Images'

        base_path = os.path.abspath('%s' % os.getcwd())
        logging.info('Qt version:' + QT_VERSION_STR)
        logging.info('PyQt version:' + PYQT_VERSION_STR)
        logging.info('Working dir:' + os.getcwd())
        logging.info('Base dir:' + base_path)
        os.chdir(base_path)

        app = QApplication(sys.argv)

        egui = ExplorationGui()
        operetta = o.Dataframe('out/nuclei.pandas', b_path)
        bgui = BrowseGui(operetta=operetta, exploration_gui=egui)
        # from pycallgraph import PyCallGraph
        # from pycallgraph.output import GraphvizOutput
        #
        # with PyCallGraph(output=GraphvizOutput()):
        bgui.show()
        egui.show()
        code = app.exec_()
        sys.exit(code)
