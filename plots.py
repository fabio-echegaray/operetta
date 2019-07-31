import ast

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from shapely.geometry import Polygon
from matplotlib.ticker import EngFormatter
import numpy as np
import matplotlib.colors as mcolors
from numpy import asarray, concatenate, ones

matplotlib.rcParams['hatch.linewidth'] = 0.1


class colors():
    alexa_488 = [.29, 1., 0]
    alexa_594 = [1., .61, 0]
    alexa_647 = [.83, .28, .28]
    hoechst_33342 = [0, .57, 1.]
    red = [1, 0, 0]
    green = [0, 1, 0]
    blue = [0, 0, 1]
    sussex_flint = mcolors.to_rgb('#013035')
    sussex_cobalt_blue = mcolors.to_rgb('#1e428a')
    sussex_mid_grey = mcolors.to_rgb('#94a596')
    sussex_fuschia_pink = mcolors.to_rgb('#eb6bb0')
    sussex_coral_red = mcolors.to_rgb('#df465a')
    sussex_turquoise = mcolors.to_rgb('#00afaa')
    sussex_warm_grey = mcolors.to_rgb('#d6d2c4')
    sussex_sunshine_yellow = mcolors.to_rgb('#ffb81c')
    sussex_burnt_orange = mcolors.to_rgb('#dc582a')
    sussex_sky_blue = mcolors.to_rgb('#40b4e5')

    sussex_navy_blue = mcolors.to_rgb('#1b365d')
    sussex_china_rose = mcolors.to_rgb('#c284a3')
    sussex_powder_blue = mcolors.to_rgb('#7da1c4')
    sussex_grape = mcolors.to_rgb('#5d3754')
    sussex_corn_yellow = mcolors.to_rgb('#f2c75c')
    sussex_cool_grey = mcolors.to_rgb('#d0d3d4')
    sussex_deep_aquamarine = mcolors.to_rgb('#487a7b')


def facs(df, ax=None, xlim=None, ylim=None, color=None):
    if ax is None:
        ax = plt.gca()

    if color is not None and color in df:
        df.loc[:, color] = df[color].transform(lambda x: (x - x.mean()) / x.std())
        ax.scatter(df['dna_int'] / 1e6 / 6, np.log(df['edu_int']), c=df[color], alpha=1)
    else:
        ax.scatter(df['dna_int'] / 1e6 / 6, np.log(df['edu_int']), alpha=0.5)

    ax.set_xlabel('dna [AU]')
    ax.set_ylabel('edu [AU]')
    formatter = EngFormatter(unit='')
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)
    if xlim is not None: ax.set_xlim(xlim)
    if ylim is not None: ax.set_ylim(ylim)
    ax.set_aspect('equal')


def render_cell(nucleus, cell, centrosomes, base_zorder=0, ax=None):
    if ax is None:
        ax = plt.gca()

    x, y = nucleus.exterior.xy
    cen = nucleus.centroid
    ax.plot(x, y, color='red', linewidth=1, solid_capstyle='round', zorder=base_zorder + 2)
    ax.plot(cen.x, cen.y, color='red', marker='+', linewidth=1, solid_capstyle='round', zorder=base_zorder + 2)

    if cell is not None:
        x, y = cell.exterior.xy
        ax.plot(x, y, color='yellow', linewidth=1, solid_capstyle='round', zorder=base_zorder + 1)
        cenc = cell.centroid
        ax.plot(cenc.x, cenc.y, color='yellow', marker='+', linewidth=1, solid_capstyle='round', zorder=base_zorder + 2)

    if centrosomes is not None:
        c1, c2 = centrosomes
        if c1 is not None:
            c = plt.Circle((c1.x, c1.y), radius=2, facecolor='none', edgecolor=colors.sussex_coral_red,
                           linewidth=2, zorder=base_zorder + 5)
            ax.add_artist(c)
            ax.plot([c1.x, cen.x], [c1.y, cen.y], color='gray', linewidth=1, zorder=base_zorder + 2)
            # ax.text(c1.x, c1.y, '%0.2f' % (c1.distance(cen)), color='w', zorder=base_zorder+10)
        if c2 is not None:
            c = plt.Circle((c2.x, c2.y), radius=2, facecolor='none', edgecolor=colors.sussex_navy_blue,
                           linewidth=2, zorder=base_zorder + 5)
            ax.add_artist(c)

    ax.set_aspect('equal')
    ax.set_axis_off()


def render_polygon(polygon: Polygon, zorder=0, ax=None):
    """
        These next two functions are taken from Sean Gillies
        https://sgillies.net/2010/04/06/painting-punctured-polygons-with-matplotlib.html
    """

    def ring_coding(ob):
        # The codes will be all "LINETO" commands, except for "MOVETO"s at the
        # beginning of each subpath
        n = len(ob.coords)
        codes = ones(n, dtype=Path.code_type) * Path.LINETO
        codes[0] = Path.MOVETO
        return codes

    def pathify(polygon):
        # Convert coordinates to path vertices. Objects produced by Shapely's
        # analytic methods have the proper coordinate order, no need to sort.
        vertices = concatenate(
            [asarray(polygon.exterior)]
            + [asarray(r) for r in polygon.interiors])
        codes = concatenate(
            [ring_coding(polygon.exterior)]
            + [ring_coding(r) for r in polygon.interiors])
        return Path(vertices, codes)

    if ax is None:
        ax = plt.gca()

    x, y = polygon.exterior.xy
    ax.plot(x, y, color='white', linewidth=0.5, solid_capstyle='round', zorder=zorder)

    path = pathify(polygon)
    patch = PathPatch(path, facecolor='none', edgecolor='white', hatch='/////', lw=0.01, zorder=zorder)

    ax.add_patch(patch)
    ax.set_aspect(1.0)


def histogram_of_every_row(df, counts_col="_hist_counts", edges_col="_hist_edges", ax=None):
    if ax is None:
        ax = plt.gca()

    for _d in df.iterrows():
        counts = ast.literal_eval(_d[counts_col])
        edges = ast.literal_eval(_d[edges_col])
        plt.hist(edges[:-1], edges, weights=counts, alpha=1. / len(df), ax=ax)
