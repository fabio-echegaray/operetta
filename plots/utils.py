import logging
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib.ticker import EngFormatter
from shapely.geometry import Polygon
import numpy as np
import matplotlib.colors as mcolors
from numpy import asarray, concatenate, ones

matplotlib.rcParams['hatch.linewidth'] = 0.1

logger = logging.getLogger('plots')
logger.setLevel(logging.DEBUG)


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


def histogram_of_every_row(counts_col, **kwargs):
    """
    Plots the histogram of every row in a dataframe, overlaying them.
    Meant to be used within a map_dataframe call.

    Implementation inspired in https://matplotlib.org/gallery/api/histogram_path.html

    :param df: Dataframe
    :param counts_col: Name of the counts column
    :param edges_col: Name of the edges column
    :param ax: Matplotlib axis object
    :return:
    """
    ax = plt.gca()
    edges_col = kwargs.pop("edges_col") if "edges_col" in kwargs else "hist_edges"
    data = kwargs.pop("data")
    color = kwargs.pop("color")
    label = kwargs.pop("label") if "label" in kwargs else None

    # FIXME: Find a way to increase opacity resolution
    min_op = 1. / 256.
    opacity = 1. / len(data)
    opacity = opacity if opacity >= min_op else min_op
    logger.info("plotting %d histograms overlaid with opacity %e." % (len(data), opacity))

    # get the corners of the rectangles for the histogram
    bins = data[edges_col].iloc[0]
    left = np.array(bins[:-1])
    right = np.array(bins[1:])
    bottom = np.zeros(len(left))

    for _ix, _d in data.iterrows():
        counts = _d[counts_col]
        top = bottom + counts

        # function to build a compound path
        XY = np.array([[left, left, right, right], [bottom, top, top, bottom]]).T

        # get the Path object
        barpath = Path.make_compound_path_from_polys(XY)

        # make a patch out of it
        patch = PathPatch(barpath, alpha=opacity, lw=0, color=color)
        ax.add_patch(patch)

    ax.xaxis.set_major_locator(ticker.LogLocator(base=10))
    ax.xaxis.set_major_formatter(ticker.LogFormatterMathtext(base=10))
    ax.yaxis.set_major_formatter(EngFormatter(unit=''))
    ax.set_xlim(left[0], right[-1])
    ax.set_ylim(bottom.min(), data[counts_col].apply(lambda r: np.max(r)).max())


def histogram_with_errorbars(df, edges_col, value_col, ax=None):
    if ax is None:
        ax = plt.gca()

    edges = df[edges_col].iloc[0]
    rng_avg = df[value_col].mean()
    rng_std = np.std(df[value_col].values, axis=0)
    bincenters = 0.5 * (edges[1:] + edges[:-1])
    plt.bar(bincenters, rng_avg, width=np.diff(edges), yerr=rng_std, error_kw={"elinewidth": 1})
