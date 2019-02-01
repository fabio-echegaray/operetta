import matplotlib.pyplot as plt
import shapely.geometry
from matplotlib.ticker import EngFormatter
import numpy as np

from gui import SUSSEX_CORAL_RED, SUSSEX_NAVY_BLUE


def facs(df, ax=None, xlim=[1, 8], ylim=[12, 18.5], color=None):
    if ax is None:
        ax = plt.gca()
    df["geometry"] = df.apply(lambda row: shapely.geometry.Point(row['dna_int'] / 1e6 / 6, np.log(row['edu_int'])),
                              axis=1)

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
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect('equal')


def render_cell(nucleus, cell, centrosomes, ax=None):
    if ax is None:
        ax = plt.gca()

    x, y = nucleus.exterior.xy
    cen = nucleus.centroid
    ax.plot(x, y, color='red', linewidth=1, solid_capstyle='round', zorder=2)
    ax.plot(cen.x, cen.y, color='red', marker='+', linewidth=1, solid_capstyle='round', zorder=2)

    if cell is not None:
        x, y = cell.exterior.xy
        ax.plot(x, y, color='yellow', linewidth=1, solid_capstyle='round', zorder=1)
        cenc = cell.centroid
        ax.plot(cenc.x, cenc.y, color='yellow', marker='+', linewidth=1, solid_capstyle='round', zorder=2)

    if centrosomes is not None:
        c1, c2 = centrosomes
        if c1 is not None:
            c = plt.Circle((c1.x, c1.y), radius=2, facecolor='none', edgecolor=SUSSEX_CORAL_RED,
                           linewidth=2, zorder=5)
            ax.add_artist(c)
            ax.plot([c1.x, cen.x], [c1.y, cen.y], color='gray', linewidth=1, zorder=2)
            ax.text(c1.x, c1.y, '%0.2f' % (c1.distance(cen)), color='w', zorder=10)
        if c2 is not None:
            c = plt.Circle((c2.x, c2.y), radius=2, facecolor='none', edgecolor=SUSSEX_NAVY_BLUE,
                           linewidth=2, zorder=5)
            ax.add_artist(c)

    ax.set_aspect('equal')
    ax.set_axis_off()
