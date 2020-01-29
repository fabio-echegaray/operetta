import os
import logging
from shutil import copyfile
import argparse

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import enlighten
from matplotlib.ticker import EngFormatter

import operetta as o
import measurements as m
from plots import colors as c
from plots import eval_into_array

plt.style.use('bmh')
pd.set_option('display.width', 320)
pd.set_option('display.max_columns', 30)
pd.set_option('display.max_rows', 50)

logger = logging.getLogger('ring')
logger.setLevel(logging.DEBUG)
logging.getLogger('matplotlib').setLevel(logging.ERROR)


def select_images(df, operetta_folder, method="copy"):
    render_path = o.ensure_dir(os.path.join(operetta_folder, 'out', 'render'))
    manager = enlighten.get_manager()
    bar = manager.counter(total=len(df), desc='Progress', unit='files')

    for ix, r in df.iterrows():
        destination_folder = o.ensure_dir(os.path.join(operetta_folder, 'out', 'selected-images',
                                                       '%s@%s' % (r["Cell Type"], r["Cell Count"]), r["Compound"]))

        # name, original_path = o.ConfiguredChannels.filename_of_render(r, render_path)
        name = 'r%d-c%d-f%d-p%s-i%d.jpg' % (r["row"], r["col"], r["fid"], str(r["p"]), r["zid"])
        original_path = os.path.join(render_path, name)
        destination_path = o.ensure_dir(os.path.join(destination_folder, name))

        try:
            if method == "link":
                logger.debug('linking %s to %s' % (name, destination_folder))
                os.symlink(original_path, destination_path, False)
            elif method == "copy":
                logger.debug('copying %s to %s' % (name, destination_folder))
                copyfile(original_path, destination_path)
            elif method == "move":
                logger.debug('moving %s to %s' % (name, destination_folder))
                os.rename(original_path, destination_path)
            bar.update()
        except Exception as e:
            logger.warning('no render for %s' % original_path)
            logger.warning(e)
            # traceback.print_stack()
    manager.stop()


def hist_area(edges, counts):
    bin_width = np.diff(edges)
    return np.sum([bin_width[i] * counts[i] for i in range(len(counts))])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Does a ring analysis based on previous measurements.')
    parser.add_argument('folder', metavar='FOLDER', type=str,
                        help='folder where ring images reside')
    args = parser.parse_args()

    df = pd.DataFrame()
    foldr = 0
    for root, directories, filenames in os.walk(args.folder):
        for filename in filenames:
            ext = filename.split('.')[-1]
            if ext == 'csv':
                logger.info("loading %s" % os.path.join(root, filename))
                f = pd.read_csv(os.path.join(root, filename))
                f.loc[:, "folder"] = foldr
                foldr += 1
                df = (df.append(f, ignore_index=True, sort=False)
                      .drop(columns=['Unnamed: 0', 'c', 'd', 'ls0', 'ls1', 'sum', 'x', 'y'])
                      .pipe(eval_into_array, column_name="l")
                      )

    df.loc[:, "indiv"] = df.apply(lambda r: "%s|%s|%s|%s" % (r['m'], r['n'], r['z'], r['folder']), axis=1)
    df.loc[:, "x"] = df.loc[:, "l"].apply(lambda v: np.arange(start=0, stop=len(v) * 0.05, step=0.05))
    # df.loc[:, "l"] = df.loc[:, "l"].apply(lambda v: v / max(v))
    print(df.columns)
    print(df["condition"].unique())
    print(df)

    x_var = 'x'
    y_var = 'l'
    b = m.vector_column_to_long_fmt(df, val_col=y_var, ix_col=x_var)
    formatter = EngFormatter(unit='')

    sns.set_palette([sns.xkcd_rgb["grey"], c.sussex_coral_red, c.sussex_cobalt_blue, c.sussex_deep_aquamarine])
    _order = ["cycling", "arrest", "cyto-arr", "noc-arr"]

    g = sns.FacetGrid(b, hue="condition", col='condition', col_wrap=2, col_order=_order, hue_order=_order, height=2)
    g = (g.map_dataframe(sns.lineplot, x=x_var, y=y_var, style='folder', units='indiv', estimator=None, alpha=1, lw=0.2)
         .set(xlim=(0, 4))
         )
    for _ax in g.axes:
        _ax.xaxis.set_major_formatter(formatter)
        _ax.yaxis.set_major_formatter(formatter)
    path = o.ensure_dir(os.path.join(args.folder, 'lines_indiv.pdf'))
    g.savefig(path)

    g = sns.FacetGrid(b, hue="condition", col='condition', col_wrap=2, col_order=_order, hue_order=_order, height=2)
    g = (g.map_dataframe(sns.lineplot, x=x_var, y=y_var, style='folder')
         .set(xlim=(0, 4))
         )
    for _ax in g.axes:
        _ax.xaxis.set_major_formatter(formatter)
        _ax.yaxis.set_major_formatter(formatter)
    path = o.ensure_dir(os.path.join(args.folder, 'lines_trend.pdf'))
    g.savefig(path)

    g = sns.FacetGrid(b, hue="condition", hue_order=_order, height=2)
    g = (g.map_dataframe(sns.lineplot, x=x_var, y=y_var)
         .set(xlim=(0, 4))
         .add_legend()
         )
    for _ax in g.axes[0]:
        # _ax.xaxis.set_major_formatter(formatter)
        _ax.yaxis.set_major_formatter(formatter)
    path = o.ensure_dir(os.path.join(args.folder, 'lines_overl.pdf'))
    g.savefig(path)
    plt.close()
