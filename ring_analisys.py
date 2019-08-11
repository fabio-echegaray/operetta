import os
import logging
from shutil import copyfile
import argparse

import pandas as pd
import numpy as np
import enlighten

import operetta as o
from plots import Ring

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
        name = 'r%d-c%d-f%d-p%s-i%d.jpg' % (r["row"], r["col"], r["fid"], str(r["p"]), r["id"])
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
                        help='folder where operetta images reside')
    args = parser.parse_args()

    operetta = o.ConfiguredChannels(args.folder)
    pl = Ring(operetta)

    select_images(pl.lines_filtered, operetta.base_path)
    # exit(0)

    # pl.nuclei_filtered()
    pl.dna_intensity()
    pl.actin_intensity_and_density_histogram()
    pl.actin_intensity_vs_density()
    pl.dna_vs_actin_intesity()
    pl.actin_ring()
    pl.actin_ring_histograms()
    pl.histogram_areas()
    pl.line_measurements()
    pl.line_integrals()
