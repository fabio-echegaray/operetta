import logging
import os

import numpy as np
import pandas as pd
import shapely.geometry
import shapely.wkt

import operetta as o

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('hhlab')
logging.getLogger('matplotlib').setLevel(logging.ERROR)
pd.set_option('display.width', 320)
pd.set_option('display.max_columns', 20)
pd.set_option('display.max_rows', 100)


def is_valid_measured_row(row):
    if row['cell'] is None: return False
    if row['centrosomes'] < 1: return False
    if row['tubulin_dens'] < 150: return False
    if np.isnan(row['c1_int']): return False
    if row['c2_int'] / row['c1_int'] < 0.6: return False
    if shapely.wkt.loads(row['nucleus']).area < 5 ** 2 * np.pi: return False

    return True


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Process analysis output.')
    parser.add_argument('folder', metavar='F', type=str,
                        help='folder where operetta images reside')
    parser.add_argument('--gate', action='store_true',
                        help='filters groups of data according to cell cycle progression')
    parser.add_argument('--sort', action='store_true',
                        help='sort rendered images according to gating')
    args = parser.parse_args()

    if not os.path.exists(os.path.join(args.folder, 'out')):
        raise Exception('Folder does not have any analysis in. Make sure you run batch.py first.')

    if args.gate:
        op = o.ConfiguredChannels(args.folder)
        g = o.CellCycle(op, is_valid_fn=is_valid_measured_row)
        g.gate()

    if args.sort:
        op = o.ConfiguredChannels(args.folder)
        g = o.CellCycle(op, is_valid_fn=is_valid_measured_row)
        g.move_images()

