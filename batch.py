import configparser
import json
import logging
import os
import warnings

from pandas.errors import EmptyDataError

import filters
import operetta as o

logger = logging.getLogger('batch')
logger.setLevel(logging.DEBUG)

# reduce console output while using batch tool
logging.getLogger('matplotlib').setLevel(logging.ERROR)
logging.getLogger('shapely').setLevel(logging.ERROR)
warnings.simplefilter(action='ignore', category=FutureWarning)


def collect(path):
    df = pd.DataFrame()
    for root, directories, filenames in os.walk(os.path.join(path)):
        for filename in filenames:
            ext = filename.split('.')[-1]
            if ext == 'csv':
                try:
                    csv = pd.read_csv(os.path.join(root, filename))
                    df = df.append(csv, ignore_index=True)
                except EmptyDataError:
                    logger.warning('found empty csv file: %s' % filename)
                    # traceback.print_stack()

    return df


def wells_config(fname):
    logging.info('parsing wells.cfg')
    if os.path.isfile(fname):
        with open(fname, 'r') as configfile:
            config = configparser.ConfigParser(strict=False)
            config.read_file(configfile)

            logging.debug('sections found in file ' + str(config.sections()))

            for section in config.sections():
                if section[:3] == 'Tag':
                    name = config.get(section, 'name')
                    rows = json.loads(config.get(section, 'rows'))
                    cols = json.loads(config.get(section, 'cols'))
                    yield name, rows, cols


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Process operetta images.')
    parser.add_argument('folder', metavar='FOLDER', type=str,
                        help='folder where operetta images reside')
    parser.add_argument('--render', action='store_true',
                        help='render image with id given by --id (in a folder called render up in the hierarchy)')
    parser.add_argument('--image', action='store_true',
                        help='retrieve image of the stack with id extracted from --id into a tiff file')
    parser.add_argument('--measure', action='store_true',
                        help='measure_into_dataframe features on the dataset')
    parser.add_argument('--id', type=int,
                        help='select an image of the stack with specified ID')
    parser.add_argument('--collect', action='store_true',
                        help='collect measurements from csv format to pandas dataframe')
    args = parser.parse_args()

    """
        MAKE SURE YOU DON'T FORGET TO CALL
        module load python/intel/3.6.039
        source ~/py36/bin/activate
    """

    if args.id:
        import operetta as o
        import numpy as np

        operetta = o.ConfiguredChannels(args.folder)

        if args.measure or args.render:
            df = operetta.measure(args.id)

        if args.render:
            operetta.save_render(args.id, max_width=300)

        if args.image:
            from skimage.external.tifffile import imsave

            logger.info('------------------------- IMAGE -------------------------')
            image = operetta.max_projection(args.id)
            imsave('%d.tiff' % args.id, np.array(image))

    if args.id:  # done with processing arguments requiring id
        logger.info("done with processing arguments requiring id")
        exit(0)

    if (args.measure or args.render or args.image) and not args.id:
        logger.error("id needed for operation")
        exit(1)

    if args.collect:
        import pandas as pd

        well_cfg_path = os.path.join(args.folder, 'out', 'wells.cfg')
        for root, directories, filenames in os.walk(os.path.join(args.folder, 'out')):
            pd_path = os.path.join(root, 'pandas')
            if os.path.isdir(pd_path):
                df = collect(pd_path)
                if os.path.exists(well_cfg_path):
                    for name, rows, cols in wells_config(well_cfg_path):
                        cnd = df[(df['row'].isin(rows)) & (df['col'].isin(cols))]
                        cnd_path = o.ensure_dir(os.path.join(root, name, 'nuclei.pandas'))
                        pd.to_pickle(cnd, cnd_path)
                else:
                    df = df.pipe(filters.cell)
                    pd.to_pickle(df, os.path.join(root, 'nuclei.pandas'))
