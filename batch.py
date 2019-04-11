import configparser
import json
import logging
import os
import traceback
import warnings

from pandas.errors import EmptyDataError

import operetta as o
from exceptions import BadParameterError

logger = logging.getLogger('batch')
logger.setLevel(logging.DEBUG)
logging.getLogger('matplotlib').setLevel(logging.ERROR)
logging.getLogger('shapely').setLevel(logging.ERROR)
warnings.simplefilter(action='ignore', category=FutureWarning)


def batch_process_operetta_folder(path):
    operetta = o.ConfiguredChannels(path)
    outdf = pd.DataFrame()
    for row, col, fid in operetta.stack_generator():
        try:
            df = operetta.measure(row, col, fid)
            outdf = outdf.append(df, ignore_index=True, sort=False)
        except o.NoSamplesError as e:
            logger.error(e)
            traceback.print_stack()

    if not outdf.empty: pd.to_pickle(outdf, operetta.save_path('nuclei.pandas'))


def batch_render(images_path):
    operetta = o.ConfiguredChannels(images_path)
    for row, col, fid in operetta.stack_generator():
        operetta.save_render(row, col, fid, max_width=300)


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
                    # logger.warning('found empty csv file: %s' % filename)
                    pass
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
    parser.add_argument('folder', metavar='F', type=str,
                        help='folder where operetta images reside')
    parser.add_argument('--render', action='store_true',
                        help='render images (in a folder called render up in the hierarchy)')
    parser.add_argument('--image', action='store_true',
                        help='retrieve image of the stack with id extracted from --id into a tiff file')
    parser.add_argument('--plot', action='store_true',
                        help='plot all graphs')
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

    if args.measure and args.id:
        raise BadParameterError("measure and hpc modes are not allowed in the same command call")

    if args.id:
        import operetta as o
        import numpy as np

        operetta = o.ConfiguredChannels(args.folder)

        if args.render:
            df = operetta.measure(args.id)
            operetta.samples = df
            operetta.save_render(args.id, max_width=300)

        if args.image:
            from skimage.external.tifffile import imsave

            logger.info('------------------------- IMAGE -------------------------')
            image = operetta.max_projection(args.id)
            imsave('%d.tiff' % args.id, np.array(image))

            # if hasattr(operetta, 'flatfield_profiles'):
            #     for plane in ['Background', 'Foreground']:
            #         ffp = [p[plane]['Profile']['Image'] for p in operetta.flatfield_profiles]
            #         imsave('ffp-%s.tiff' % plane.lower(), np.array(ffp))

    if args.collect and not args.id:
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
                    pd.to_pickle(df, os.path.join(root, 'nuclei.pandas'))

    if args.measure and not args.id:
        import operetta as o
        import pandas as pd

        logger.info('------------------------- MEASURING -------------------------')
        logger.debug(args.folder)
        batch_process_operetta_folder(args.folder)

    if args.image and not args.id:
        import operetta as o
        import numpy as np
        from skimage.external.tifffile import imsave

        operetta = o.ConfiguredChannels(args.folder)
        for row, col, fid in operetta.stack_generator():
            image = operetta.max_projection(row, col, fid)
            name = 'r%d-c%d-i%d.tiff' % (row, col, fid)
            l = operetta.layout[(operetta.layout['row'] == row) & (operetta.layout['col'] == col)]
            _folder = '%s%d - %s' % (l['Cell Type'].values[0], l['Cell Count'].values[0] * 10, l['Compound'].values[0])
            destination_path = o.ensure_dir(os.path.join(operetta.render_path, _folder, name))
            imsave(destination_path, np.array(image))

    if args.render and not args.id:
        import operetta as o

        for root, directories, filenames in os.walk(os.path.join(args.folder)):
            pd_path = os.path.join(root, 'out', 'nuclei.pandas')
            if os.path.exists(pd_path):
                try:
                    batch_render(root)
                except o.NoSamplesError as e:
                    logger.error(e)
            else:
                for dir in directories:
                    try:
                        pd_path = os.path.join(args.folder, 'out', dir)
                        batch_render(pd_path)
                    except o.NoSamplesError as e:
                        logger.error(e)

    if args.plot and not args.id:
        import matplotlib.pyplot as plt
        import pandas as pd
        import plots as p
        import operetta as o

        pd_path = os.path.join(args.folder, 'out', 'nuclei.pandas')
        df = pd.read_pickle(pd_path)
        pd_path = os.path.join(args.folder, 'out', 'nuclei.csv')
        df.to_csv(pd_path)

        fig = plt.figure(figsize=(8, 8))
        p.facs(df, ax=fig.gca())
        path = o.ensure_dir(os.path.join(args.folder, 'out', 'graphs', 'facs.pdf'))
        fig.savefig(path)
