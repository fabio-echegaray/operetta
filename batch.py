import logging
import os
import warnings
import enlighten

from pandas.errors import EmptyDataError
import filters

logger = logging.getLogger('batch')
logger.setLevel(logging.DEBUG)

# reduce console output while using batch tool
logging.getLogger('matplotlib').setLevel(logging.ERROR)
logging.getLogger('shapely').setLevel(logging.ERROR)
warnings.simplefilter(action='ignore', category=FutureWarning)


def collect(path, csv_fname="nuclei.pandas.csv"):
    df = pd.DataFrame()
    cols_to_delete = ['nucleus', 'nuc_pix', 'cell', 'cell_pix', 'ring', 'ring_pix']
    manager = enlighten.get_manager()

    csv_file = os.path.join(path, csv_fname)
    if os.path.isfile(csv_file):
        os.remove(csv_file)

    for root, directories, filenames in os.walk(os.path.join(path, "pandas")):
        bar = manager.counter(total=len(filenames), desc='Progress', unit='files')
        for k, filename in enumerate(filenames):
            ext = filename.split('.')[-1]
            if ext == 'csv':
                logger.info("adding %s" % filename)
                try:
                    csv = (pd.read_csv(os.path.join(root, filename))
                           .pipe(filters.cell)
                           .pipe(filters.nucleus, radius_min=4, radius_max=10)
                           .pipe(filters.polsby_popper, column="nucleus")
                           )
                    csv = csv.drop(columns=[c for c in cols_to_delete if c in csv])
                    with open(csv_file, 'a') as f:
                        csv.to_csv(f, mode='a', header=not f.tell())

                except EmptyDataError:
                    logger.warning('found empty csv file: %s' % filename)
                except ValueError:
                    logger.warning('weird... maybe empty output after filter? in %s' % filename)
                    # traceback.print_stack()
            bar.update()
    manager.stop()
    return df


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
                df = collect(root)
