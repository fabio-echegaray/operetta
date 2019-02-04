import logging
import os
import traceback
import warnings

from exceptions import BadParameterError

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('batch')
logging.getLogger('matplotlib').setLevel(logging.ERROR)
warnings.simplefilter(action='ignore', category=FutureWarning)


def batch_process_operetta_folder(path, row=None, col=None, name=None):
    operetta = o.FourChannels(path, row=row, col=col, condition_name=name)
    outdf = pd.DataFrame()
    for row, col, fid in operetta.stack_generator():
        try:
            df = operetta.measure(row, col, fid)
            outdf = outdf.append(df, ignore_index=True, sort=False)
        except o.NoSamplesError as e:
            logger.error(e)
            traceback.print_stack()

    if not outdf.empty: pd.to_pickle(outdf, operetta.save_path('nuclei.pandas'))


def batch_render(images_path, name=None):
    operetta = o.FourChannels(images_path)
    for row, col, fid in operetta.stack_generator():
        operetta.save_render(row, col, fid, max_width=300)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Process operetta images.')
    parser.add_argument('folder', metavar='F', type=str,
                        help='folder where operetta images reside')
    parser.add_argument('--column', type=int,
                        help='column to restrict the analysis on')
    parser.add_argument('--name', type=str,
                        help='name for the condition of the experiment')
    parser.add_argument('--render', action='store_true',
                        help='render images (in a folder called render up in the hierarchy)')
    parser.add_argument('--plot', action='store_true',
                        help='plot all graphs')
    parser.add_argument('--measure', action='store_true',
                        help='measure_into_dataframe features on the dataset')
    parser.add_argument('--id', type=int,
                        help='measure an image of the stack with id into a csv file')
    args = parser.parse_args()

    """
        MAKE SURE YOU DON'T FORGET TO CALL
        # virtualenv --python=/mnt/pactsw/python/2.7.12/bin/python ~/py27
        module load python/intel/3.6.039
        source ~/py36/bin/activate
    """

    if args.measure and args.id:
        raise BadParameterError("measure and hpc modes are not allowed in the same command call")

    if args.id:
        import operetta as o

        operetta = o.FourChannels(args.folder, col=args.column, condition_name=args.name)
        operetta.measure(args.id)

    if args.measure:
        import operetta as o
        import pandas as pd

        logger.debug('------------------------- MEASURING -------------------------')
        logger.debug(args.folder)
        logger.debug(args.column)
        logger.debug(args.name)
        batch_process_operetta_folder(args.folder, col=args.column, name=args.name)

    if args.render:
        import operetta as o

        for root, directories, filenames in os.walk(os.path.join(args.folder, 'out')):
            pd_path = os.path.join(root, 'nuclei.pandas')
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

    if args.plot:
        import matplotlib.pyplot as plt
        import pandas as pd
        import plots as p
        import operetta as o

        pd_path = os.path.join(args.folder, 'out/nuclei.pandas')
        df = pd.read_pickle(pd_path)
        pd_path = os.path.join(args.folder, 'out/nuclei.csv')
        df.to_csv(pd_path)

        fig = plt.figure(figsize=(8, 8))
        p.facs(df, ax=fig.gca())
        path = o.ensure_dir(os.path.join(args.folder, 'out/graphs/facs.pdf'))
        fig.savefig(path)
