import logging
import os
import traceback

import matplotlib.pyplot as plt

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('hhlab')
logging.getLogger('matplotlib').setLevel(logging.ERROR)


def batch_process_operetta_folder(path, row=None, col=None, name=None):
    operetta = o.Montage(path, row=row, col=col, name=name)
    outdf = pd.DataFrame()
    for row, col, fid in operetta.stack_generator():
        try:
            hoechst, tubulin, pericentrin, edu = operetta.max_projection(row, col, fid)
            r = 10  # [um]
            pix_per_um = operetta.pix_per_um

            imgseg, nuclei = m.nuclei_segmentation(hoechst, radius=r * pix_per_um)
            n_nuclei = len(np.unique(imgseg))
            if n_nuclei > 1:
                operetta.add_mesurement(row, col, fid, 'nuclei found', n_nuclei)

                for i, n in enumerate(nuclei):
                    n['id'] = i

                cells, _ = m.cell_boundary(tubulin, hoechst)

                samples, df = m.measure_into_dataframe(hoechst, pericentrin, edu, tubulin, nuclei, cells, pix_per_um)
                if len(df) > 0:
                    df['fid'] = fid
                    df['row'] = row
                    df['col'] = col
                    outdf = outdf.append(df, ignore_index=True, sort=False)
        except o.NoSamplesError as e:
            logger.error(e)
            traceback.print_stack()

    pd.to_pickle(outdf, operetta.save_path(file='nuclei.pandas'))
    operetta.files.to_csv(operetta.save_path(file='operetta.csv'))
    return outdf


def batch_render(df_path, images_path, name=None):
    operetta = o.FourChannels(df_path, images_path)
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
    args = parser.parse_args()

    import numpy as np
    import pandas as pd

    import measurements as m
    import plots as p
    import operetta as o

    """
        MAKE SURE YOU DON'T FORGET TO CALL
        module load python/intel/3.6.039
        # virtualenv --python=/mnt/pactsw/python/2.7.12/bin/python ~/py27
        source ~/py36/bin/activate
    """
    if args.measure:
        logger.debug('------------------------- MEASURING -------------------------')
        logger.debug(args.folder)
        logger.debug(args.column)
        logger.debug(args.name)
        df = batch_process_operetta_folder(args.folder, col=args.column, name=args.name)

    if args.render:
        for root, directories, filenames in os.walk(os.path.join(args.folder, 'out')):
            for dir in directories:
                try:
                    pd_path = os.path.join(args.folder, 'out', dir, 'nuclei.pandas')
                    batch_render(pd_path, args.folder)
                except o.NoSamplesError as e:
                    logger.error(e)

    if args.plot:
        pd_path = os.path.join(args.folder, 'out/nuclei.pandas')
        df = pd.read_pickle(pd_path)
        pd_path = os.path.join(args.folder, 'out/nuclei.csv')
        df.to_csv(pd_path)
        print(df)

        fig = plt.figure(figsize=(8, 8))
        p.facs(df, ax=fig.gca())
        path = o.ensure_dir(os.path.join(args.folder, 'out/graphs/facs.pdf'))
        fig.savefig(path)
