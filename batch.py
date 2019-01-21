import logging
import os

from shapely.geometry.polygon import Polygon
import matplotlib.pyplot as plt
from matplotlib.ticker import EngFormatter
import shapely.geometry
import shapely.wkt
import shapely.wkt

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('hhlab')
logging.getLogger('matplotlib').setLevel(logging.ERROR)


def batch_process_operetta_folder(path, row=None, col=None, name=None):
    operetta = o.Montage(path, row=row, col=col, name=name)
    outdf = pd.DataFrame()
    for row, col, fid in operetta.stack_generator():
        hoechst, tubulin, pericentrin, edu = operetta.max_projection(row, col, fid)
        r = 30  # [um]
        pix_per_um = operetta.pix_per_um

        imgseg = m.nuclei_segmentation(hoechst, radius=r * pix_per_um)
        n_nuclei = len(np.unique(imgseg))
        operetta.add_mesurement(row, col, fid, 'nuclei found', n_nuclei)

        if n_nuclei > 1:
            # self.nuclei_features = m.nuclei_features(imgseg, area_thresh=(r * self.resolution) ** 2 * np.pi)
            nuclei = m.nuclei_features(imgseg)
            for i, n in enumerate(nuclei):
                n['id'] = i

            cells, _ = m.cell_boundary(tubulin, hoechst)

            samples, df = m.measure_into_dataframe(hoechst, pericentrin, edu, tubulin, nuclei, cells, pix_per_um)
            if len(df) > 0:
                df['fid'] = fid
                df['row'] = row
                df['col'] = col
                outdf = outdf.append(df, ignore_index=True, sort=False)

    pd.to_pickle(outdf, os.path.join(operetta.save_path(), 'nuclei.pandas'))
    operetta.files.to_csv(os.path.join(operetta.save_path(), 'operetta.csv'))
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
    import operetta as o

    """
        MAKE SURE YOU DON'T FORGET TO CALL
        module load python/intel/3.6.039
        # virtualenv --python=/mnt/pactsw/python/2.7.12/bin/python ~/py27
        source ~/py36/bin/activate
    """
    if args.measure:
        df = batch_process_operetta_folder(args.folder, col=args.column, name=args.name)

    if args.render:
        for root, directories, filenames in os.walk(os.path.join(args.folder, 'out')):
            for dir in directories:
                pd_path = os.path.join(args.folder, 'out', dir, 'nuclei.pandas')
                batch_render(pd_path, args.folder)

    if args.plot:
        pd_path = os.path.join(args.folder, 'out/nuclei.pandas')
        df = pd.read_pickle(pd_path)
        pd_path = os.path.join(args.folder, 'out/nuclei.csv')
        df.to_csv(pd_path)
        print(df)

        df["geometry"] = df.apply(lambda row: shapely.geometry.Point(row['dna_int'] / 1e6 / 6, np.log(row['edu_int'])),
                                  axis=1)
        df["cluster"] = -1

        fig = plt.figure(figsize=(8, 8))
        ax = fig.gca()
        ax.set_xlabel('dna [AU]')
        ax.set_ylabel('edu [AU]')
        formatter = EngFormatter(unit='')
        ax.xaxis.set_major_formatter(formatter)
        ax.yaxis.set_major_formatter(formatter)
        ax.set_xlim([1, 8])
        ax.set_ylim([12, 18.5])
        ax.set_aspect('equal')

        df.loc[:, 'phospho_rb_int'] = df['phospho_rb_int'].transform(
            lambda x: (x - x.mean()) / x.std())

        map = ax.scatter(df['dna_int'] / 1e6 / 6, np.log(df['edu_int']), c=df['phospho_rb_int'], alpha=0.1)

        path = o.ensure_dir(os.path.join(args.folder, 'out/graphs/facs.pdf'))
        fig.savefig(path)
