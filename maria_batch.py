import logging
import os

from shapely.geometry.polygon import Polygon
from shapely import affinity
import matplotlib.pyplot as plt
from matplotlib.ticker import EngFormatter
import shapely.geometry
import shapely.wkt
import shapely.wkt

from gui import convert_to, meter, pix, um

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('hhlab')
logging.getLogger('matplotlib').setLevel(logging.ERROR)


def is_valid_sample(image, nuclei_polygon, nuclei_list=None, pix_per_um=1):
    # check that neither nucleus or cell boundary touch the ends of the frame
    maxw, maxh = image.shape
    frame = Polygon([(0, 0), (0, maxw), (maxh, maxw), (maxh, 0)])

    if pix_per_um != 1:
        scale = pix_per_um
        nuclei_polygon = affinity.scale(nuclei_polygon, xfact=scale, yfact=scale, origin=(0, 0, 0))

    # FIXME: not working
    if frame.touches(nuclei_polygon.buffer(2)):
        # if frame.intersects(nuclei_polygon.buffer(2)):
        logger.debug('sample rejected because it was touching the frame')
        return False

    return True


def measure_into_dataframe(dapi, edu, phospho_rb, nuclei, pix_per_um):
    df = pd.DataFrame()
    scale = 1. / pix_per_um
    for nucleus in nuclei:
        # x0, y0, xf, yf = [int(u) for u in nucleus['boundary'].bounds]
        nucl_bnd = nucleus['boundary']

        if is_valid_sample(dapi, nucl_bnd, nuclei_list=nuclei):
            prb_int = m.integral_over_surface(phospho_rb, nucl_bnd)
            edu_int = m.integral_over_surface(edu, nucl_bnd)
            dna_int = m.integral_over_surface(dapi, nucl_bnd)

            nucl_bndum = affinity.scale(nucl_bnd, xfact=scale, yfact=scale, origin=(0, 0, 0))

            d = pd.DataFrame(data={
                'id': [nucleus['id']],
                'dna_int': [dna_int],
                'edu_int': [edu_int],
                'phospho_rb_int': [prb_int],
                'dna_dens': [dna_int / nucl_bnd.area],
                'phospho_rb_dens': [prb_int / nucl_bnd.area],
                'nucleus': nucl_bndum.wkt,
            })
            df = df.append(d, ignore_index=True, sort=False)

    return df


def batch_process_operetta_folder(path):
    operetta = o.Montage(path)
    outdf = pd.DataFrame()
    for row, col, fid in operetta.stack_generator():
        dapi, phospho_rb, edu = operetta.max_projection(row, col, fid)
        r = 30  # [um]
        um_per_pix = convert_to(1.8983367649421008E-07 * meter / pix, um / pix).n()
        pix_per_um = 1 / um_per_pix
        pix_per_um = float(pix_per_um.args[0])

        imgseg = m.nuclei_segmentation(dapi, radius=r * pix_per_um)
        n_nuclei = len(np.unique(imgseg))
        operetta.add_mesurement(row, col, fid, 'nuclei found', n_nuclei)

        if n_nuclei > 1:
            nuclei = m.nuclei_features(imgseg)
            for i, n in enumerate(nuclei):
                n['id'] = i

            df = measure_into_dataframe(dapi, edu, phospho_rb, nuclei, pix_per_um)
            if len(df) > 0:
                df['fid'] = fid
                df['row'] = row
                df['col'] = col
                outdf = outdf.append(df, ignore_index=True, sort=False)

    pd_path = os.path.join(path, 'out/nuclei.pandas')
    pd.to_pickle(outdf, o.ensure_dir(pd_path))
    o_path = os.path.join(path, 'out/operetta.csv')
    operetta.files.to_csv(o.ensure_dir(o_path))
    return outdf


def batch_render(df_path, images_path):
    operetta = o.ThreeChannels(df_path, images_path)
    for row, col, fid in operetta.stack_generator():
        operetta.save_render(row, col, fid, max_width=300)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Process operetta images.')
    parser.add_argument('folder', metavar='F', type=str,
                        help='folder where operetta images reside')
    parser.add_argument('--render', action='store_true',
                        help='render images (in a folder called render up in the hierarchy)')
    parser.add_argument('--plot', action='store_true',
                        help='plot all graphs')
    parser.add_argument('--measure_into_dataframe', action='store_true',
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
    # df=None
    if args.measure_into_dataframe:
        df = batch_process_operetta_folder(args.folder)

    if args.render:
        pd_path = os.path.join(args.folder, 'out/nuclei.pandas')
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

        path = o.ensure_dir(os.path.join(args.folder, 'graphs/facs.pdf'))
        fig.savefig(path)
