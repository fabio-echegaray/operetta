import logging

from gui import convert_to, meter, pix, um

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('hhlab')


def batch_process_operetta_folder(path):
    operetta = o.Montage(path)
    outdf = pd.DataFrame()
    for row, col, fid in operetta.stack_generator():
        hoechst, tubulin, pericentrin, edu = operetta.max_projection(row, col, fid)
        r = 30  # [um]
        um_per_pix = convert_to(1.8983367649421008E-07 * meter / pix, um / pix).n()
        pix_per_um = 1 / um_per_pix
        pix_per_um = float(pix_per_um.args[0])

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

    pd.to_pickle(outdf, 'out/nuclei.pandas')
    operetta.files.to_csv('out/operetta.csv')
    return outdf


def batch_render(df_path, images_path):
    operetta = o.Dataframe(df_path, images_path)
    for row, col, fid in operetta.stack_generator():
        operetta.save_render(row, col, fid, max_width=300)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Process operetta images.')
    parser.add_argument('folder', metavar='F', type=str,
                        help='folder where operetta images reside')
    parser.add_argument('--pandas', type=str,
                        help='pandas dataframe file obtained from previous analysis')
    parser.add_argument('--render', action='store_true',
                        help='render images (in a folder called render up in the hierarchy)')
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

    if args.render and len(args.pandas) > 0:
        batch_render(args.pandas, args.folder)
