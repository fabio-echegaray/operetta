import argparse
import logging

import numpy as np
import pandas as pd

import measurements as m
import operetta as o

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('hhlab')


def batch_process_operetta_folder(path):
    operetta = o.Montage(path)
    outdf = pd.DataFrame()
    for row, col, fid in operetta.stack_generator():
        logger.info('%d %d %d' % (row, col, fid))
        #     operetta.save_render(row, col, fid,max_width=300)
        hoechst, tubulin, pericentrin, edu = operetta.max_projection(row, col, fid)
        r = 30  # [um]
        resolution = 1550.3e-4
        imgseg, props = m.nuclei_segmentation(hoechst, radius=r * resolution)
        operetta.add_mesurement(row, col, fid, 'nuclei found', len(np.unique(imgseg)))

        if len(props) > 0:
            outdf = outdf.append(props)

            # self.nuclei_features = m.nuclei_features(imgseg, area_thresh=(r * self.resolution) ** 2 * np.pi)
            nuclei = m.nuclei_features(imgseg)
            for i, n in enumerate(nuclei):
                n['id'] = i

            cells, _ = m.cell_boundary(tubulin, hoechst)

            samples, df = m.measure_into_dataframe(hoechst, pericentrin, edu, nuclei, cells, resolution)
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
        logger.info('%d %d %d' % (row, col, fid))
        operetta.save_render(row, col, fid, max_width=300)


if __name__ == '__main__':
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

    # df=None
    if args.measure_into_dataframe:
        df = batch_process_operetta_folder(args.folder)

    if args.render and len(args.pandas) > 0:
        batch_render(args.pandas, args.folder)
