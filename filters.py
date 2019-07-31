import shapely.wkt
import numpy as np
import logging

logger = logging.getLogger('filters')
logger.setLevel(logging.DEBUG)


def cell(df):
    if not "cell" in df:
        return df
    return df[~df["cell"].isna()]


def nucleus(df, radius_min=0, radius_max=np.inf):
    """
    Returns a dataframe with all samples with nucleus area greater than π*radius^2.

    :param df: Input dataframe.
    :param radius_min: Minimum radius of disk in um for area comparison. Areas lesser than this area are discarded.
    :param radius_max: Areas greater than the equivalent disk area of radius radius_max are discarded.
    :return:  Filtered dataframe.
    """
    area_min_thresh = np.pi * radius_min ** 2
    area_max_thresh = np.pi * radius_max ** 2
    logger.info("filtering nuclei with area greater than %0.2f[um^2] and less than %0.2f[um^2]" % (
        area_min_thresh, area_max_thresh))
    n_idx = df.apply(lambda row: area_max_thresh > shapely.wkt.loads(row['nucleus']).area > area_min_thresh, axis=1)
    return df[n_idx]
