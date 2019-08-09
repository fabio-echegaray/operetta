import logging

import shapely.wkt
import numpy as np

logger = logging.getLogger('filters')
logger.setLevel(logging.DEBUG)


def cell(df):
    if "cell" not in df:
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


def polsby_popper(df, column):
    def _pp(_df):
        pol = shapely.wkt.loads(_df[column])
        pp = pol.area * np.pi * 4 / pol.length ** 2
        return pp > 0.8

    logger.info("filtering %s with a Polsby-Popper score greater than %0.2f" % (column, 0.8))
    n_idx = df.apply(_pp, axis=1)
    return df[n_idx]


def histogram(df, edges=None, values=None, agg_fn="sum", edge_min=0, edge_max=np.inf, value_min=0, value_max=np.inf):
    def _hh(_df):
        # hist = np.array(ast.literal_eval(_df[edges]))
        # vals = np.array(ast.literal_eval(_df[values]))
        hist = _df[edges][:-1]
        vals = _df[values]
        ix = np.where((edge_min <= hist) & (hist <= edge_max))
        if agg_fn == "sum":
            aggout = vals[ix].sum()
        elif agg_fn == "max":
            aggout = vals[ix].max()
        elif agg_fn == "min":
            aggout = vals[ix].min()
        elif agg_fn == "avg" or agg_fn == "mean":
            aggout = vals[ix].mean()
        else:
            aggout = 0

        accepted = value_min < aggout < value_max
        # logger.debug("%0.1f < %0.1f(%s) < %0.1f %s" % (value_min, aggout, agg_fn, value_max, accepted))
        return accepted

    if edges not in df or values not in df:
        return df

    logger.info("filtering based on histogram")
    n_idx = df.apply(_hh, axis=1)
    return df[n_idx]


def lines(df):
    # filter rows with all the profiles meeting the criteria
    ix1 = df["value"].apply(lambda v: (v < 500).all())
    ix2 = df["value"].apply(lambda v: (v.max() - v.min()) < 200)
    # ix3 = a["value"].apply(lambda v: (v.max() - v.min()) > 8000)
    df.loc[:, "not_quite"] = ix1 | ix2  # | ix3
    valid_units_ix = ~df.groupby(["unit"])["not_quite"].apply(np.any)
    valid_units = valid_units_ix[valid_units_ix].index.values

    print("before removing incomplete profiles %d" % len(df))
    df = df[df['unit'].isin(valid_units)]
    print("after %d" % len(df))

    # center all the curves on the maximum points
    minlen = df["value"].apply(lambda v: v.shape[0]).min()
    df.loc[:, "xpeak"] = df["value"].apply(lambda v: np.argmax(v))
    df.loc[:, "x"] = df["value"].apply(lambda v: np.arange(start=0, stop=v.shape[0], step=1))
    df.loc[:, "x_center"] = df["x"] - df["xpeak"]
    df.drop(columns=["xpeak", "x"], inplace=True)

    # filter the curves that are too off
    de = minlen * 0.2
    df.loc[:, "off"] = df.apply(lambda r: (r['x_center'].min() > -de) or (r['x_center'].max() < de), axis=1)
    not_off_ix = ~df.groupby(["unit"])["off"].apply(np.any)
    not_off = not_off_ix[not_off_ix].index.values
    print("before removing extreme lines %d" % len(df))
    df = df[df['unit'].isin(not_off)]
    print("after %d" % len(df))
    df.drop(columns=["not_quite", "off"], inplace=True)

    return df
