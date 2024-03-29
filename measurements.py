import itertools
import logging
import math
from math import sqrt

import cv2
import numpy as np
import pandas as pd
import scipy.ndimage as ndi
import skimage.draw as draw
import skimage.exposure as exposure
import skimage.feature as feature
import skimage.filters as filters
import skimage.measure as measure
import skimage.morphology as morphology
import skimage.segmentation as segmentation
import skimage.transform as tf
from shapely.geometry import Polygon, LineString
from scipy.ndimage.morphology import distance_transform_edt
from shapely.geometry.point import Point
from shapely import affinity
from shapely.wkt import dumps
from shapely.geometry import LineString, MultiLineString, Polygon

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('hhlab')

REJECTION_TOUCHING_FRAME = -1
REJECTION_NO_NUCLEUS = -2
REJECTION_TWO_NUCLEI = -3
REJECTION_CELL_TOO_BIG = -4


def eng_string(x, format='%s', si=False):
    '''
    Returns float/int value <x> formatted in a simplified engineering format -
    using an exponent that is a multiple of 3.

    format: printf-style string used to format the value before the exponent.

    si: if true, use SI suffix for exponent, e.g. k instead of e3, n instead of
    e-9 etc.

    E.g. with format='%.2f':
        1.23e-08 => 12.30e-9
             123 => 123.00
          1230.0 => 1.23e3
      -1230000.0 => -1.23e6

    and with si=True:
          1230.0 => 1.23k
      -1230000.0 => -1.23M
    '''
    sign = ''
    if x == 0: return ('%s' + format) % (sign, 0)
    if x < 0:
        x = -x
        sign = '-'
    exp = int(math.floor(math.log10(x)))
    exp3 = exp - (exp % 3)
    x3 = x / (10 ** exp3)

    if si and exp3 >= -24 and exp3 <= 24 and exp3 != 0:
        exp3_text = 'yzafpnum kMGTPEZY'[int((exp3 - (-24)) / 3)]
    elif exp3 == 0:
        exp3_text = ''
    else:
        exp3_text = 'e%s' % exp3

    return ('%s' + format + '%s') % (sign, x3, exp3_text)


def vector_column_to_long_fmt(a, val_col, ix_col):
    # transform signal and domain vectors into long format (see https://stackoverflow.com/questions/27263805
    b = pd.DataFrame({
        col: pd.Series(data=np.repeat(a[col].values, a[val_col].str.len()))
        for col in a.columns.drop([val_col, ix_col])}
    ).assign(**{ix_col: np.concatenate(a[ix_col].values), val_col: np.concatenate(a[val_col].values)})[a.columns]
    return b


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


def integral_over_surface(image, polygon: Polygon):
    assert polygon.is_valid, "Polygon is invalid"

    try:
        c, r = polygon.exterior.xy
        rr, cc = draw.polygon(r, c)
        ss = np.sum(image[rr, cc])
        for interior in polygon.interiors:
            c, r = interior.xy
            rr, cc = draw.polygon(r, c)
            ss -= np.sum(image[rr, cc])
        return ss
    except Exception:
        logger.warning('integral_over_surface measured incorrectly')
        return np.nan


def histogram_of_surface(image, polygon: Polygon, bins=None):
    assert polygon.is_valid, "Polygon is invalid"

    try:
        hh = np.zeros(shape=image.shape, dtype=np.bool)
        c, r = polygon.exterior.xy
        rr, cc = draw.polygon(r, c)
        hh[rr, cc] = True
        for interior in polygon.interiors:
            c, r = interior.xy
            rr, cc = draw.polygon(r, c)
            hh[rr, cc] = False
        hist, edges = np.histogram(image[hh].ravel(), bins)
        return hist, edges
    except Exception:
        logger.warning('histogram_of_surface measured incorrectly')
        return np.nan, np.nan


def integral_over_line(image, line: LineString):
    assert line.is_valid, "LineString is invalid"
    try:
        for pt0, pt1 in pairwise(line.coords):
            r0, c0, r1, c1 = np.array(list(pt0) + list(pt1)).astype(int)
            rr, cc = draw.line(r0, c0, r1, c1)
            ss = np.sum(image[rr, cc])
            return ss
    except Exception:
        logger.warning('integral_over_line measured incorrectly')
        return np.nan


def generate_mask_from(polygon: Polygon, shape=None):
    if shape is None:
        minx, miny, maxx, maxy = polygon.bounds
        image = np.zeros((maxx - minx, maxy - miny), dtype=np.bool)
    else:
        image = np.zeros(shape, dtype=np.bool)

    c, r = polygon.boundary.xy
    rr, cc = draw.polygon(r, c)
    image[rr, cc] = True
    return image


def nuclei_segmentation(image, compute_distance=False, radius=10, simp_px=None):
    # apply threshold
    logger.debug('thresholding images')
    thresh_val = filters.threshold_otsu(image)
    thresh = image >= thresh_val
    thresh = morphology.remove_small_holes(thresh)
    thresh = morphology.remove_small_objects(thresh)

    # remove artifacts connected to image border
    cleared = segmentation.clear_border(thresh)

    if len(cleared[cleared > 0]) == 0: return None, None

    if compute_distance:
        distance = distance_transform_edt(cleared)
        local_maxi = feature.peak_local_max(distance, indices=False, labels=cleared,
                                            min_distance=radius / 4, exclude_border=False)
        markers, num_features = ndi.label(local_maxi)
        if num_features == 0:
            logger.info('no nuclei found for current stack')
            return None, None

        labels = morphology.watershed(-distance, markers, watershed_line=True, mask=cleared)
    else:
        labels = cleared

    logger.info('storing nuclei features')

    # store all contours found
    contours = measure.find_contours(labels, 0.9)
    tform = tf.SimilarityTransform(rotation=math.pi / 2)

    _list = list()
    for k, contr in enumerate(contours):
        contr = tform(contr)
        contr[:, 0] *= -1
        pol = Polygon(contr)
        if simp_px is not None:
            pol = pol.simplify(simp_px, preserve_topology=True)
        _list.append({
            'id': k,
            'boundary': pol
        })

    return labels, _list


def centrosomes(image, min_size=0.2, max_size=0.5, threshold=0.1):
    blobs_log = feature.blob_log(image, min_sigma=min_size, max_sigma=max_size, num_sigma=10, threshold=threshold)
    # blobs_log = feature.blob_doh(image, min_sigma=0.05, max_sigma=max_sigma, num_sigma=10, threshold=.1)
    # Compute radii in the 3rd column.
    blobs_log[:, 2] = blobs_log[:, 2] * sqrt(2)
    tform = tf.SimilarityTransform(rotation=math.pi / 2)
    blobs_log[:, 0:2] = tform(blobs_log[:, 0:2])
    blobs_log[:, 0] *= -1

    return blobs_log


def cell_boundary(tubulin, hoechst, threshold=80, markers=None):
    def build_gabor_filters():
        filters = []
        ksize = 9
        for theta in np.arange(0, np.pi, np.pi / 8):
            kern = cv2.getGaborKernel((ksize, ksize), 4.0, theta, 6.0, 0.5, 0, ktype=cv2.CV_32F)
            kern /= kern.sum()
            filters.append(kern)
        return filters

    def process_gabor(img, filters):
        accum = np.zeros_like(img)
        for kern in filters:
            fimg = cv2.filter2D(img, cv2.CV_16UC1, kern)
            np.maximum(accum, fimg, accum)
        return accum

    p2 = np.percentile(tubulin, 2)
    p98 = np.percentile(tubulin, 98)
    tubulin = exposure.rescale_intensity(tubulin, in_range=(p2, p98))
    p2 = np.percentile(hoechst, 2)
    p98 = np.percentile(hoechst, 98)
    hoechst = exposure.rescale_intensity(hoechst, in_range=(p2, p98))

    # img = np.maximum(tubulin, hoechst)
    img = tubulin

    img = morphology.erosion(img, morphology.square(3))
    filters = build_gabor_filters()
    gabor = process_gabor(img, filters)

    gabor = cv2.convertScaleAbs(gabor, alpha=(255.0 / 65535.0))
    ret, bin1 = cv2.threshold(gabor, threshold, 255, cv2.THRESH_BINARY)

    # gaussian blur on gabor filter result
    ksize = 31
    blur = cv2.GaussianBlur(bin1, (ksize, ksize), 0)
    ret, cells_mask = cv2.threshold(blur, threshold, 255, cv2.THRESH_OTSU)
    # ret, bin2 = cv2.threshold(blur, 70, 255, cv2.THRESH_BINARY)

    if markers is None:
        # get markers for watershed from hoescht channel
        hoechst_8 = cv2.convertScaleAbs(hoechst, alpha=(255.0 / 65535.0))
        blur_nuc = cv2.GaussianBlur(hoechst_8, (ksize, ksize), 0)
        ret, bin_nuc = cv2.threshold(blur_nuc, 0, 255, cv2.THRESH_OTSU)
        markers = ndi.label(bin_nuc)[0]

    labels = morphology.watershed(-gabor, markers, mask=cells_mask)

    boundaries_list = list()
    # loop over the labels
    for (i, l) in enumerate([l for l in np.unique(labels) if l > 0]):
        # find contour of mask
        cell_boundary = np.zeros(shape=labels.shape, dtype=np.uint8)
        cell_boundary[labels == l] = 255
        cnts = cv2.findContours(cell_boundary.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        contour = cnts[0]

        boundary = np.array([[x, y] for x, y in [i[0] for i in contour]], dtype=np.float32)
        if len(boundary) >= 3:
            boundaries_list.append({'id': l, 'boundary': Polygon(boundary)})

    return boundaries_list, cells_mask > 255


def exclude_contained(polygons):
    if polygons is None: return []
    for p in polygons:
        p['valid'] = True
    for p1, p2 in itertools.combinations(polygons, 2):
        if not p1['valid'] or not p2['valid']: continue
        if p1['boundary'].contains(p2['boundary']):
            p2['valid'] = False
        if p2['boundary'].contains(p1['boundary']):
            p1['valid'] = False
    return [p for p in polygons if p['valid']]


def is_valid_sample(frame_polygon, cell_polygon, nuclei_polygon, nuclei_list=None):
    # check that neither nucleus or cell boundary touch the ends of the frame

    if np.any(np.abs(np.array(cell_polygon.bounds) - np.array(frame_polygon.bounds)) <= 2):
        return False, REJECTION_TOUCHING_FRAME
    if not cell_polygon.contains(nuclei_polygon):
        return False, REJECTION_NO_NUCLEUS

    # make sure that there's only one nucleus inside cell
    if nuclei_list is not None:
        n_nuc = 0
        for nuc in nuclei_list:
            if cell_polygon.contains(nuc['boundary']):
                n_nuc += 1
        if n_nuc > 1:
            return False, REJECTION_TWO_NUCLEI

    # nucleus area should be at least three to four times the are of the cell
    area_ratio = cell_polygon.area / nuclei_polygon.area
    if area_ratio > 5:
        return False, REJECTION_CELL_TOO_BIG
    logger.debug('sample accepted with an area ratio of %0.2f' % area_ratio)

    return True, None


def measure_lines_around_polygon(image, polygon, pix_per_um=1, n_lines=3, rng_thick=3, dl=None):
    width, height = image.shape
    rng_thick *= pix_per_um
    if dl is not None:
        dl *= pix_per_um
    angle_delta = 2 * np.pi / n_lines
    frame = Polygon([(0, 0), (0, width), (height, width), (height, 0)]).buffer(-rng_thick)

    minx, miny, maxx, maxy = polygon.bounds
    radius = max(maxx - minx, maxy - miny)
    for k, angle in enumerate([angle_delta * i for i in range(n_lines)]):
        ray = LineString([polygon.centroid,
                          (polygon.centroid.x + radius * np.cos(angle),
                           polygon.centroid.y + radius * np.sin(angle))])
        r_seg = ray.intersection(polygon)

        if r_seg.is_empty:
            continue
        if type(r_seg) == MultiLineString:
            r_seg = r_seg[0]

        pt = Point(r_seg.coords[-1])
        if not frame.contains(pt):
            continue

        for pt0, pt1 in pairwise(polygon.exterior.coords):
            # if pt.touches(LineString([pt0, pt1])):
            if Point(pt).distance(LineString([pt0, pt1])) < 1e-6:
                # compute normal vector
                dx = pt1[0] - pt0[0]
                dy = pt1[1] - pt0[1]
                # touching point of the polygon line segment
                px, py = pt.x, pt.y
                # normalize normal vector
                alpha = np.arctan2(dy, dx)
                dx, dy = np.cos(alpha), np.sin(alpha)

                if dl is not None:
                    nsteps = int(rng_thick / dl)
                    x0 = px + dy * rng_thick / 2
                    y0 = py - dx * rng_thick / 2
                    _x = -dl * dy
                    _y = dl * dx
                    cc, rr = list(), list()
                    for n in range(nsteps):
                        # note that rows and cols are interchanged to account for another rotation
                        rr.append(x0 + n * _x)
                        cc.append(y0 + n * _y)
                    rr = np.array(rr, dtype=np.int16)
                    cc = np.array(cc, dtype=np.int16)

                else:
                    # note that rows and cols are interchanged to account for another rotation
                    r0, c0, r1, c1 = np.array([px + dy * rng_thick / 2, py - dx * rng_thick / 2,
                                               px - dy * rng_thick / 2, py + dx * rng_thick / 2]).astype(int)
                    rr, cc = draw.line(r0, c0, r1, c1)
                lin = LineString([(rr[0], cc[0]), (rr[-1], cc[-1])])

                yield lin, image[cc, rr]
