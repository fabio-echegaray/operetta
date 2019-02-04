import logging
import math
from math import sqrt

import cv2
import numpy as np
import scipy.ndimage as ndi
import skimage.draw as draw
import skimage.exposure as exposure
import skimage.feature as feature
import skimage.filters as filters
import skimage.measure as measure
import skimage.morphology as morphology
import skimage.segmentation as segmentation
import skimage.transform as tf
from shapely.geometry.polygon import Polygon
from scipy.ndimage.morphology import distance_transform_edt

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('hhlab')


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


def integral_over_surface(image, polygon):
    c, r = polygon.boundary.xy
    rr, cc = draw.polygon(r, c)

    try:
        ss = np.sum(image[rr, cc])
        return ss
    except Exception:
        logger.warning('integral_over_surface measured incorrectly')
        return np.nan


def nuclei_segmentation(image, radius=10):
    # apply threshold
    logger.debug('thresholding images')
    thresh_val = filters.threshold_otsu(image)
    thresh = image >= thresh_val
    thresh = morphology.remove_small_holes(thresh)
    thresh = morphology.remove_small_objects(thresh)

    # remove artifacts connected to image border
    cleared = segmentation.clear_border(thresh)

    if len(cleared[cleared > 0]) == 0: return None, None

    # logger.debug('computing Lapacian of Gaussian for image')
    # image = exposure.equalize_hist(image)  # improves detection
    # image_gray = color.rgb2gray(image)
    # markers = np.zeros(image.shape, dtype=np.int8)
    # points = feature.blob_log(image_gray, min_sigma=0.8 * radius, max_sigma=radius, num_sigma=10, threshold=.1)
    # # points = feature.blob_dog(image_gray, min_sigma=0.5 * radius, max_sigma=radius, threshold=1.0)
    # print(points)
    # if len(points) == 0:
    #     logger.info('no nuclei found for current stack')
    #     return None, None
    # for k, (r, c, sg) in enumerate(points, start=1):
    #     markers[int(r), int(c)] = k

    distance = distance_transform_edt(cleared)
    local_maxi = feature.peak_local_max(distance, indices=False, labels=cleared,
                                        min_distance=radius / 4, exclude_border=False)
    markers, num_features = ndi.label(local_maxi)
    if num_features == 0:
        logger.info('no nuclei found for current stack')
        return None, None

    labels = morphology.watershed(-distance, markers, watershed_line=True, mask=cleared)

    logger.info('nuclei_features')

    # store all contours found
    contours = measure.find_contours(labels, 0.9)
    tform = tf.SimilarityTransform(rotation=math.pi / 2)

    _list = list()
    for k, contr in enumerate(contours):
        contr = tform(contr)
        contr[:, 0] *= -1
        _list.append({
            'id': k,
            'boundary': Polygon(contr)
        })

    return labels, _list


def centrosomes(image, max_sigma=1):
    blobs_log = feature.blob_log(image, min_sigma=0.05, max_sigma=max_sigma, num_sigma=10, threshold=.017)
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
    ret, bin2 = cv2.threshold(blur, threshold, 255, cv2.THRESH_OTSU)
    # ret, bin2 = cv2.threshold(blur, 70, 255, cv2.THRESH_BINARY)

    if markers is None:
        # get markers for watershed from hoescht channel
        hoechst_8 = cv2.convertScaleAbs(hoechst, alpha=(255.0 / 65535.0))
        blur_nuc = cv2.GaussianBlur(hoechst_8, (ksize, ksize), 0)
        ret, bin_nuc = cv2.threshold(blur_nuc, 0, 255, cv2.THRESH_OTSU)
        markers = ndi.label(bin_nuc)[0]

    gabor_proc = gabor
    labels = morphology.watershed(-gabor_proc, markers, mask=bin2)

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

    return boundaries_list, gabor_proc
