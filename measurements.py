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
from shapely.geometry.point import Point
from shapely.geometry.polygon import Polygon

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

    # import matplotlib.pyplot as plt
    # f1 = plt.figure(10)
    # f1.gca().imshow(image)
    # f2 = plt.figure(20)
    # msk = np.zeros(image.shape, dtype=np.uint8)
    # msk[rr, cc] = 1
    # f2.gca().imshow(msk)
    # plt.show(block=False)
    return np.sum(image[rr, cc])


def nuclei_segmentation(image, radius=30):
    # image_gray = color.rgb2gray(image)
    # blobs_log = feature.blob_log(image_gray, min_sigma=0.8 * radius, max_sigma=1.2 * radius, num_sigma=10, threshold=.1)
    # # Compute radii in the 3rd column.
    # blobs_log[:, 2] = blobs_log[:, 2] * sqrt(2)

    # apply threshold
    logger.info('thresholding images')
    thresh_val = filters.threshold_otsu(image)
    thresh = image >= thresh_val
    thresh = morphology.remove_small_holes(thresh)
    thresh = morphology.remove_small_objects(thresh)
    # thresh = morphology.closing(image > thresh, morphology.square(3))

    # remove artifacts connected to image border
    cleared = segmentation.clear_border(thresh)
    label_image = measure.label(cleared)

    logger.info('computing properties')
    properties = list()
    indices = list()
    # TODO: need integral of pixel intensity over surface
    columns = ('area', 'bbox_area',
               'centroid_x', 'centroid_u', 'eccentricity', 'equivalent_diameter',
               'euler_number', 'extent',
               'max_intensity', 'mean_intensity', 'min_intensity', 'orientation', 'perimeter')
    for p in measure.regionprops(label_image, intensity_image=image):
        properties.append([p.area,
                           # p.bbox,
                           p.bbox_area,
                           p.centroid[0],
                           p.centroid[1],
                           p.eccentricity,
                           p.equivalent_diameter,
                           p.euler_number,
                           p.extent,
                           # p.inertia_tensor,
                           p.max_intensity,
                           p.mean_intensity,
                           p.min_intensity,
                           p.orientation,
                           p.perimeter,
                           ])
        indices.append(p.label)
    if not len(indices):
        return label_image, pd.DataFrame([], index=[])
    else:
        indices = pd.Index(indices, name='label')
        properties = pd.DataFrame(properties, index=indices, columns=columns)

        return label_image, properties


def nuclei_features(image, ax=None, area_thresh=100):
    logger.info('nuclei_features')

    def polygon_area(x, y):
        return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

    # Display the image and plot all contours found
    contours = measure.find_contours(image, 0.9)
    tform = tf.SimilarityTransform(rotation=math.pi / 2)

    _list = list()
    for k, contr in enumerate(contours):
        if polygon_area(contr[:, 0], contr[:, 1]) > area_thresh:
            contr = tform(contr)
            contr[:, 0] *= -1
            _list.append({
                'id': k,
                'boundary': Polygon(contr)
            })
            if ax is not None:
                ax.plot(contr[:, 1], contr[:, 0], linewidth=1)

    return _list


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


def is_valid_sample(img, cell_polygon, nuclei_polygon, nuclei_list):
    # check that neither nucleus or cell boundary touch the ends of the frame
    maxw, maxh = img.shape
    frame = Polygon([(0, 0), (0, maxw), (maxh, maxw), (maxh, 0)])
    if frame.touches(cell_polygon) or frame.touches(nuclei_polygon):
        return False
    if not frame.contains(cell_polygon) or not cell_polygon.contains(nuclei_polygon):
        return False

    # make sure that there's only one nucleus inside cell
    n_nuc = 0
    for ncl in nuclei_list:
        nuclei = ncl['boundary']
        if cell_polygon.contains(nuclei):
            n_nuc += 1
    if n_nuc != 1:
        return False

    return True


def measure_into_dataframe(hoechst, pericentrin, edu, tubulin, nuclei, cells, resolution):
    out = list()
    df = pd.DataFrame()
    for nucleus in nuclei:
        x0, y0, xf, yf = [int(u) for u in nucleus['boundary'].bounds]

        # search for closest cell boundary based on centroids
        clls = list()
        for cl in cells:
            clls.append({'id': cl['id'],
                         'boundary': cl['boundary'],
                         'd': cl['boundary'].centroid.distance(nucleus['boundary'].centroid)})
        clls = sorted(clls, key=lambda k: k['d'])

        if is_valid_sample(hoechst, clls[0]['boundary'], nucleus['boundary'], nuclei):
            pericentrin_crop = pericentrin[y0:yf, x0:xf]
            logger.info('applying centrosome algorithm for nuclei %d' % nucleus['id'])
            # self.pericentrin = exposure.equalize_adapthist(pcrop, clip_limit=0.03)
            cntr = centrosomes(pericentrin_crop, max_sigma=resolution * 0.5)
            cntr[:, 0] += x0
            cntr[:, 1] += y0
            cntrsmes = list()
            for k, c in enumerate(cntr):
                pt = Point(c[0], c[1])
                pti = integral_over_surface(pericentrin, pt.buffer(resolution * 5))
                cntrsmes.append({'id': k, 'pt': pt, 'i': pti})
                cntrsmes = sorted(cntrsmes, key=lambda k: k['i'], reverse=True)

            logger.debug('centrosomes {:s}'.format(str(cntrsmes)))

            twocntr = len(cntrsmes) >= 2
            c1 = cntrsmes[0] if len(cntrsmes) > 0 else None
            c2 = cntrsmes[1] if twocntr else None

            nucl_bnd = nucleus['boundary']
            cell_bnd = clls[0]['boundary']

            edu_int = integral_over_surface(edu, nucl_bnd)
            dna_int = integral_over_surface(hoechst, nucl_bnd)
            tubulin_int = integral_over_surface(tubulin, cell_bnd)

            lc = 2 if c2 is not None else 1
            d = pd.DataFrame(data={
                'id': [nucleus['id']],
                'dna_int': [dna_int],
                'edu_int': [edu_int],
                'tubulin_int': [tubulin_int],
                'centrosomes': [lc],
                'c1_int': [c1['i'] if c1 is not None else np.nan],
                'c2_int': [c2['i'] if c2 is not None else np.nan],
                'c1_d_nuc_centr': [nucl_bnd.centroid.distance(c1['pt']) if c1 is not None else np.nan],
                'c2_d_nuc_centr': [nucl_bnd.centroid.distance(c2['pt']) if twocntr else np.nan],
                'c1_d_nuc_bound': [nucl_bnd.exterior.distance(c1['pt']) if c1 is not None else np.nan],
                'c2_d_nuc_bound': [nucl_bnd.exterior.distance(c2['pt']) if twocntr else np.nan],
                'c1_d_cell_centr': [cell_bnd.centroid.distance(c1['pt']) if c1 is not None else np.nan],
                'c2_d_cell_centr': [cell_bnd.centroid.distance(c2['pt']) if twocntr else np.nan],
                'c1_d_cell_bound': [cell_bnd.exterior.distance(c1['pt']) if c1 is not None else np.nan],
                'c2_d_cell_bound': [cell_bnd.exterior.distance(c2['pt']) if twocntr else np.nan],
                'c1_d_c2': [c1['pt'].distance(c2['pt']) if twocntr else np.nan],
                'cell': cell_bnd.wkt,
                'nucleus': nucl_bnd.wkt,
                'c1': c1['pt'].wkt if c1 is not None else None,
                'c2': c2['pt'].wkt if c2 is not None else None,
            })
            df = df.append(d, ignore_index=True, sort=False)

            out.append({'id': nucleus['id'], 'cell': cell_bnd, 'nucleus': nucl_bnd,
                        'centrosomes': [c1, c2], 'edu_int': edu_int, 'dna_int': dna_int})
    return out, df
