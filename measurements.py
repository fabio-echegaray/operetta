import logging
import os
from math import sqrt

import matplotlib.pyplot as plt
import numpy as np
import skimage.color as color
import skimage.feature as feature
import skimage.filters as filters
import skimage.measure as measure
import skimage.morphology as morphology
import skimage.segmentation as segmentation

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('hhlab')


def nuclei_segmentation(image, ax=None):
    image_gray = color.rgb2gray(image)

    filename = 'out/blobs.npy'
    if os.path.exists(filename):
        logger.info('reading blob file data')
        blobs_log = np.load(filename)
    else:
        logger.info('applying blobs algorithm')
        blobs_log = feature.blob_log(image_gray, max_sigma=30, num_sigma=10, threshold=.1)
        # Compute radii in the 3rd column.
        blobs_log[:, 2] = blobs_log[:, 2] * sqrt(2)

        np.save(filename, blobs_log)

    # apply threshold
    thresh_otsu = filters.threshold_otsu(image)
    thresh = image >= thresh_otsu
    thresh = morphology.remove_small_holes(thresh)
    thresh = morphology.remove_small_objects(thresh)
    # thresh = morphology.closing(image > thresh, morphology.square(3))

    # remove artifacts connected to image border
    cleared = segmentation.clear_border(thresh)

    radii_valid = blobs_log[:, 2] > 20
    if ax is not None:
        ax.imshow(cleared)
        radii = blobs_log[radii_valid]
        excluded = blobs_log[~radii_valid]
        for blob in radii:
            y, x, r = blob
            c = plt.Circle((x, y), r, color='lime', linewidth=1, fill=False)
            ax.add_patch(c)
        for blob in excluded:
            y, x, r = blob
            c = plt.Circle((x, y), r, color='red', linewidth=0.2, fill=False)
            ax.add_patch(c)

        ax.set_title('hoechst blobs')
        ax.set_axis_off()

    return cleared, radii_valid


def nuclei_features(image, ax=None, area_thresh=100):
    def polygon_area(x, y):
        return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

    # Display the image and plot all contours found
    contours = measure.find_contours(image, 0.9)

    _list = list()
    for k, contr in enumerate(contours):
        if polygon_area(contr[:, 0], contr[:, 1]) > area_thresh:
            # contr[:, 0] = -contr[:, 0] + image.shape[1]
            _list.append({
                'id': k,
                # 'properties': region,
                'contour': contr
            })
            if ax is not None:
                ax.plot(contr[:, 1], contr[:, 0], linewidth=1)

    return _list


def centrosomes(image, ax=None, max_sigma=1):
    blobs_log = feature.blob_log(image, max_sigma=max_sigma, num_sigma=10, threshold=.1)
    logger.debug(blobs_log)
    # Compute radii in the 3rd column.
    blobs_log[:, 2] = blobs_log[:, 2] * sqrt(2)

    return blobs_log
