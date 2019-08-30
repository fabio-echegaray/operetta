import logging
import os
import xml.etree.ElementTree

import numpy as np
import skimage.external.tifffile as tf
from czifile import CziFile

logger = logging.getLogger(__name__)


def load_tiff(path):
    _, img_name = os.path.split(path)
    with tf.TiffFile(path) as tif:
        if tif.is_imagej is not None:
            metadata = tif.pages[0].imagej_tags
            dt = metadata['finterval'] if 'finterval' in metadata else 1

            # asuming square pixels
            xr = tif.pages[0].tags['x_resolution'].value
            res = float(xr[0]) / float(xr[1])  # pixels per um
            if metadata['unit'] == 'centimeter':
                res = res / 1e4

            images = None
            if len(tif.pages) == 1:
                if ('slices' in metadata and metadata['slices'] > 1) or (
                        'frames' in metadata and metadata['frames'] > 1):
                    images = tif.pages[0].asarray()
                else:
                    images = [tif.pages[0].asarray()]
            elif len(tif.pages) > 1:
                # frames = np.ndarray((len(tif.pages), tif.pages[0].image_length, tif.pages[0].image_width), dtype=np.int32)
                images = list()
                for i, page in enumerate(tif.pages):
                    images.append(page.asarray())

            return images, res, dt, metadata['frames'], metadata['channels'] if 'channels' in metadata else 1


def load_zeiss(path):
    _, img_name = os.path.split(path)
    with CziFile(path) as czi:
        xmltxt = czi.metadata()
        meta = xml.etree.ElementTree.fromstring(xmltxt)

        # next line is somewhat cryptic, but just extracts um/pix (calibration) of X and Y into res
        res = [float(i[0].text) for i in meta.findall('.//Scaling/Items/*') if
               i.attrib['Id'] == 'X' or i.attrib['Id'] == 'Y']
        assert res[0] == res[1], "pixels are not square"

        # get first calibration value and convert it from meters to um
        res = res[0] * 1e6

        ts_ix = [k for k, a1 in enumerate(czi.attachment_directory) if a1.filename[:10] == 'TimeStamps'][0]
        timestamps = list(czi.attachments())[ts_ix].data()
        dt = np.median(np.diff(timestamps))

        ax_dct = {n: k for k, n in enumerate(czi.axes)}
        n_frames = czi.shape[ax_dct['T']]
        n_channels = czi.shape[ax_dct['C']]
        n_zstacks = czi.shape[ax_dct['Z']]
        n_X = czi.shape[ax_dct['X']]
        n_Y = czi.shape[ax_dct['Y']]

        images = list()
        for sb in czi.subblock_directory:
            images.append(sb.data_segment().data().reshape((n_X, n_Y)))

        return np.array(images), 1 / res, dt, n_frames, n_channels  # , n_zstacks


def find_image(img_name, folder=None):
    if folder is None:
        folder = os.path.dirname(img_name)
        img_name = os.path.basename(img_name)

    for root, directories, filenames in os.walk(folder):
        for file in filenames:
            joinf = os.path.abspath(os.path.join(root, file))
            if os.path.isfile(joinf) and joinf[-4:] == '.tif' and file == img_name:
                return load_tiff(joinf)
            if os.path.isfile(joinf) and joinf[-4:] == '.czi' and file == img_name:
                return load_zeiss(joinf)


def retrieve_image(image_arr, frame=0, zstack=0, channel=0, number_of_channels=1, number_of_zstacks=1):
    ix = frame * (number_of_channels * number_of_zstacks) + zstack * number_of_channels + channel
    logger.debug("retrieving frame %d of channel %d at s-stack=%d (index=%d)" % (frame, channel, zstack, ix))
    return image_arr[ix]


def image_iterator(image_arr, channel=0, number_of_frames=1):
    nimgs = image_arr.shape[0]
    n_channels = int(nimgs / number_of_frames)
    for f in range(number_of_frames):
        ix = f * n_channels + channel
        logger.debug("retrieving frame %d of channel %d (index=%d)" % (f, channel, ix))
        if ix < nimgs: yield image_arr[ix]
