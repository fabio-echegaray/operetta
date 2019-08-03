import os
import logging
from shutil import copyfile

import pandas as pd
import enlighten

import operetta as o
from plots import Ring

pd.set_option('display.width', 320)
pd.set_option('display.max_columns', 20)

logger = logging.getLogger('ring')
logger.setLevel(logging.DEBUG)
logging.getLogger('matplotlib').setLevel(logging.ERROR)


def move_images(df):
    render_path = o.ensure_dir(os.path.join(folder, 'out', 'render'))
    manager = enlighten.get_manager()
    bar = manager.counter(total=len(df), desc='Progress', unit='files')

    for ix, r in df.iterrows():
        destination_folder = o.ensure_dir(os.path.join(folder, 'out', 'selected-images',
                                                       '%s@%s' % (r["Cell Type"], r["Cell Count"]), r["Compound"]))

        # name, original_path = o.ConfiguredChannels.filename_of_render(r, render_path)
        name = 'r%d-c%d-f%d-p%i-i%d.jpg' % (r["row"], r["col"], r["fid"], r["p"], r["id"])
        original_path = os.path.join(render_path, name)
        destination_path = o.ensure_dir(os.path.join(destination_folder, name))

        try:
            logger.info('moving %s to %s' % (name, destination_folder))
            # os.symlink(original_path, destination_path, False)
            copyfile(original_path, destination_path)
            bar.update()
        except Exception as e:
            logger.warning('no render for %s' % original_path)
            logger.warning(e)
            # traceback.print_stack()
    manager.stop()


# folder = "/Volumes/Kidbeat/data/20190715-u2os-actingring"
folder = "/Volumes/Kidbeat/data/20190729-u2os-noc-cyto-40X"
# folder = "../data/20190729-u2os-noc-cyto-40X/"
operetta = o.ConfiguredChannels(folder)
pl = Ring(operetta)

dmax = pl.dmax
idx = (dmax['row'] == 2) & (dmax['col'] == 2) & (dmax['fid'].isin([86, 94]))
print(dmax[idx])

# move_images(dmax)
# exit(0)


# pl.nuclei_filtered()
pl.dna_intensity()
pl.actin_intensity_and_density_histogram()
pl.actin_intensity_vs_density()
pl.dna_vs_actin_intesity()
pl.actin_ring()
