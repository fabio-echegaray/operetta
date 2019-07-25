import os
import logging
import traceback
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import EngFormatter
import pandas as pd
import numpy as np
import seaborn as sns
import plots as p
import operetta as o
import shapely.wkt
import filters

pd.set_option('display.width', 320)
pd.set_option('display.max_columns', 20)

logger = logging.getLogger('ring')
logger.setLevel(logging.DEBUG)
logging.getLogger('matplotlib').setLevel(logging.ERROR)


def move_images(df):
    render_path = o.ensure_dir(os.path.join(folder, 'out', 'render'))

    for ix, r in df.iterrows():
        destination_folder = o.ensure_dir(os.path.join(folder, 'out', 'selected-images',
                                                       '%s@%s' % (r["Cell Type"], r["Cell Count"]), r["Compound"]))

        # name, original_path = o.ConfiguredChannels.filename_of_render(r, render_path)
        name = 'r%d-c%d-f%d-i%d.jpg' % (r["row"], r["col"], r["fid"], r["id"])
        original_path = os.path.join(render_path, name)
        destination_path = o.ensure_dir(os.path.join(destination_folder, name))

        try:
            logger.info('moving %s to %s' % (name, destination_folder))
            os.symlink(original_path, destination_path, False)
        except Exception as e:
            logger.warning('no render for %s' % original_path)
            logger.warning(e)
            # traceback.print_stack()


folder = "/Volumes/Kidbeat/data/20190715-u2os-actingring"
operetta = o.ConfiguredChannels(folder)

print(operetta.files.head(10))
print(operetta.layout.head(10))

# layout_path = os.path.join(folder, 'out', 'layout.csv')
# operetta.layout.to_csv(layout_path, index=False)

um_per_pix = operetta.um_per_pix

df = (operetta.samples
      # .pipe(lambda _df: _df[(_df['dna_int'] > 1e6) & (_df['dna_int'] < 2e6) &
      #                       (_df['act_rng_int'] > 0.8e6) & (_df['act_rng_int'] < 1.25e6)])
      .pipe(filters.nucleus, radius_min=4, radius_max=10)
      .merge(operetta.layout, on=["row", "col"])
      )
move_images(df)
print(df.head(10))
# exit(0)

fig = plt.figure(figsize=(8, 8))
ax = fig.gca()

# Nucleus area distribution
fig.set_size_inches(16, 4)
df.loc[:, "nuc_area"] = df.apply(lambda row: shapely.wkt.loads(row['nucleus']).area, axis=1)
df["nuc_area"].plot.hist(ax=ax, bins=1000)
plt.xlim([-1, 1e3])
plt.ylim([0, 300])
ax.xaxis.set_major_locator(ticker.MultipleLocator(100))
ax.xaxis.set_minor_locator(ticker.MultipleLocator(10))

path = o.ensure_dir(os.path.join(folder, 'out', 'graphs', 'filtered_nuclei_area_histogram.pdf'))
fig.savefig(path)

# Histogram of hoechst intensity
# logbins = np.logspace(0, np.log10(np.iinfo(np.uint16).max), 1000)
# sns.distplot(df["dna_int"], bins=logbins, ax=ax)
# plt.xscale('log')
# plt.xlim([1e3, 1e5])

# DNA intensity vs Actin intensity
ax.cla()
fig.set_size_inches(8, 8)
ax.scatter(df["dna_dens"], df["act_rng_dens"], alpha=0.01)
path = o.ensure_dir(os.path.join(folder, 'out', 'graphs', 'dna_vs_actin_scatter.pdf'))
fig.savefig(path)

ax.cla()
fig.set_size_inches(8, 8)
sns.scatterplot(x="dna_int", y="act_rng_int", data=df, hue="Compound", alpha=0.01)
formatter = EngFormatter(unit='')
ax.xaxis.set_major_formatter(formatter)
ax.yaxis.set_major_formatter(formatter)
path = o.ensure_dir(os.path.join(folder, 'out', 'graphs', 'dna_vs_actin_scatter_cond.pdf'))
fig.savefig(path)
