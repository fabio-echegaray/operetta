import os

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import EngFormatter
import numpy as np
import pandas as pd
import seaborn as sns
import shapely.wkt

from plots.utils import histogram_of_every_row
import operetta as o


class Ring():
    # def __init__(self, cc: o.ConfiguredChannels):
    def __init__(self, cc):
        self.cc = cc

        df = (pd.read_csv(os.path.join(cc.base_path, "out", "nuclei.pandas.csv"), index_col=False)
              .merge(cc.layout, on=["row", "col"])
              )
        df['ring_int_ratio'] = df['act_rng_int'] / df['act_int']
        df['ring_dens_ratio'] = df['act_rng_dens'] / df['act_dens']
        idx_max = df.groupby(['row', 'col', 'fid'])['ring_dens_ratio'].transform(max) == df['ring_dens_ratio']
        dmax = df[idx_max]
        self._df = df
        self.dmax = dmax
        self.formatter = EngFormatter(unit='')

    def nuclei_filtered(self):
        fig = plt.figure(figsize=(16, 4), dpi=150)
        ax = fig.gca()
        self._df.loc[:, "nuc_area"] = self._df.apply(lambda row: shapely.wkt.loads(row['nucleus']).area, axis=1)
        self._df["nuc_area"].plot.hist(ax=ax, bins=1000)
        plt.xlim([-1, 1e3])
        plt.ylim([0, 300])
        ax.xaxis.set_major_locator(ticker.MultipleLocator(100))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(10))
        path = o.ensure_dir(os.path.join(self.cc.base_path, 'out', 'graphs', 'filtered_nuclei_area_histogram.pdf'))
        fig.savefig(path)
        plt.close()

    def dna_intensity(self):
        # Histogram of hoechst intensity
        fig = plt.figure(figsize=(8, 8), dpi=150)
        ax = fig.gca()
        logbins = np.logspace(0, np.log10(np.iinfo(np.uint16).max), 1000)
        sns.distplot(self._df["dna_int"], bins=logbins, ax=ax)
        plt.xscale('log')
        plt.xlim([1e3, 1e5])
        plt.close()

    # Actin intensity vs Actin density
    def actin_intensity_vs_density(self):
        fig = plt.figure(figsize=(8, 8), dpi=150)
        ax = fig.gca()
        sns.scatterplot(x="ring_int_ratio", y="ring_dens_ratio", data=self._df, hue="Compound", alpha=0.01)
        plt.xscale('log')
        plt.yscale('log')
        ax.xaxis.set_major_formatter(self.formatter)
        ax.yaxis.set_major_formatter(self.formatter)
        path = o.ensure_dir(os.path.join(self.cc.base_path, 'out', 'graphs', 'actin_int_vs_dens_scatter.png'))
        fig.savefig(path)
        plt.close()

    def actin_intensity_and_density_histogram(self):
        bins = np.logspace(2 ** -1, np.log2(self.dmax["ring_dens_ratio"].max()), 1000)
        logform = ticker.LogFormatterMathtext(base=2)
        ticks = ticker.LogLocator(base=2, ).tick_values(2 ** -1, np.log2(self.dmax["ring_dens_ratio"].max()))
        labels = [logform(i) for i in ticks]

        g = sns.FacetGrid(self.dmax, hue="Compound", legend_out=True)
        g = (g.map_dataframe(_histogram_act_rng_ratio, "ring_dens_ratio", bins=bins)
             .set(xscale='log')
             .set(xticks=ticks, xticklabels=labels, xlim=(min(ticks), max(ticks)))
             .add_legend()
             )
        path = o.ensure_dir(os.path.join(self.cc.base_path, 'out', 'graphs', 'actinring_dens_hist_all.pdf'))
        g.savefig(path)
        plt.close()

        g = sns.FacetGrid(self.dmax, row="Compound", hue="Compound", legend_out=True)
        g = (g.map_dataframe(_histogram_act_rng_ratio, "ring_dens_ratio", bins=bins)
             .set(xscale='log')
             .set(xticks=ticks, xticklabels=labels, xlim=(min(ticks), max(ticks)))
             .add_legend()
             )
        path = o.ensure_dir(os.path.join(self.cc.base_path, 'out', 'graphs', 'actinring_dens_hist_each.pdf'))
        g.savefig(path)
        plt.close()

    # DNA intensity vs Actin intensity
    def dna_vs_actin_intesity(self):
        g = sns.FacetGrid(self.dmax, hue="Compound", legend_out=True)
        g = (g.map(sns.kdeplot, "dna_dens", "act_rng_dens", shade=True, shade_lowest=False)
             )
        for _ax in g.axes[0]:
            _ax.xaxis.set_major_formatter(self.formatter)
            _ax.yaxis.set_major_formatter(self.formatter)
        path = o.ensure_dir(os.path.join(self.cc.base_path, 'out', 'graphs', 'dna_vs_actin_kde_dens.pdf'))
        g.savefig(path)
        plt.close()

        g = sns.FacetGrid(self.dmax, hue="Compound", legend_out=True)
        g = (g.map(sns.kdeplot, "dna_int", "act_rng_int", shade=True, shade_lowest=False)
             )
        for _ax in g.axes[0]:
            _ax.xaxis.set_major_formatter(self.formatter)
            _ax.yaxis.set_major_formatter(self.formatter)
        path = o.ensure_dir(os.path.join(self.cc.base_path, 'out', 'graphs', 'dna_vs_actin_kde_int.pdf'))
        g.savefig(path)
        plt.close()

        # ax.cla()
        # fig.set_size_inches(8, 8)
        # sns.scatterplot(x="dna_int", y="act_rng_int", data=df, hue="Compound", alpha=0.01)
        # plt.xscale('log')
        # plt.yscale('log')
        # ax.xaxis.set_major_formatter(formatter)
        # ax.yaxis.set_major_formatter(formatter)
        # path = o.ensure_dir(os.path.join(folder, 'out', 'graphs', 'dna_vs_actin_scatter_int.png'))
        # fig.savefig(path)

        # ax.cla()
        # fig.set_size_inches(8, 8)
        # ax.scatter(df["dna_dens"], df["act_rng_dens"], alpha=0.01)
        # path = o.ensure_dir(os.path.join(folder, 'out', 'graphs', 'dna_vs_actin_scatter.png'))
        # fig.savefig(path)

    def actin_ring(self):
        fig = plt.figure(figsize=(8, 8), dpi=150)
        ax = fig.gca()
        sns.boxenplot(x="Compound", y="ring_int_ratio", data=self.dmax)
        ax.yaxis.set_major_formatter(self.formatter)
        path = o.ensure_dir(os.path.join(self.cc.base_path, 'out', 'graphs', 'actinring_boxplot_cond_int.pdf'))
        fig.savefig(path)
        plt.close()

        fig = plt.figure(figsize=(8, 8), dpi=150)
        ax = fig.gca()
        fig.set_size_inches(8, 8)
        sns.boxenplot(x="Compound", y="ring_dens_ratio", data=self.dmax)
        ax.yaxis.set_major_formatter(self.formatter)
        path = o.ensure_dir(os.path.join(self.cc.base_path, 'out', 'graphs', 'actinring_boxplot_cond_dens.pdf'))
        fig.savefig(path)
        plt.close()

    def actin_ring_histograms(self):
        g = sns.FacetGrid(self.dmax, row="Compound", height=2, aspect=2, legend_out=True)
        g = (g.map_dataframe(histogram_of_every_row, "act_rng_hist")
             .set(xscale='log')
             .set(xlim=(100, 1e4))
             )
        path = o.ensure_dir(os.path.join(self.cc.base_path, 'out', 'graphs', 'actin_rng_hist.png'))
        g.savefig(path, dpi=150)
        plt.close()


# Histogram of actin ring ratio
def _histogram_act_rng_ratio(col_lbl, **kwargs):
    ax = plt.gca()
    data = kwargs.pop("data")
    color = kwargs.pop("color")
    _bins = kwargs.pop("bins")
    sns.distplot(data[col_lbl], bins=_bins, hist=False, color=color, ax=ax)
