import os
import ast

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import EngFormatter
import numpy as np
import pandas as pd
import seaborn as sns
import shapely.wkt

from plots.utils import histogram_of_every_row, histogram_with_errorbars
import operetta as o
import filters


def eval_into_array(df, column_name):
    df.loc[:, column_name] = df[column_name].apply(lambda r: np.array(ast.literal_eval(r)))
    return df


class Ring():
    _columns = ['row', 'col', 'fid', 'p',
                'nuc_int', 'nuc_dens',
                'act_int', 'act_dens',
                'act_rng_int', 'act_rng_dens',
                'hist_edges', 'act_nuc_hist', 'act_rng_hist']

    # def __init__(self, cc: o.ConfiguredChannels):
    def __init__(self, cc):
        self.cc = cc

        self.all = (pd.read_csv(os.path.join(cc.base_path, "out", "nuclei.pandas.csv"), usecols=self._columns)
                    .merge(cc.layout, on=["row", "col"])
                    )
        df = (self.all
              .pipe(eval_into_array, column_name="hist_edges")
              .pipe(eval_into_array, column_name="act_nuc_hist")
              .pipe(eval_into_array, column_name="act_rng_hist")
              .pipe(filters.histogram, edges="hist_edges", values="act_rng_hist", agg_fn="max",
                    edge_min=0, edge_max=600, value_min=0, value_max=20)
              .pipe(filters.histogram, edges="hist_edges", values="act_nuc_hist", agg_fn="max",
                    edge_min=500, edge_max=np.inf, value_min=0, value_max=150)
              )
        df.loc[:, 'ring_int_ratio'] = df['act_rng_int'] / df['act_int']
        df.loc[:, 'ring_dens_ratio'] = df['act_rng_dens'] / df['act_dens']
        idx_max = df.groupby(['row', 'col', 'fid'])['ring_dens_ratio'].transform(max) == df['ring_dens_ratio']
        self.df = df
        self.dmax = df[idx_max]
        self.formatter = EngFormatter(unit='')

    def nuclei_filtered(self):
        fig = plt.figure(figsize=(16, 4), dpi=150)
        ax = fig.gca()
        self.df.loc[:, "nuc_area"] = self.df.apply(lambda row: shapely.wkt.loads(row['nucleus']).area, axis=1)
        self.df["nuc_area"].plot.hist(ax=ax, bins=1000)
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
        sns.distplot(self.df["nuc_int"], bins=logbins, ax=ax)
        plt.xscale('log')
        plt.xlim([1e3, 1e5])
        plt.close()

    # Actin intensity vs Actin density
    def actin_intensity_vs_density(self):
        fig = plt.figure(figsize=(8, 8), dpi=150)
        ax = fig.gca()
        sns.scatterplot(x="ring_int_ratio", y="ring_dens_ratio", data=self.df, hue="Compound", alpha=0.01)
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
        g = (g.map(sns.kdeplot, "nuc_dens", "act_rng_dens", shade=True, shade_lowest=False)
             )
        for _ax in g.axes[0]:
            _ax.xaxis.set_major_formatter(self.formatter)
            _ax.yaxis.set_major_formatter(self.formatter)
        path = o.ensure_dir(os.path.join(self.cc.base_path, 'out', 'graphs', 'dna_vs_actin_kde_dens.pdf'))
        g.savefig(path)
        plt.close()

        g = sns.FacetGrid(self.dmax, hue="Compound", legend_out=True)
        g = (g.map(sns.kdeplot, "nuc_int", "act_rng_int", shade=True, shade_lowest=False)
             )
        for _ax in g.axes[0]:
            _ax.xaxis.set_major_formatter(self.formatter)
            _ax.yaxis.set_major_formatter(self.formatter)
        path = o.ensure_dir(os.path.join(self.cc.base_path, 'out', 'graphs', 'dna_vs_actin_kde_int.pdf'))
        g.savefig(path)
        plt.close()

        # ax.cla()
        # fig.set_size_inches(8, 8)
        # sns.scatterplot(x="nuc_int", y="act_rng_int", data=df, hue="Compound", alpha=0.01)
        # plt.xscale('log')
        # plt.yscale('log')
        # ax.xaxis.set_major_formatter(formatter)
        # ax.yaxis.set_major_formatter(formatter)
        # path = o.ensure_dir(os.path.join(folder, 'out', 'graphs', 'dna_vs_actin_scatter_int.png'))
        # fig.savefig(path)

        # ax.cla()
        # fig.set_size_inches(8, 8)
        # ax.scatter(df["nuc_dens"], df["act_rng_dens"], alpha=0.01)
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
        sns.boxenplot(x="Compound", y="ring_dens_ratio", data=self.dmax)
        ax.yaxis.set_major_formatter(self.formatter)
        path = o.ensure_dir(os.path.join(self.cc.base_path, 'out', 'graphs', 'actinring_boxplot_cond_dens.pdf'))
        fig.savefig(path)
        plt.close()

    def actin_ring_histograms(self):
        for place in ["nuc", "rng"]:
            for _df, flt in [(self.all, "all"), (self.df, "flt")]:
                fig = plt.figure(figsize=(8, 4), dpi=150)
                ax = fig.gca()
                histogram_with_errorbars(_df, "hist_edges", "act_%s_hist" % place)
                ax.set_xscale('log')
                ax.xaxis.set_major_formatter(self.formatter)
                ax.yaxis.set_major_formatter(self.formatter)
                ax.set_xlim([100, 1e4])
                path = o.ensure_dir(
                    os.path.join(self.cc.base_path, 'out', 'graphs', 'histogram_actin_%s_%s.pdf' % (place, flt)))
                fig.savefig(path)
                plt.close()

                # if flt == "all": continue
                g = sns.FacetGrid(_df, row="Compound", height=3, aspect=2, legend_out=True)
                g = (g.map_dataframe(histogram_of_every_row, "act_%s_hist" % place)
                     .set(xscale='log')
                     .set(xlim=(100, 1e4))
                     )
                path = o.ensure_dir(
                    os.path.join(self.cc.base_path, 'out', 'graphs', 'histogram_actin_%s_%s.png' % (place, flt)))
                g.savefig(path, dpi=150)
                plt.close()


# Histogram of actin ring ratio
def _histogram_act_rng_ratio(col_lbl, **kwargs):
    ax = plt.gca()
    data = kwargs.pop("data")
    color = kwargs.pop("color")
    _bins = kwargs.pop("bins")
    sns.distplot(data[col_lbl], bins=_bins, hist=False, color=color, ax=ax)
