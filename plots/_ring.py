import os
import ast

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import EngFormatter
import logging
import numpy as np
import pandas as pd
import seaborn as sns
import shapely.wkt
import scipy.signal

from plots.utils import histogram_of_every_row, histogram_with_errorbars
import operetta as o
import filters

logger = logging.getLogger('ring')
logger.setLevel(logging.DEBUG)


def eval_into_array(df, column_name):
    if column_name in df:
        df.loc[:, column_name] = df[column_name].apply(lambda r: np.array(ast.literal_eval(r)) if type(r) == str else r)
    return df


def hist_area(edges, counts):
    bin_width = np.diff(edges)
    return np.sum([bin_width[i] * counts[i] for i in range(len(counts))])


class Ring():
    _columns = ['id', 'row', 'col', 'fid', 'p',
                'nuc_int', 'nuc_dens',
                'act_int', 'act_dens',
                'act_rng_int', 'act_rng_dens',
                'hist_edges', 'act_nuc_hist', 'act_rng_hist']
    _cat = ['zid', 'row', 'col', 'fid', 'p', 'Cell Type', 'Cell Count', 'Compound', 'Concentration']

    # def __init__(self, cc: o.ConfiguredChannels):
    def __init__(self, cc):
        self.cc = cc

        self.all = (pd.read_csv(os.path.join(cc.base_path, "out", "nuclei.pandas.csv"))
                    .merge(cc.layout, on=["row", "col"])
                    .drop(columns=["Unnamed: 0"])
                    )

        self._df = None
        self._lines = None
        self._lines_filt = None
        self._dmax = None
        self.formatter = EngFormatter(unit='')

    @property
    def df(self):
        if self._df is not None: return self._df

        df = (self.all
              .pipe(eval_into_array, column_name="hist_edges")
              .pipe(eval_into_array, column_name="act_nuc_hist")
              .pipe(eval_into_array, column_name="act_rng_hist")
              .pipe(filters.histogram, edges="hist_edges", values="act_rng_hist", agg_fn="max",
                    edge_min=0, edge_max=600, value_min=0, value_max=20)
              .pipe(filters.histogram, edges="hist_edges", values="act_nuc_hist", agg_fn="max",
                    edge_min=500, edge_max=np.inf, value_min=0, value_max=150)
              )
        if np.all([c in df.columns for c in ['act_rng_int', 'act_int', 'act_rng_dens', 'act_dens']]):
            df.loc[:, 'ring_int_ratio'] = df['act_rng_int'] / df['act_int']
            df.loc[:, 'ring_dens_ratio'] = df['act_rng_dens'] / df['act_dens']

        self._df = df

        return self._df

    @property
    def lines(self):
        if self._lines is not None: return self._lines

        a = self.df
        columns = [c for c in a.columns if c[:9] == "act_line_"]
        for c in columns:
            a = a.pipe(eval_into_array, column_name=c)
        a = pd.melt(a, id_vars=self._cat, value_vars=columns, value_name='signal')
        a.loc[:, "unit"] = a.apply(lambda r: "%s|%s|%s|%s|%s" % (r['zid'], r['row'], r['col'], r['fid'], r['p']), axis=1)
        # calculate the sum of signal intensities
        a.loc[:, "sum"] = a["signal"].apply(np.sum)
        a.loc[:, "s_max"] = a["signal"].apply(np.max)
        # calculate domains of signals (x's) and center all the curves on the maximum points
        a.loc[:, "xpeak"] = a["signal"].apply(lambda v: np.argmax(v))

        idx_nparr = a["signal"].apply(lambda v: type(v) == np.ndarray)
        a.loc[:, "x"] = a.loc[idx_nparr, "signal"].apply(lambda v: np.arange(start=0, stop=len(v), step=1))
        a.loc[:, "x_center"] = a["x"] - a["xpeak"]

        # calculate width of peaks
        def _w(r):
            try:
                width = scipy.signal.peak_widths(r["signal"], [r["xpeak"]])[0][0]
            except ValueError as ve:
                width = np.nan
            return width

        a.loc[:, "v_width"] = a.loc[~a["signal"].isna()].apply(_w, axis=1)

        self._lines = a

        return self._lines

    @property
    def lines_filtered(self):
        if self._lines_filt is not None: return self._lines_filt
        lflt_path = os.path.join(self.cc.base_path, 'out', 'lines.filtered.csv')
        if not os.path.exists(lflt_path):
            self._lines_filt = (self.lines.pipe(filters.lines)
                                .dropna(axis=0, subset=["zid"])
                                )
            txt_safe = self._lines_filt.copy()
            arr_ops = {"precision": 2, "separator": ",", "suppress_small": True,
                       "formatter": {"float": lambda x: "%0.2f" % x}}
            idx = txt_safe["signal"].apply(lambda v: type(v) == np.ndarray)
            for col in ["x", "x_center", "signal"]:
                txt_safe.loc[idx, col] = txt_safe.loc[idx, col].apply(lambda v: np.array2string(v, **arr_ops))
            txt_safe.to_csv(lflt_path, index=False)
        else:
            logger.info("Loading filtered lines from file.")
            self._lines_filt = (pd.read_csv(lflt_path)
                                .pipe(eval_into_array, column_name="x")
                                .pipe(eval_into_array, column_name="x_center")
                                .pipe(eval_into_array, column_name="signal")
                                )
        return self._lines_filt

    @property
    def dmax(self):
        if self._dmax is not None: return self._dmax

        idx_max = self.df.groupby(['row', 'col', 'fid'])['ring_dens_ratio'].transform(max) == self.df['ring_dens_ratio']
        self._dmax = self.df[idx_max]

        return self._dmax

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
        g = (g.map_dataframe(_distplot, "ring_dens_ratio", bins=bins)
             .set(xscale='log')
             .set(xticks=ticks, xticklabels=labels, xlim=(min(ticks), max(ticks)))
             .add_legend()
             )
        path = o.ensure_dir(os.path.join(self.cc.base_path, 'out', 'graphs', 'actinring_dens_hist_all.pdf'))
        g.savefig(path)
        plt.close()

        g = sns.FacetGrid(self.dmax, row="Compound", hue="Compound", legend_out=True)
        g = (g.map_dataframe(_distplot, "ring_dens_ratio", bins=bins)
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
             .add_legend()
             )
        for _ax in g.axes[0]:
            _ax.xaxis.set_major_formatter(self.formatter)
            _ax.yaxis.set_major_formatter(self.formatter)
        path = o.ensure_dir(os.path.join(self.cc.base_path, 'out', 'graphs', 'dna_vs_actin_kde_dens.pdf'))
        g.savefig(path)
        plt.close()

        g = sns.FacetGrid(self.dmax, hue="Compound", legend_out=True)
        g = (g.map(sns.kdeplot, "nuc_int", "act_rng_int", shade=True, shade_lowest=False)
             .add_legend()
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
                     .add_legend()
                     )
                path = o.ensure_dir(
                    os.path.join(self.cc.base_path, 'out', 'graphs', 'histogram_actin_%s_%s.png' % (place, flt)))
                g.savefig(path, dpi=150)
                plt.close()

    def histogram_areas(self):
        a = self.all
        a.loc[:, 'rng_hist_area'] = a.apply(lambda r: hist_area(r['hist_edges'], r['act_rng_hist']), axis=1)
        a.loc[:, 'nuc_hist_area'] = a.apply(lambda r: hist_area(r['hist_edges'], r['act_nuc_hist']), axis=1)
        a.loc[:, 'hist_area_ratio'] = a['rng_hist_area'] / a['nuc_hist_area']
        self.all = a

        fig = plt.figure(figsize=(8, 8), dpi=150)
        ax = fig.gca()
        sns.scatterplot(x="nuc_hist_area", y="rng_hist_area", data=a, hue="Compound", alpha=0.01, rasterized=True)
        plt.xscale('log')
        plt.yscale('log')
        ax.xaxis.set_major_formatter(self.formatter)
        ax.yaxis.set_major_formatter(self.formatter)
        path = o.ensure_dir(os.path.join(self.cc.base_path, 'out', 'graphs', 'histareas_scatter.pdf'))
        fig.savefig(path)
        plt.close()

        logbins = np.logspace(0, np.log10(1e6), 1000)
        g = sns.FacetGrid(a, hue="Compound")
        g = (g.map_dataframe(_distplot, "nuc_hist_area", bins=logbins)
             .set(xscale='log')
             # .set(xticks=ticks, xticklabels=labels, xlim=(min(ticks), max(ticks)))
             .add_legend()
             )
        path = o.ensure_dir(os.path.join(self.cc.base_path, 'out', 'graphs', 'histareas_nuc_dist.pdf'))
        g.savefig(path)
        plt.close()

        g = sns.FacetGrid(a, hue="Compound")
        g = (g.map_dataframe(_distplot, "rng_hist_area", bins=logbins)
             .set(xscale='log')
             # .set(xticks=ticks, xticklabels=labels, xlim=(min(ticks), max(ticks)))
             .add_legend()
             )
        path = o.ensure_dir(os.path.join(self.cc.base_path, 'out', 'graphs', 'histareas_rng_dist.pdf'))
        g.savefig(path)
        plt.close()

        logbins = np.logspace(0, np.log10(a["hist_area_ratio"].max()), 1000)
        g = sns.FacetGrid(a, hue="Compound")
        g = (g.map_dataframe(_distplot, "hist_area_ratio", bins=logbins)
             .set(xscale='log')
             # .set(xticks=ticks, xticklabels=labels, xlim=(min(ticks), max(ticks)))
             .add_legend()
             )
        path = o.ensure_dir(os.path.join(self.cc.base_path, 'out', 'graphs', 'histareas_ratio_dist.pdf'))
        g.savefig(path)
        plt.close()

    def line_integrals(self):
        for a, kind in zip([self.lines, self.lines_filtered], ["all", "flt"]):
        # for a, kind in zip([self.lines_filtered], ["flt"]):
            # optional: filter a subgroup
            # col_order = ["Arrest", "Cycling"]
            # col_order = ['Cycling', 'Arrest', 'Release',
            #              'Cyto2ug-Cyc', 'Cyto2ug-Arr', 'Cyto2ug-Rel',
            #              'Noc20ng-Cyc', 'Noc20ng-Arr', 'Noc20ng-Rel']
            col_order = a["Compound"].unique()
            print(col_order)
            a = a[a["Compound"].isin(col_order)]
            # get only one row per z-stack
            idx = a.groupby(["unit"])["s_max"].transform(max) == a["s_max"]
            a = a.loc[idx]

            fig = plt.figure(figsize=(8, 8), dpi=150)
            ax = fig.gca()
            sns.boxenplot(x="Compound", y="sum", order=col_order, data=a)
            ax.yaxis.set_major_formatter(self.formatter)
            ax.set_yscale('log')
            ax.set_xticklabels(ax.xaxis.get_ticklabels(), rotation=45, multialignment='right')
            path = o.ensure_dir(os.path.join(self.cc.base_path, 'out', 'graphs', 'line_boxplot_%s.pdf' % kind))
            fig.savefig(path)
            plt.close()

            fig = plt.figure(figsize=(8, 8), dpi=150)
            ax = fig.gca()
            sns.scatterplot(x="v_width", y="sum", data=a, hue="Compound", alpha=0.1, rasterized=True)
            # plt.xscale('log')
            # plt.yscale('log')
            ax.set_xlim((0, 16))
            ax.set_ylim((0, 350e3))
            ax.xaxis.set_major_formatter(self.formatter)
            ax.yaxis.set_major_formatter(self.formatter)
            path = o.ensure_dir(os.path.join(self.cc.base_path, 'out', 'graphs', 'lines_scatter_%s.pdf' % kind))
            fig.savefig(path)
            plt.close()

    def line_measurements(self):
        a = self.lines_filtered
        # optional: filter a subgroup
        # a = a[a["Compound"].isin(["Arrest", "Cycling"])]
        # get only one row per z-stack
        idx = a.groupby(["unit"])["s_max"].transform(max) == a["s_max"]
        a = a[idx]

        # remove constant component of signal vector
        a.loc[:, "crit"] = a['signal'].apply(lambda v: v.max() - v.min())
        a.loc[:, "v_mean"] = a['signal'].apply(lambda v: v.mean())
        a.loc[:, "signal_n"] = a["signal"] - a["v_mean"]
        a.drop(columns=["v_mean", "crit", "signal", "x", "sum", "v_width", "xpeak"], inplace=True)

        # transform signal and domain vectors into long format (see https://stackoverflow.com/questions/27263805
        val_col = 'signal_n'
        ix_col = 'x_center'
        b = pd.DataFrame({
            col: pd.Series(data=np.repeat(a[col].values, a[val_col].str.len()))
            for col in a.columns.drop([val_col, ix_col])}
        ).assign(**{ix_col: np.concatenate(a[ix_col].values), val_col: np.concatenate(a[val_col].values)})[a.columns]

        # build a new index
        b = b.set_index(['unit', 'variable'])
        b.index = [b.index.map('{0[0]}|{0[1]}'.format)]
        b = b.reset_index().rename(columns={"level_0": "unit"})

        # plots
        x_var = 'x_center'
        y_var = 'signal_n'

        g = sns.FacetGrid(b, hue="Compound", row='Compound', height=2, aspect=2)
        g = (g.map_dataframe(sns.lineplot, x=x_var, y=y_var, units='unit', estimator=None, alpha=1, lw=0.1)
             # .set(yscale='log')
             .set(xlim=(-20, 20))
             )
        for _ax in g.axes[0]:
            _ax.xaxis.set_major_formatter(self.formatter)
            _ax.yaxis.set_major_formatter(self.formatter)
        path = o.ensure_dir(os.path.join(self.cc.base_path, 'out', 'graphs', 'lines_flt_indiv.pdf'))
        g.savefig(path)
        plt.close()

        fig = plt.figure(figsize=(6, 4), dpi=150)
        ax = fig.gca()
        sns.lineplot(x=x_var, y=y_var, data=b, hue='Compound', ax=ax)
        ax.xaxis.set_major_formatter(self.formatter)
        ax.yaxis.set_major_formatter(self.formatter)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(5))
        ax.set_xlim([-20, 20])
        # plt.yscale('log')
        path = o.ensure_dir(os.path.join(self.cc.base_path, 'out', 'graphs', 'lines_trend.pdf'))
        fig.savefig(path)
        plt.close()


# Histogram of actin ring ratio
def _distplot(col_lbl, **kwargs):
    ax = plt.gca()
    data = kwargs.pop("data")
    color = kwargs.pop("color")
    _bins = kwargs.pop("bins")
    _lbl = kwargs.pop("label")
    sns.distplot(data[col_lbl], bins=_bins, hist=False, color=color, label=_lbl, axlabel=False, ax=ax)
