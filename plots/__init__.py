import matplotlib
from .utils import colors
from ._ring import Ring, eval_into_array

# Type 2/TrueType fonts.
matplotlib.rcParams.update({'pdf.fonttype': 42})
matplotlib.rcParams.update({'ps.fonttype': 42})

matplotlib.rcParams.update({'font.family': 'sans-serif'})
matplotlib.rcParams.update({'font.sans-serif': ['Arial']})

matplotlib.rcParams.update({'axes.titlesize': 8})
matplotlib.rcParams.update({'axes.labelsize': 7})
matplotlib.rcParams.update({'xtick.labelsize': 7})
matplotlib.rcParams.update({'ytick.labelsize': 7})
matplotlib.rcParams.update({'legend.fontsize': 7})
