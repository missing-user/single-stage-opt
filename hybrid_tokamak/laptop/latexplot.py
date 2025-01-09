interactive = False
# interactive = True
import matplotlib
if not interactive:
  matplotlib.use("pgf")
#   matplotlib.use("pdf")
import matplotlib.pyplot as plt

fontsize =  10.95# pt
# plt.style.use("Solarize_Light2")
plt.rcParams.update({
    "font.family": "sans-serif",  # use serif/main font for text elements
    "text.usetex": True,     # use inline math for ticks
    "pgf.rcfonts": False,    # don't setup fonts from rc parameters
    "pgf.texsystem": "pdflatex",

    # Use 10pt font in plots, to match 10pt font in document
    "axes.labelsize": fontsize,
    "font.size": fontsize,
    "axes.titlesize": fontsize, # large by default
    # "axes.titleweight": "bold", # normal by default

    # Make the legend/label fonts a little smaller
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,

    'axes.facecolor' : 'FFFFFF'
})
def set_cmap(cycles:int|list=4):
    if isinstance(cycles, list) and len(cycles) > 0:
        assert sum(cycles) > 0
        scaling = 1/(1-0.5/len(cycles))
        midevalpoints = [i/len(cycles) * scaling for i in range(len(cycles))] 
        evalpoints = []
        for i, cat in zip(midevalpoints, cycles):
            ministep = 0.5/len(cycles)/cat
            for j in range(cat):
                evalpoints.append(i + ministep*j)
    else:
      evalpoints = [i/cycles for i in range(cycles)]
      if cycles == 2:
          evalpoints = [0, 0.7]

    plt.rcParams['axes.prop_cycle'] = matplotlib.cycler(color=[plt.cm.plasma(e) for e in evalpoints])
    plt.rcParams['image.cmap'] = 'plasma'
# set_cmap()


# 483.6969pt. text width
# Shamelessly stolen from https://jwalton.info/Matplotlib-latex-PGF/
def get_size(fraction=1, subplots=(1, 1), width_pt=483.6969):
    """Set figure dimensions to sit nicely in our document.

    Parameters
    ----------
    width_pt: float
            Document width in points
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    subplots: array-like, optional
            The number of rows and columns of subplots.
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    # Width of figure (in pts)
    fig_width_pt = width_pt * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    golden_ratio = (5**.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])

    return (fig_width_in, fig_height_in)

def figure(fraction=1, subplots=(1, 1)):
    return plt.figure(figsize=get_size(fraction, subplots))

def savenshow(basename):
    plt.tight_layout()
    plt.savefig(basename + ".png")
    plt.savefig(basename + ".pdf")
    plt.savefig(basename + ".pgf")
    plt.show()

# Default figure is 1 line wide
matplotlib.rcParams['figure.figsize'] = get_size(1)
