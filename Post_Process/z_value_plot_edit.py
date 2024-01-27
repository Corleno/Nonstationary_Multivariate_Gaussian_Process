import numpy as np 
import matplotlib
#matplotlib.use("Agg")
import matplotlib.pyplot as plt 
from mpl_toolkits.axes_grid1 import AxesGrid, make_axes_locatable
from scipy import stats
from matplotlib import rc
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('font',**{'family':'serif','serif':['Times']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)


def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):
    '''
    Function to offset the "center" of a colormap. Useful for
    data with a negative min and positive max and you want the
    middle of the colormap's dynamic range to be at zero.

    Input
    -----
      cmap : The matplotlib colormap to be altered
      start : Offset from lowest point in the colormap's range.
          Defaults to 0.0 (no lower offset). Should be between
          0.0 and `midpoint`.
      midpoint : The new center of the colormap. Defaults to 
          0.5 (no shift). Should be between 0.0 and 1.0. In
          general, this should be  1 - vmax / (vmax + abs(vmin))
          For example if your data range from -15.0 to +5.0 and
          you want the center of the colormap at 0.0, `midpoint`
          should be set to  1 - 5/(5 + 15)) or 0.75
      stop : Offset from highest point in the colormap's range.
          Defaults to 1.0 (no upper offset). Should be between
          `midpoint` and 1.0.
    '''
    cdict = {
        'red': [],
        'green': [],
        'blue': [],
        'alpha': []
    }
    


    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)

    # shifted index to match the data
    shift_index = np.hstack([
        np.linspace(0.0, midpoint, 128, endpoint=False), 
        np.linspace(midpoint, 1.0, 129, endpoint=True)
    ])

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))

    newcmap = matplotlib.colors.LinearSegmentedColormap(name, cdict)
    plt.register_cmap(cmap=newcmap)

    return newcmap


def heatmap(data, row_labels, col_labels, ax=None, cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    
    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = ax.figure.colorbar(im, cax=cax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}", textcolors=["black", "yellow", "black"], boundary=None, threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A list or array of two color specifications.  The first is used for
        values below a threshold, the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """



    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Normalize the boundary rto the images color range
    if boundary is not None:
        lower_bound = im.norm(boundary[0])
        upper_bound = im.norm(boundary[1])
    else:
        print("Please add boundary values!")

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center",
              fontsize=10)
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if im.norm(data[i, j]) < lower_bound:
                index = 0
            elif im.norm(data[i,j]) < upper_bound:
                index = 1
            else:
                index = 2
            print(data[i,j], index)
            kw.update(color=textcolors[index])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


if __name__ == "__main__":
#    rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    rc('font',**{'family':'serif','serif':['Times']})
    ## for Palatino and other serif fonts use:
#    rc('font',**{'family':'serif','serif':['Palatino']})
    rc('text', usetex=True)
    # features = [r"$\textbf{Smoothness}$", r"\textbf{BPDIA vs BPSYS}", r"\textbf{BPDIA vs HRTRT}", r"\textbf{BPDIA vs O2SAT}", r"\textbf{BPDIA vs PP}", r"\textbf{BPSYS vs HRTRT}", r"\textbf{BPSYS vs O2SAT}", r"\textbf{BPSYS vs PP}", r"\textbf{HRTRT vs O2SAT}", r"\textbf{HRTRT vs PP}", r"\textbf{O2SAT vs PP}"]
    features = [r'$\textbf{Smoothness}$', r"\textbf{BPDIA vs BPSYS}", r"\textbf{BPDIA vs HRTRT}", r"\textbf{BPDIA vs O2SAT}", r"\textbf{BPSYS vs HRTRT}", r"\textbf{BPSYS vs O2SAT}", r"\textbf{HRTRT vs O2SAT}"]
    # reps = ["rep 1", "rep 2", "rep 3", "rep 4"]
    reps = [r"\textbf{Sepsis}", r"\textbf{Nonsepsis}"]
    # t_values_sepsis0 = [-16.64, 6.97, 1.95, -0.93, 1.08, 2.80, -2.27, -4.67, -2.29, 3.49, -0.87]
    # t_values_nonsepsis0 = [-14.26, 7.83, 3.64, -0.55, 0.24, 5.53, -0.44, -6.78, -1.29, 3.53, -0.48]
    # t_values_sepsis1 = [-17.05, 6.01, 3.15, -1.10, 0.49, 4.62, -2.59, -4.44, -3.21, 2.90, 0.77]
    # t_values_nonsepsis1 = [-11.98, 7.93, 2.11, -0.25, 1.78, 2.59, 1.22, -4.01, -2.35, 1.07, 1.15]
    # t_values_sepsis2 = [-14.38, 5.57, 0.58, -0.38, -0.81, 1.98, -1.46, -5.64, -2.78, 1.72, -0.45]
    # t_values_nonsepsis2 = [-14.60, 7.94, 3.74, -1.28, 0.25, 5.43, -0.07, -5.40, -2.47, 3.75, 0.72]
    # t_values_sepsis3 = [-18.04, 6.33, 2.01, 0.77, 1.30, 3.06, -0.66, -2.98, -1.59, 3.04, -0.67]
    # t_values_nonsepsis3 = [-13.35, 7.98, 4.31, -0.62, 0.90, 4.09, -1.14, -5.78, -1.85, 1.38, -0.04]
    # t_values_sepsis = np.array([t_values_sepsis0, t_values_sepsis1, t_values_sepsis2, t_values_sepsis3])
    # t_values_nonsepsis = np.array([t_values_nonsepsis0, t_values_nonsepsis1, t_values_nonsepsis2, t_values_nonsepsis3])

    # z_values_sepsis = [-17.93, 7.69, 2.33, -0.63, 0.85, 3.89, -2.59, -5.63, -3.00, 3.71, -1.30]
    # z_values_nonsepsis = [-15.52, 9.93, 4.61, -2.46, 1.30, 6.08, -0.89, -5.59, -2.73, 3.35, -0.72]
    # z_values_sepsis = [-12.40, 6.32, 3.73, -1.90, 4.43, -0.44, -2.59]
    # z_values_nonsepsis = [-5.12, 5.06, 3.04, -0.76, 3.60, -0.38, -2.28]
    z_values_sepsis = [-11.00, 3.86, 4.36, 0.25, 5.06, 0.76, -2.53]
    z_values_nonsepsis = [-5.25, 6.32, 0.70, -1.33, 2.15, -0.44, -2.15]
    z_values_table = np.array([z_values_sepsis, z_values_nonsepsis])

    orig_cmap = matplotlib.cm.coolwarm
    # vmin_sepsis, vmax_sepsis = np.min(t_values_sepsis), np.max(t_values_sepsis)
    # midpoint_sepsis = 1. - vmax_sepsis/ (vmax_sepsis - vmin_sepsis)
    # vmin_nonsepsis, vmax_nonsepsis = np.min(t_values_nonsepsis), np.max(t_values_nonsepsis)
    # midpoint_nonsepsis = 1. - vmax_nonsepsis/ (vmax_nonsepsis - vmin_nonsepsis)
    vmin_table, vmax_table = np.min(z_values_table), np.max(z_values_table)
    midpoint_table = 1. - vmax_table/ (vmax_table - vmin_table)
     
    # shifted_cmap_sepsis = shiftedColorMap(orig_cmap, midpoint=midpoint_sepsis, name='shifted')
    # shrunk_cmap_sepsis = shiftedColorMap(orig_cmap, start=0.15, midpoint=midpoint_sepsis, stop=0.85, name='shrunk')
    # shifted_cmap_nonsepsis = shiftedColorMap(orig_cmap, midpoint=midpoint_nonsepsis, name='shifted')
    # shrunk_cmap_nonsepsis = shiftedColorMap(orig_cmap, start=0.15, midpoint=midpoint_nonsepsis, stop=0.85, name='shrunk')
    shifted_cmap_table = shiftedColorMap(orig_cmap, midpoint=midpoint_table, name='shifted')
    shrunk_cmap_table = shiftedColorMap(orig_cmap, start=0.15, midpoint=midpoint_table, stop=0.85, name='shrunk')
    
    alpha = 0.05
    threshold = stats.norm.ppf(1 - alpha/2)
    boundary_005 = np.array([-threshold, threshold])
    print("{}\% confidence interval: ({}, {})".format((1-alpha)*100, -threshold, threshold))
    
    alpha = 0.01
    threshold = stats.norm.ppf(1 - alpha/2)
    boundary_001 = np.array([-threshold, threshold])
    print("{}\% confidence interval: ({}, {})".format((1-alpha)*100, -threshold, threshold))

    # fig, ax = plt.subplots()
    # im, cbar = heatmap(t_values_sepsis, reps, features, ax = ax, cmap=shifted_cmap_sepsis, cbarlabel="t value")
    # texts = annotate_heatmap(im, valfmt="{x}", boundary=boundary_005)
    # fig.tight_layout()
    # plt.savefig("t_values_of_sepsis_005.png")

    # fig, ax = plt.subplots()
    # im, cbar = heatmap(t_values_nonsepsis, reps, features, ax = ax, cmap=shifted_cmap_nonsepsis, cbarlabel="t value")
    # texts = annotate_heatmap(im, valfmt="{x}", boundary=boundary_005)
    # fig.tight_layout()
    # plt.savefig("t_values of nonsepsis_005.png")

    fig, ax = plt.subplots(figsize=(8, 3), dpi=300)
#    ax.set_aspect(aspect=1)
    im, cbar = heatmap(z_values_table, reps, features, ax = ax, cmap=shifted_cmap_table, cbarlabel=r"$\bf{z}$ \textbf{value}", aspect="auto")
    texts = annotate_heatmap(im, valfmt="{x}", boundary=boundary_005)
    fig.tight_layout()
    plt.savefig("z_values_005_update1.pdf")

    # fig, ax = plt.subplots()
    # im, cbar = heatmap(t_values_sepsis, reps, features, ax = ax, cmap=shifted_cmap_sepsis, cbarlabel="t value")
    # texts = annotate_heatmap(im, valfmt="{x}", boundary=boundary_001)
    # fig.tight_layout()
    # plt.savefig("t_values_of_sepsis_001.png")

    # fig, ax = plt.subplots()
    # im, cbar = heatmap(t_values_nonsepsis, reps, features, ax = ax, cmap=shifted_cmap_nonsepsis, cbarlabel="t value")
    # texts = annotate_heatmap(im, valfmt="{x}", boundary=boundary_001)
    # fig.tight_layout()
    # plt.savefig("t_values_of_nonsepsis_001.png")

    fig, ax = plt.subplots(figsize=(8, 3), dpi=300)
    im, cbar = heatmap(z_values_table, reps, features, ax=ax, cmap=shifted_cmap_table, cbarlabel="z value", aspect='auto')
    texts = annotate_heatmap(im, valfmt="{x}", boundary=boundary_001)
    fig.tight_layout()
    plt.savefig("z_values_001_updated1.pdf")
    plt.show()
#    import pdb
#    pdb.set_trace()
