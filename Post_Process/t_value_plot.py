import numpy as np 
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt 
from mpl_toolkits.axes_grid1 import AxesGrid
from scipy import stats

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
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
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
              fontsize=7)
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
    features = ["smoothness", "BPDIA_vs_BPSYS", "BPDIA_vs_HRTRT", "BPDIA_vs_O2SAT", "BPDIA_vs_PP", "BPSYS_vs_HRTRT", "BPSYS_vs_O2SAT", "BPSYS_vs_PP", "HRTRT_vs_O2SAT", "HRTRT_vs_PP", "O2SAT_vs_PP"]
    # reps = ["rep 1", "rep 2", "rep 3", "rep 4"]
    reps = ["sepsis (4000)", "nonsepsis (4000)", "all (4000+4000)"]
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

    t_values_sepsis = [-32.97, 12.39, 3.56, -0.67, 0.96, 6.00, -3.45, -8.71, -4.86, 5.48, -1.86]
    t_values_nonsepsis = [-27.71, 15.69, 6.73, -1.24, 1.53, 9.00, -0.50, -11.11, -4.04, 5.07, 0.50]
    t_values = [-42.91, 19.74, 7.19, -1.33, 1.74, 10.51, -2.86, -13.96, -6.31, 7.47, -0.99]
    t_values_table = np.array([t_values_sepsis, t_values_nonsepsis, t_values])

    orig_cmap = matplotlib.cm.coolwarm
    # vmin_sepsis, vmax_sepsis = np.min(t_values_sepsis), np.max(t_values_sepsis)
    # midpoint_sepsis = 1. - vmax_sepsis/ (vmax_sepsis - vmin_sepsis)
    # vmin_nonsepsis, vmax_nonsepsis = np.min(t_values_nonsepsis), np.max(t_values_nonsepsis)
    # midpoint_nonsepsis = 1. - vmax_nonsepsis/ (vmax_nonsepsis - vmin_nonsepsis)
    vmin_table, vmax_table = np.min(t_values_table), np.max(t_values_table)
    midpoint_table = 1. - vmax_table/ (vmax_table - vmin_table)
     
    # shifted_cmap_sepsis = shiftedColorMap(orig_cmap, midpoint=midpoint_sepsis, name='shifted')
    # shrunk_cmap_sepsis = shiftedColorMap(orig_cmap, start=0.15, midpoint=midpoint_sepsis, stop=0.85, name='shrunk')
    # shifted_cmap_nonsepsis = shiftedColorMap(orig_cmap, midpoint=midpoint_nonsepsis, name='shifted')
    # shrunk_cmap_nonsepsis = shiftedColorMap(orig_cmap, start=0.15, midpoint=midpoint_nonsepsis, stop=0.85, name='shrunk')
    shifted_cmap_table = shiftedColorMap(orig_cmap, midpoint=midpoint_table, name='shifted')
    shrunk_cmap_table = shiftedColorMap(orig_cmap, start=0.15, midpoint=midpoint_table, stop=0.85, name='shrunk')
    

    alpha = 0.05
    threshold = stats.t.ppf(1 - alpha/2, 1000-1)
    boundary_005 = np.array([-threshold, threshold])
    print("{}\% confidence interval: ({}, {})".format((1-alpha)*100, -threshold, threshold))
    
    alpha = 0.01
    threshold = stats.t.ppf(1 - alpha/2, 1000-1)
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

    fig, ax = plt.subplots()
    im, cbar = heatmap(t_values_table, reps, features, ax = ax, cmap=shifted_cmap_table, cbarlabel="t value")
    texts = annotate_heatmap(im, valfmt="{x}", boundary=boundary_005)
    fig.tight_layout()
    plt.savefig("t_values_005.png")

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

    fig, ax = plt.subplots()
    im, cbar = heatmap(t_values_table, reps, features, ax = ax, cmap=shifted_cmap_table, cbarlabel="t value")
    texts = annotate_heatmap(im, valfmt="{x}", boundary=boundary_001)
    fig.tight_layout()
    plt.savefig("t_values_001.png")

    import pdb
    pdb.set_trace()
