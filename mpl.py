
"""
matplotlib utils
"""

def block_chart_from_dataframe(dm, figsize=(12, 4), fig=None, ax=None, label=None, xlabel='', title='block chart', size=16, loc='upper left', bbox_to_anchor=(0, 1), alpha=0.6):
    """
    block chart based on pandas DataFrame
    - each row becomes a block on chart
    - requisite colums are 'start' and 'end', may be timestamp or numeric objects
    - legend uses label column
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from ipdb import set_trace


    # validate args
    assert isinstance(dm, pd.DataFrame) and isinstance(figsize, tuple) and isinstance(title, str) and isinstance(size, int)
    assert ('start' in dm.columns) and ('end' in dm.columns)
    if label is not None:
        assert isinstance(label, list)
        assert all([x in dm.columns for x in label])
    c0 = all([pd.core.dtypes.common.is_datetime64_dtype(dm[col]) for col in ['start', 'end']])
    c1 = all([pd.core.dtypes.common.is_numeric_dtype(dm[col]) for col in ['start', 'end']])
    assert np.logical_xor(c0, c1)
    assert all(dm['end'] >= dm['start'])

    # fig and ax objects
    if (fig is None) and (ax is None):
        fig, ax = open_figure(title, figsize=figsize)

    # scan over blocks
    for x, row in dm.iterrows():

        # create block, set color and label
        ta, tb = row['start'], row['end']
        p = ax.fill_between(x=np.array([ta, tb]), y1=np.tile(0, 2), y2=np.tile(1, 2), alpha=alpha)
        if 'colors' in dm.columns:
            p.set_color(dm.loc[x, 'colors'])
        if label is not None:
            p.set_label(', '.join([row[x] for x in label]))

    # clean up
    if label is not None:
        leg = ax.legend(loc=loc, bbox_to_anchor=bbox_to_anchor, title=', '.join(label), title_fontproperties={'weight':'bold'}, handlelength=4, shadow=True, fancybox=True)
        leg._legend_box.align = 'left'
        for cx in ax.get_legend().get_texts():
            cx.set_fontsize(size - 4)
    ax.set_ylim(0, 1)
    ax.set_yticklabels([])
    if c0:
        ax.set_xlim(dm.iloc[0]['start'] - pd.Timedelta(hours=1), dm.iloc[-1]['end'] + pd.Timedelta(hours=1))
        format_axes(xlabel, '', title, ax, apply_concise_date_formatter=True)
    else:
        margin = 0.01 * (dm['end'].max() - dm['start'].min())
        ax.set_xlim(dm['start'].min() - margin, dm['end'].max() + margin)
        format_axes(xlabel, '', title, ax)
    ax.grid(visible=False, axis='both')
    largefonts(size, legend=False)
    fig.tight_layout()

def table_from_dataframe(dm, figsize=(12, 4), title='table', size=16, colors=None, alpha=0.6, xscale=1, yscale=2, fig=None, ax=None):
    """
    table based on pandas DataFrame
    - each row of dataframe becomes a row in table
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from ipdb import set_trace


    # validate args
    assert isinstance(dm, pd.DataFrame)
    assert dm.shape[0] < 10
    assert isinstance(figsize, tuple)
    assert isinstance(title, str)
    assert isinstance(size, int)

    # row colors
    if colors is not None:
        assert isinstance(colors, np.ndarray)
        assert colors.shape == (dm.shape[0], 4)

    # figure object and scan over blocks / colors
    if (fig is None) and (ax is None):
        fig, ax = open_figure(title, figsize=figsize)

    # create table on ax
    table = ax.table(cellText=dm.values, colLabels=dm.columns, cellLoc='center', loc='upper center')
    if colors is not None:
        for pos, cell in table._cells.items():
            if (pos[0] == 0) or (pos[1]) == -1:
                continue
            cell.set_color(colors[pos[0] - 1])
            cell.set_alpha(alpha)
    table.auto_set_column_width(list(range(dm.shape[1])))
    table.auto_set_font_size(False)
    table.set_fontsize(size)
    table.scale(xscale, yscale)
    ax.grid(visible=False, axis='both')
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_axis_off()

def metric_distribution(x, bins, title='distribution', ax_title=None, xlabel=None, ylabel='bin count', fig=None, ax=None,
        legend=None, loc='upper left', bbox_to_anchor=(1, 1), figsize=(12, 6), size=18, logscale=False, pdf=False, alpha=0.8):
    """
    create or update matplotlib distribution
    - x - data
    - bins - bins for data
    - title - figure title always, axes title if ax_title is None
    - ax_title - axes title if not None
    - xlabel - axes x label
    - legend - label for legend (None to disable)
    - loc, bbox_to_anchor - legend properties if applicable
    - figsize and size - formatting
    - logscale - use log-scale for y-axis
    - pdf - scale data as a probability distribution
    """

    import numpy as np

    # validate bins, remove nans
    assert all(np.sort(bins) == bins)
    bx = np.unique(np.diff(bins))
    assert bx.size >= 1
    if bx.size > 1:
        assert all([np.isclose(bx[0], bx[xi]) for xi in range(1, bx.size)])
    width = bx[0]
    x = x[~np.isnan(x)]

    # fig and ax objects
    if (fig is None) and (ax is None):
        fig, ax = open_figure(title, figsize=figsize)

    # distribution
    xd = np.digitize(x, bins)
    height = np.array([(xd == xi).sum() for xi in range(0, bins.size + 1)])

    # scale as pdf
    if pdf:
        height = height / height.sum()

    # distribution and limits data based on cases of height data
    assert height.size == bins.size + 1
    if (height[0] == 0) and (height[-1] == 0):
        centers = (bins[1:] + bins[:-1]) / 2
        xmin = bins[0]
        xmax = bins[-1]
        bins = bins[:-1]
        height = height[1:-1]
    elif (height[0] > 0) and (height[-1]) > 0:
        bins = np.hstack((bins[0] - width, bins))
        bx = np.hstack((bins, bins[-1] + width))
        centers = (bx[1:] + bx[:-1]) / 2
        xmin = bx[0]
        xmax = bx[-1]
    elif (height[0] == 0) and (height[-1] > 0):
        height = height[1:]
        bx = np.hstack((bins, bins[-1] + width))
        centers = (bx[1:] + bx[:-1]) / 2
        xmin = bx[0]
        xmax = bx[-1]
    elif (height[0] > 0) and (height[-1] == 0):
        bins = np.hstack((bins[0] - width, bins[:-1]))
        height = height[:-1]
        bx = np.hstack((bins, bins[-1] + width))
        centers = (bx[1:] + bx[:-1]) / 2
        xmin = bx[0]
        xmax = bx[-1]
    else:
        raise ValueError('new case')

    # plot distribution and outline
    ax.bar(x=bins, height=height, align='edge', width=width, alpha=alpha)
    ax.plot(centers, height, '-', lw=2, label=legend)

    # clean up
    ax.set_xlim(xmin, xmax)
    title = title if ax_title is None else ax_title
    if pdf:
        format_axes(xlabel, 'probability density', title, ax)
    else:
        format_axes(xlabel, ylabel, title, ax)
    if logscale:
        ax.set_yscale('log')
    if legend is not None:
        leg = ax.legend(loc=loc, bbox_to_anchor=bbox_to_anchor)
        for x in leg.get_lines():
            x.set_linewidth(4)
    largefonts(size)
    fig.tight_layout()

    return fig, ax

def expanding_bar_chart(x, labels, text=None, title=None, xlabel=None, legend=None, figsize=(10, 6), size=14, height=0.8, fig=None, ax=None):
    """
    create or update horizontal bar chart
    - x - data
    - labels - for y-axis
    - text - text at end of bars
    - title - figure title and axes title
    - xlabel - axes x label
    - legend - label for legend
    - figsize, size, height - misc formatting
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from pyrb.mpl import open_figure, format_axes, save_pngs, largefonts
    from ipdb import set_trace
    plt.style.use('bmh')

    # validate args
    assert isinstance(x, np.ndarray)
    assert isinstance(labels, np.ndarray)
    assert np.unique(labels).size == labels.size
    assert x.size == labels.size
    if text is not None:
        assert isinstance(text, np.ndarray) and (text.dtype.type is np.str_) and (text.size == x.size)
    xlabel = xlabel if xlabel is not None else ''
    title = title if title is not None else 'bar chart'

    # fig and ax objects
    if (fig is None) and (ax is None):
        fig, ax = open_figure(title, figsize=figsize)
    def clean_up():
        largefonts(size)
        format_axes(xlabel, '', title, ax)
        if legend is not None:
            ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
        largefonts(size)
        fig.tight_layout()

    # initialize chart
    if not ax.containers:
        y = range(x.size)
        ax.barh(y=y, width=x, height=height, align='center', label=legend)
        ax.set_yticks(y)
        ax.set_ylim(-height, x.size - 1 + height)
        ax.set_yticklabels(labels)
        clean_up()
        return fig, ax

    # existing data
    assert ax.get_xlabel() == xlabel
    width = np.array([[cx.get_width() for cx in container] for container in ax.containers])
    y = np.array([[cx.get_y() for cx in container] for container in ax.containers])
    ylabels = np.array([x.get_text() for x in ax.get_yticklabels()])

    # update legend
    assert ax.get_legend() is not None
    legend = np.hstack((np.array([x.get_text() for x in ax.get_legend().texts]), legend))

    # update for existing labels
    xn = np.expand_dims(np.array([x[lx == labels] if (lx in labels) else np.array([0]) for lx in ylabels]).flatten(), axis=0)
    assert width.shape[1] == xn.shape[1]
    width = np.vstack((width, xn))

    # update for new labels
    new = np.array([x for x in labels if x not in ylabels])
    if new.size > 0:
        w2 = np.vstack((np.expand_dims(np.zeros(new.size), axis=0),
            np.expand_dims(np.array([x[labels == xx] for xx in new]).flatten(), axis=0)))
        width = np.hstack((w2, width))
        labels = np.hstack((new, ylabels))
    else:
        labels = ylabels

    # close existing chart, regenerate chart (needs refactor for n > 2)
    plt.close(fig)
    fig, ax = open_figure(title, figsize=figsize)
    n = width.shape[0]
    assert legend.size == n
    height /= n
    y = np.arange(labels.size)
    assert n < 3
    if n == 2:
        ax.barh(y=y, width=width[0], height=height, align='edge', label=legend[0])
        ax.barh(y=y - height, width=width[1], height=height, align='edge', label=legend[1])
    ax.set_yticks(y)
    ax.set_ylim(-height, labels.size - 1 + height)
    ax.set_yticklabels(labels)
    clean_up()

    return fig, ax

def hbar_chart(x, labels, text=None, title=None, xlabel=None, legend=None, figsize=(10, 6), size=14, height=0.8, xscale=1.2, fig=None, ax=None):
    """
    horizontal bar chart
    - x - data
    - labels - for y-axis
    - text - text at end of bars
    - title - figure title and axes title
    - xlabel - axes x label
    - legend - label for legend
    - figsize, size, height - misc formatting
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from pyrb.mpl import open_figure, format_axes, save_pngs, largefonts
    from ipdb import set_trace
    plt.style.use('bmh')

    # validate args
    assert isinstance(x, np.ndarray)
    assert isinstance(labels, np.ndarray)
    assert np.unique(labels).size == labels.size
    assert x.size == labels.size
    if text is not None:
        assert isinstance(text, np.ndarray) and (text.dtype.type is np.str_) and (text.size == x.size)
    xlabel = xlabel if xlabel is not None else ''
    title = title if title is not None else 'bar chart'

    # fig and ax objects
    if (fig is None) and (ax is None):
        fig, ax = open_figure(title, figsize=figsize)

    # create chart
    y = range(x.size)
    ax.barh(y=y, width=x, height=height, align='center', label=legend)
    ax.set_yticks(y)
    ax.set_ylim(-height, x.size - 1 + height)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, xscale * x.max())
    if text is not None:
        for xx, yy, t in zip(x, y, text):
            ax.text(xx, yy, t, ha='left', va='center', fontsize=size - 2, fontweight='bold')
    largefonts(size)
    format_axes(xlabel, '', title, ax)
    if legend is not None:
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    largefonts(size)
    fig.tight_layout()

def get_current_figs():
    """
    get list of current matplotlib figure managers
    """

    import matplotlib

    managers = matplotlib._pylab_helpers.Gcf.get_all_fig_managers()
    titles = [x.canvas.manager.get_window_title() for x in managers]
    return managers, titles

def subplots_adjust(**kwargs):
    """
    apply tight_layout and subplots_adjust with kwargs to all figs
    """

    figs = get_current_figs()[0]
    for fig in figs:
        fig.canvas.figure.tight_layout()
        fig.canvas.figure.subplots_adjust(**kwargs)

def update_figs(show=False):
    """
    redraw all figures and run plt.show for show=True
    """
    figs = get_current_figs()
    for fig in figs[0]:
        fig.canvas.draw()
    if show:
        plt.show()

def thicklines(size=3):
    """
    adjust all lines in all matplotlib figures for linewidth proportional to size
    """

    # get list of current figs
    figs = get_current_figs()

    # scan thru figs, find each axes
    for fig in figs[0]:
        axes = fig.canvas.figure.get_axes()

        # scan thru axes, find all lines in the plot
        for ax in axes:
            lines = ax.get_lines()

            # scan thru lines, change linewidth if not specline
            for line in lines:
                if line.get_linewidth() == 2 and line.get_linestyle() == '--' and line.get_color() == 'r':
                    continue
                line.set_linewidth(size)

            # update legend to reflect modified linewidths
            leg = ax.get_legend()
            if leg is None: continue
            lines = leg.get_lines()
            for line in lines:
                line.set_linewidth(size)

    # redraw figures
    update_figs()

def largefonts(size=18, title=True, xaxis=True, yaxis=True, legend=True):
    """
    adjust font size in all matplotlib figures proportional to 'size' input
    use kwargs for independent adjustment of title, xaxis, yaxis, and legend
    """

    # get list of current figs
    figs = get_current_figs()

    # scan thru figs, find each axes
    for fig in figs[0]:
        axes = fig.canvas.figure.get_axes()

        # scan thru axes, change font size of title, labels, ticks, legend
        for ax in axes:

            # update fontsize of title
            if title:
                ax.title.set_fontsize(size + 2)

            # update fontsize of ylabel and yticks
            if yaxis:
                ax.set_ylabel(ax.get_ylabel(), fontsize = size)
                ticks = ax.yaxis.get_majorticklabels()
                for tick in ticks:
                    tick.set_fontsize(size)

            # update fontsize of xlabel and xticks
            if xaxis:
                ax.set_xlabel(ax.get_xlabel(), fontsize = size)
                ticks = ax.xaxis.get_majorticklabels()
                for tick in ticks:
                    tick.set_fontsize(size)
                ax.xaxis.get_offset_text().set_size(size)

            # update fontsize of legend
            if legend:
                leg = ax.get_legend()
                if leg is None:
                    continue
                for child in leg.get_texts():
                    child.set_fontsize(size)

    # finally, redraw the figures
    update_figs()

def format_axes(xlabel, ylabel, title=None, axes=None, apply_concise_date_formatter=False):
    """
    set xlabel, ylabel, title, and turn on grid for ax or plt.gca()
    ax can be a list of Axes or a single Axes object
    * apply ConciseDateFormatter if requested
    """

    import matplotlib.pyplot as plt

    # get ax as current axes if not provided
    if axes is None:
        axes = plt.gca()

    # make sure ax is iterable
    if not hasattr(axes, '__iter__'):
        axes = [axes]

    # format axes for all axes
    for ax in axes:
        assert isinstance(ax, plt.Axes)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if title:
            ax.set_title(title)
        ax.grid(True)
        if apply_concise_date_formatter:
            import matplotlib.dates as mdates
            locator = mdates.AutoDateLocator()
            formatter = mdates.ConciseDateFormatter(locator)
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(formatter)

def addspecline(spec, ax=None, percent_margin=5, color='r'):
    """
    add a specification line to matplotlib axes object
    """

    import matplotlib.pyplot as plt
    import numpy as np

    # sometimes this function needs to be called and do nothing
    if spec is None:
        return

    # get ax as current axes if not provided as input, handle case of iterable ax
    if ax is None:
        ax = plt.gca()
    if isinstance(ax, list) or isinstance(ax, np.ndarray):
        for x in ax:
            addspecline(spec, ax=x, percent_margin=percent_margin, color=color)
        return

    # ensure spec is a list if provided as a float or an int
    # track if only a single spec was provided
    if type(spec) is int or type(spec) is float:
        singleSpec = True
        spec = [spec]
    else:
        singleSpec = False

    # get original x lims, then make axis tight around data
    x0 = ax.get_xlim()
    ax.autoscale(tight = True)

    # add spec lines one at a time, use dashed line of 'color' (default is red)
    for lim in spec:
        ax.plot(ax.get_xlim(), np.tile(lim, [2, 1]), '--', linewidth = 2, color = color)

    # autoscale tight and restore original xlims
    ax.autoscale(tight = True)
    ax.set_xlim(x0)

    # set better ylims based on lims after tight scaling
    a, b = ax.get_ylim()

    if singleSpec:
        if spec[0] > 0:         # adjust a or b in the case of a
            a = 0               # single spec provided so that one is zero
        else:
            b = 0

    extra = (percent_margin/100) * (b - a)
    if a == 0 and b > 0:
        y1 = [0, b + extra]
    elif a < 0 and b == 0:
        y1 = [a - extra, 0]
    else:
        y1 = [a - extra, b + extra]

    ax.set_ylim(y1)

def maximize_figs():
    """
    maximize all open figures
    (may not work with all matplotlib backends)
    """

    import matplotlib

    figList = get_current_figs()
    for fig in figList[0]:
        figTitle = fig.get_window_title()
        matplotlib._pylab_helpers.Gcf.set_active(fig)
        fig.window.showMaximized()

def save_pngs(save_dir, maximize=False, close=True):
    """
    save all open matplotlib figures to png files
    """

    import matplotlib
    import matplotlib.pyplot as plt
    import os

    fig_list = get_current_figs()
    for fig in fig_list[0]:
        fig_title = fig.get_window_title()
        fig_name = os.path.join(save_dir, fig_title + '.png').replace('\n', ' ')
        matplotlib._pylab_helpers.Gcf.set_active(fig)
        if maximize:
            fig.window.showMaximized()
        fig.canvas.figure.savefig(fig_name, facecolor='none', bbox_inches='tight', pad_inches=0.05)
    if close:
        plt.close('all')

def open_figure(fig_title='', rows=1, columns=1, handle_2d1d=False, handle_1d0d=False, use_existing=True, **kwargs):
    """
    wrapper around plt.subplots:
    -- opens a figure of fig_title, returns fig and axes objects
    -- if a figure of fig_title already exists, returns existing fig and axes objects
    -- **kwargs are passed to plt.subplots
    -- this method is useful when additional data needs to be added to existing axes, and only
        the figure name is available (i.e. the figure object is not directly available)
    -- the 'handle_2d1d' (default False) kwarg, when True, will return 1d arrays with an extra dimension, i.e. shape is (x, 1)
        in doing so, the method can transition from 2d to 1d behavior without errors
    -- the 'handle_1d0d' (default False) kwarg, when True, will return single axes objects in a 1d arrays, i.e. shape is (x, )
        in doing so, the method can transition from 1d to 0d behavior without errors
    """

    import matplotlib.pyplot as plt
    import numpy as np

    # get a list of current figs
    if use_existing:
        figs = get_current_figs()
    else:
        figs = ([], [])

    # get fig and ax objects differently depending on existence of fig_title
    if fig_title in figs[1]:
        fig = figs[0][figs[1].index(fig_title)].canvas.figure
        if rows > 1 and columns > 1:
            ax = np.array(fig.get_axes()).reshape(rows, columns)
        elif rows == 1 and columns == 1:
            ax = fig.get_axes()[0]
        else:
            ax = np.array(fig.get_axes())
    else:
        fig, ax = plt.subplots(rows, columns, **kwargs)
        fig.canvas.manager.set_window_title(fig_title)

    # add an extra dimension to 1d ax array if requested
    if handle_2d1d and ax.ndim == 1:
        if rows == 1:
            ax = np.expand_dims(ax, axis=0)
        elif columns == 1:
            ax = np.expand_dims(ax, axis=1)

    # store ax in a 1d numpy array if requested
    if handle_1d0d and isinstance(ax, plt.Axes):
        ax = np.array([ax])
        assert ax.ndim == 1

    # return figure and axes objects
    return fig, ax

def get_random_closed_data_marker(n=None):
    """
    return a random closed data marker available in matplotlib for n=None
    return a list of unique n closed data markers for integer n input,
    handle the case of recycling data markers when the available list is exhausted
    """
    from random import choice

    # define a list of closed matplotlib markers
    mlist = ['o', 'v', '^', '<', '>', 's', 'p', '*', 'h', 'H', 'd', 'D']

    # return the markers
    if n is None:
        return choice(mlist)
    else:
        mlist_tmp = mlist.copy()
        markers = []
        for _ in range(n):
            if len(mlist_tmp) == 0:
                mlist_tmp = mlist
            marker = choice(mlist_tmp)
            markers.append(marker)
            mlist_tmp.remove(marker)
        return markers
