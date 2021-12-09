
"""
short matplotlib utility methods
"""

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
        fig_name = os.path.join(save_dir, fig_title + '.png')
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
