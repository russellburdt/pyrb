
"""
short bokeh utility methods
"""


def column(*args, **kwargs):
    """
    replace bokeh.layouts.column with a special case of bokeh.layouts.gridplot

    gridplot is meant to be called with a list of lists that define a grid:
    e.g. 'gridplot([[plot_1, plot_2], [plot_3, plot_4]])'

    a nice side effect of gridplot is just 1 toolbar is used, whereas in a
    column layout every figure gets its own toolbar (major clutter)

    this column method uses the special case of gridplot for just 1 column
    of figures, thus replacing the standard column layout
    """

    from bokeh.layouts import _handle_children
    from bokeh.layouts import gridplot

    # get a list of figure objects from *args, return a gridplot as a column plot
    figs = _handle_children(*args)
    return gridplot([[x] for x in figs], **kwargs)

def linkfigs(*args, axis='x'):
    """
    link axes for a list of bokeh figures
    'axis' is either 'x', 'y', or 'xy'
    """

    from bokeh.layouts import _handle_children

    # get a list of figure objects from *args, scan over them
    figs = _handle_children(*args)
    for fig in figs[1:]:

        # link x axes for all figures if requested
        if 'x' in axis:
            fig.x_range = figs[0].x_range

        # link y axes for all figures if requested
        if 'y' in axis:
            fig.y_range = figs[0].y_range
