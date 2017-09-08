
"""
short bokeh utility methods
"""


def get_glyphs(fig):
    """
    return a list of bokeh GlyphRenderer objects rendered by fig
    """
    from bokeh.models import GlyphRenderer
    return [x for x in fig.renderers if isinstance(x, GlyphRenderer)]

def empty_data():
    """
    return an empty ColumnDataSource object
    this can be used to initialize a glyph and update the data source later on
    """
    from bokeh.models import ColumnDataSource
    return ColumnDataSource(data={'x': [], 'y': []})

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

def autoload_static(model, script_path):
    """
    simplifies usage of bokeh.embed.autoload_static
    see source code
    """

    from bokeh.embed import autoload_static
    from bokeh.resources import CDN

    script, tag = autoload_static(model=model, resources=CDN, script_path=script_path)
    with open(script_path, 'w') as fid:
        fid.write(script)

    return tag
