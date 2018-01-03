
"""
short bokeh utility methods
"""

def axis_labels(axis, labels):
    """
    set labels on axis with FuncTickFormatter and custom JS

    for example,

    from bokeh.plotting import figure
    from bokeh.io import show, output_file
    from pyrb.bokeh import axis_labels

    fig = figure()
    fig.vbar(x=[0, 1, 2, 3], top=[5, 4, 5, 4], width=0.9)
    axis_labels(fig.xaxis, ['label 1', 'label 2', 'label 3', 'label 4'])

    output_file('delete_me.html')
    show(fig)
    """
    from bokeh.models import FuncTickFormatter

    # assert that one tick already exists for each label
    assert len(axis[0].ticker.ticks) == len(labels)

    # create the FuncTickFormatter object
    label_dict = {a: b for (a, b) in zip(range(len(labels)), labels)}
    axis.formatter = FuncTickFormatter(code=
        """
        var labels = {};
        return labels[tick];
        """.format(label_dict))

def largefonts(figs, size=12):
    """
    make all fonts 'size' for each bokeh Figure object in figs list
    """
    from bokeh.plotting import Figure

    # ensure figs is a list of bokeh Figure objects
    if not isinstance(figs, list):
        figs = [figs]
    figs = [x for x in figs if isinstance(x, Figure)]

    # create 3 sizes based on the 'size' input
    s1 = '{}pt'.format(size)
    s2 = '{}pt'.format(size)
    s3 = '{}pt'.format(size + 2)

    # update fontsize for each Figure in figs list
    for fig in figs:

        # title
        fig.title.text_font_size = s3

        # xaxis
        fig.xaxis.axis_label_text_font_size = s2
        fig.xaxis.major_label_text_font_size = s1

        # yaxis
        fig.yaxis.axis_label_text_font_size = s2
        fig.yaxis.major_label_text_font_size = s1

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

def format_fig(fig, xlabel='', ylabel='', title='', size=12):
    """
    apply xlabel, ylabel, and title to a bokeh figure object
    apply pyrb.bokeh.largefonts with size
    """
    fig.xaxis.axis_label = xlabel
    fig.yaxis.axis_label = ylabel
    fig.title.text = title
    largefonts(fig, size)
