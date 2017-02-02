
"""
short bokeh utility methods
"""

from bokeh.models.widgets import Slider
class SliderWithButtons(Slider):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # from ipdb import set_trace
        # set_trace()

def slider_with_buttons(width=300, dim=40, **kwargs):
    """
    return a bokeh layout object representing a slider widget with button widgets
    on each side that increment and decrement the slider value by step
    the slider 'on_change' method can be accessed from the returned object as:

    ** 29 Jan 2017, this does not work for a few reasons:
        the on_change method of the slider is not easily accessed by the
        calling method as the returned object is a Row layout, and the slider
        object is a child of a widgetbox making up that layout, i.e. it is
        available as: returned_object.children[1].children[0].on_change

        callbacks linked the slider are not working as bokeh thinks the 'value'
        attribute is that of the returned Row layout, and this does not have a
        'value' attribute

        the proper way to create this widget is to create a new Slider class
        that properly links the bokeh Slider and Button classes, but this is
        difficult as it is not really clear how the bokeh Slider class is
        created (it inherits a bokeh class that inherits a bokeh class...)
    **
    """

    from bokeh.models.widgets import Slider, Button
    from bokeh.layouts import widgetbox, row

    # initialize widget objects
    start = kwargs.pop('start', None)
    end = kwargs.pop('end', None)
    step = kwargs.pop('step', None)
    value = kwargs.pop('value', None)
    slider = Slider(start=start, end=end, step=step, value=value, **kwargs)
    minus = Button(label='-')
    plus = Button(label='+')

    # create callbacks that update the slider on a button press
    def plus_callback():
        slider.value += slider.step
    def minus_callback():
        slider.value -= slider.step

    # set the event handlers
    minus.on_click(minus_callback)
    plus.on_click(plus_callback)

    # return a row layout made up of the slider and button widgets
    return row(
        widgetbox(minus, width=dim, height=dim, sizing_mode='fixed'),
        widgetbox(slider, width=width, height=dim, sizing_mode='fixed'),
        widgetbox(plus, width=dim, height=dim, sizing_mode='fixed'))

def open_figure(N=10, cmap='plasma'):
    """
    returns a figure and list of glyph objects not pointing to any data
    the idea is to return up to N glyph objects that can later be populated
    with data, but already have the colormap set
    """
    return

    from bokeh.plotting import figure

    # initialize a bokeh figure object
    fig = figure()
    glyphs = None
    return fig, glyphs

def column(*args, **kwargs):
    """
    replace bokeh.layouts.column with a special case of bokeh.layouts.gridplot

    gridplot is meant to be called with a list of lists that define a grid:
    e.g. 'gridplot([[plot_1, plot_2], [plot_3, plot_4]])'

    a nice side effect of gridplot is just 1 toolbar is used, whereas in a
    column layout every figure gets its own toolbar

    this column method uses the special case of gridplot for just 1 column
    of figures, thus replacing the standard column layout

    ** another way to do this is just use ncols=1 with bokeh.layouts.column **
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
