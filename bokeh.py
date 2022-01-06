
"""
common bokeh utils
"""

import numpy as np
from bokeh import palettes
from bokeh.plotting import figure
from bokeh.tile_providers import Vendors, get_provider
from bokeh.models import ColumnDataSource, DataRange1d, Range1d, GlyphRenderer, Legend, LegendItem, Band
from bokeh.models import PanTool, BoxZoomTool, WheelZoomTool, ResetTool, SaveTool, TapTool, HoverTool
from bokeh.models import FuncTickFormatter, FixedTicker
from pyrb.processing import gps_to_webm
from ipdb import set_trace

class MultiLineInterface:
    """
    interface for figure object with multiple lines
    - arguments to init
        - width, height, xlabel, ylabel, title, size, legend_location - figure formatting
        - n and palette - number and color palette for lines
        - hover - include hover tool - not assigned any attributes, eg renderers, tooltips, formatters
        - tap - include tap tool - no tap tool callback is created
    - attributes created on init
        - fig - figure object
        - data_sources - length n tuple of ColumnDataSource objects
        - renderers - numpy array, n rows x 2 columns (line and circle glyph for each data source)
        - legend - Legend object
    - methods
        - reset_interface
    """
    def __init__(self, width=700, height=300, xlabel='', ylabel='', title='', size=12, legend_location='bottom_right',
        n=20, palette='Category20_20', hover=False, tap=False):

        # standard tools
        pan, wheel, reset, save = PanTool(), WheelZoomTool(), ResetTool(), SaveTool()
        tools = [pan, wheel, reset, save]

        # hover and tap tools
        self.hover = hover
        self.tap = tap
        hover_tool = HoverTool() if self.hover else None
        tap_tool = TapTool() if self.tap else None
        if self.hover:
            tools.append(hover_tool)
            self.hover = hover_tool
        if self.tap:
            tools.append(tap_tool)
            self.tap = tap_tool

        # figure object
        self.fig = figure(width=width, height=height, tools=tools, toolbar_location='right')
        self.fig.toolbar.logo = None
        self.fig.x_range = DataRange1d()
        self.fig.y_range = DataRange1d()
        self.fig.toolbar.active_drag = pan
        self.fig.toolbar.active_scroll = wheel
        self.fig.toolbar.active_inspect = hover_tool
        self.fig.toolbar.active_tap = tap_tool

        # ColumnDataSource objects
        self.n = n
        self.data_sources = tuple([ColumnDataSource() for _ in range(n)])

        # circle and line glyphs for each data source
        colors = palettes.linear_palette(getattr(palettes, palette), n)
        for src, color in zip(self.data_sources, colors):
            self.fig.circle('x', 'y', source=src, size=6, color=color)
            self.fig.line('x', 'y', source=src, line_width=1, color=color)
        self.renderers = np.array(self.fig.renderers).reshape(n, 2)

        # legend object
        self.legend = Legend(location=legend_location)
        self.fig.add_layout(self.legend, 'center')

        # format axes and reset interface
        format_axes(self.fig, xlabel, ylabel, title, size)
        self.reset_interface()

    def reset_interface(self):
        """
        reset data sources, legend items, hover and tap tools
        """
        for x in range(self.n):
            self.data_sources[x].data = {'x': np.array([]), 'y': np.array([])}
        self.legend.items = []
        if self.hover:
            self.hover.tooltips = []
            self.hover.formatters = {}
            self.hover.renderers = 'auto'
        if self.tap:
            self.tap.renderers = []

class MapInterface:
    """
    interface for map object supporting path, position, and events objects
    - arguments to init
        - width, height, map_vendor, size - map formatting
        - hover - include hover tool - not assigned any attributes, eg renderers, tooltips, formatters
        - tap - include tap tool - no tap tool callback is created
        - path_color - color for line/circle path glyphs
        - position_color - color for asterik position glyph
        - events_color / events_selected_color - colors for circle events glyph
    - attributes created on init
        - map - map as a bokeh figure object
        - ColumnDataSource and list of glyphs for path / position / events
        - hover - HoverTool object if indicated, eg use as hover.renderers = [glyphs]
        - tap - TapTool object if indicated, eg use as <column-data-source>.selected.on_change('indices', callback))
    - methods
        - reset_map_view
        - reset_interface
    """
    def __init__(self, width=700, height=300, map_vendor='OSM', size=12, hover=False, tap=False,
            path_color='darkblue', position_color='lime', events_color='darkorange', events_selected_color='red'):

        # standard tools
        pan, wheel, reset, save = PanTool(), WheelZoomTool(), ResetTool(), SaveTool()
        tools = [pan, wheel, reset, save]

        # hover and tap tools
        self.hover = hover
        self.tap = tap
        hover_tool = HoverTool() if self.hover else None
        tap_tool = TapTool() if self.tap else None
        if self.hover:
            tools.append(hover_tool)
            self.hover = hover_tool
        if self.tap:
            tools.append(tap_tool)
            self.tap = tap_tool

        # interactive map as a figure object
        self.fig = figure(width=width, height=height, tools=tools, toolbar_location='right')
        self.fig.add_tile(tile_source=get_provider(getattr(Vendors, map_vendor)))
        self.fig.toolbar.logo = None
        self.fig.x_range = DataRange1d()
        self.fig.y_range = DataRange1d()
        self.fig.toolbar.active_drag = pan
        self.fig.toolbar.active_scroll = wheel
        self.fig.toolbar.active_inspect = hover_tool
        self.fig.toolbar.active_tap = tap_tool
        self.fig.outline_line_width = 2
        self.fig.xaxis.visible = False
        self.fig.yaxis.visible = False
        self.fig.grid.visible = False
        largefonts(self.fig, size)

        # path data source and glyphs
        self.path = ColumnDataSource()
        self.path_glyphs = (
            self.fig.circle('lon', 'lat', source=self.path, size=8, color=path_color),
            self.fig.line('lon', 'lat', source=self.path, line_width=2, color=path_color))

        # position data source and glyph
        self.position = ColumnDataSource()
        self.position_glyph = self.fig.asterisk('lon', 'lat', source=self.position, size=12, color=position_color, line_color=position_color, line_width=2)

        # events data source and glyph
        self.events = ColumnDataSource()
        self.events_glyph = self.fig.circle('lon', 'lat', source=self.events, size=8, color=events_color,
            # visual properties for selected
            selection_color=events_selected_color,
            # visual properties for non-selected
            nonselection_fill_alpha=1,
            nonselection_fill_color=events_color)

        # reset interface
        self.reset_interface()

    def reset_map_view(self, lon0, lon1, lat0, lat1):
        """
        reset map view based on coords as arguments
        """
        assert lon0 < lon1
        assert lat0 < lat1
        self.lon0 = lon0
        self.lon1 = lon1
        self.lat0 = lat0
        self.lat1 = lat1

        # convert, update coords to maintain aspect ratio
        lon0, lat0 = gps_to_webm(lon=-110, lat=25)
        lon1, lat1 = gps_to_webm(lon=-70, lat=50)
        lon0, lon1, lat0, lat1 = get_map_ranges(lon0, lon1, lat0, lat1, aspect=self.fig.width / self.fig.height)

        # update coords on map
        self.fig.x_range.start = lon0
        self.fig.x_range.end = lon1
        self.fig.y_range.start = lat0
        self.fig.y_range.end = lat1

    def reset_interface(self):
        """
        reset data sources, map view, hover and tap tools
        """

        # data sources
        self.path.data = {'lon': np.array([]), 'lat': np.array([])}
        self.position.data = {'lon': np.array([]), 'lat': np.array([])}
        self.events.data = {'lon': np.array([]), 'lat': np.array([])}

        # map view
        if hasattr(self, 'lon0'):
            self.reset_map_view(self.lon0, self.lon1, self.lat0, self.lat1)

        # hover and tap tools
        if self.hover:
            self.hover.tooltips = []
            self.hover.formatters = {}
            self.hover.renderers = []
        if self.tap:
            self.tap.renderers = []

class LearningCurveInterface:
    """
    interface for figure object representing a ML learning curve
    - arguments to init
        - width, height, xlabel, ylabel, title, size, legend_location - figure formatting
        - train_color / test_color / alpha - formatting
        - hover - include hover tool - not assigned any attributes, eg renderers, tooltips, formatters
    - attributes created on init
        - fig - figure object
        - train - ColumnDataSource objects for train data
        - test - ColumnDataSource objects for test data
        - legend - legend object
    - methods
        - reset_interface
    """
    def __init__(self, width=700, height=300, xlabel='', ylabel='', title='', size=12, legend_location='top_right',
            train_color='darkblue', test_color='darkorange', alpha=0.4, hover=False):

        # standard tools
        pan, wheel, reset, save = PanTool(dimensions='height'), WheelZoomTool(dimensions='height'), ResetTool(), SaveTool()
        tools = [pan, wheel, reset, save]

        # hover tool
        self.hover = hover
        hover_tool = HoverTool() if self.hover else None
        if self.hover:
            tools.append(hover_tool)
            self.hover = hover_tool

        # figure object
        self.fig = figure(width=width, height=height, tools=tools, toolbar_location='right', x_axis_type='log')
        self.fig.toolbar.logo = None
        self.fig.x_range = Range1d(start=1e-3, end=1)
        self.fig.y_range = Range1d(start=0, end=1)
        self.fig.toolbar.active_drag = pan
        self.fig.toolbar.active_scroll = wheel
        self.fig.toolbar.active_inspect = hover_tool

        # ColumnDataSource objects
        self.train = ColumnDataSource()
        self.test = ColumnDataSource()

        # train glyphs
        train_glyphs = [
            self.fig.circle('x', 'mean', source=self.train, size=6, color=train_color),
            self.fig.line('x', 'mean', source=self.train, line_width=1, color=train_color)]
        self.fig.add_layout(Band(base='x', lower='min', upper='max', source=self.train, fill_color=train_color, fill_alpha=alpha))

        # test glyphs
        test_glyphs = [
            self.fig.circle('x', 'mean', source=self.test, size=6, color=test_color),
            self.fig.line('x', 'mean', source=self.test, line_width=1, color=test_color)]
        self.fig.add_layout(Band(base='x', lower='min', upper='max', source=self.test, fill_color=test_color, fill_alpha=alpha))

        # legend
        self.legend = Legend(location=legend_location, items=[
            LegendItem(label='train data', renderers=train_glyphs),
            LegendItem(label='test data', renderers=test_glyphs)])
        self.fig.add_layout(self.legend, 'center')

        # format axes and reset interface
        format_axes(self.fig, xlabel, ylabel, title, size)
        self.reset_interface()

    def reset_interface(self):
        """
        reset data-sources, legend, hover-tool
        """

        # data-sources
        self.train.data = {'x': np.array([]), 'mean': np.array([]), 'min': np.array([]), 'max': np.array([])}
        self.test.data = {'x': np.array([]), 'mean': np.array([]), 'min': np.array([]), 'max': np.array([])}

        # legend
        self.legend.visible = False

        # hover-tool
        if self.hover:
            self.hover.tooltips = []
            self.hover.formatters = {}
            self.hover.renderers = []

class HorizontalBarChartInterface:
    """
    interface for bar chart
    - arguments to init
        - width, height, xlabel, ylabel, title, size - figure formatting
        - bar_color, alpha - bar formatting
    - attributes created on init
        - fig - figure object
        - data - ColumnDataSource
    - methods
        - reset_interface
    """
    def __init__(self, width=700, height=300, xlabel='', ylabel='', title='', size=12, bar_color='darkblue', alpha=0.8):

        # standard tools
        pan, wheel, reset, save = PanTool(dimensions='height'), WheelZoomTool(dimensions='height'), ResetTool(), SaveTool()
        tools = [pan, wheel, reset, save]

        # figure object
        self.fig = figure(width=width, height=height, tools=tools, toolbar_location='right')
        self.fig.toolbar.logo = None
        self.fig.x_range = Range1d()
        self.fig.y_range = Range1d()
        self.fig.yaxis.ticker = FixedTicker()
        self.fig.toolbar.active_drag = pan
        self.fig.toolbar.active_scroll = wheel
        format_axes(self.fig, xlabel, ylabel, title, size)

        # ColumnDataSource object
        self.data = ColumnDataSource()

        # horizontal bar glyph
        self.fig.hbar('y', 'height', 'right', source=self.data, line_color=bar_color, fill_color=bar_color, fill_alpha=alpha)

        # reset interface
        self.reset_interface()

    def reset_interface(self):
        """
        reset data source and yaxis ticks
        """
        self.data.data = {'y': np.array([]), 'height': np.array([]), 'right': np.array([])}
        self.fig.yaxis.ticker.ticks = []

def get_map_ranges(lon0, lon1, lat0, lat1, aspect, border=0.05):
    """
    return ranges for map view that preserves 'aspect' ratio and includes all of lon0 to lon1, lat0 to lat1
    - 'border' represents the fraction of the limiting span (latitude or longitude) to include beyond the span
    - returns x0, x1, y0, y1 representing x_range start/end (longitude) and y_range start/end (latitude)
    - range objects on map should be Range1d objects, not DataRange1d objects to avoid auto-scaling
    """

    # data ranges
    dlon = lon1 - lon0
    dlat = lat1 - lat0

    # map view set by longitude span
    if dlon > aspect * dlat:
        x0 = lon0 - border * dlon
        x1 = lon1 + border * dlon
        dlon_view = dlon + 2 * border * dlon
        dlat_view = dlon_view / aspect
        assert dlat_view > dlat
        margin = (dlat_view - dlat) / 2
        y0 = lat0 - margin
        y1 = lat1 + margin

    # map view set by latitude span
    else:
        y0 = lat0 - border * dlat
        y1 = lat1 + border * dlat
        dlat_view = dlat + 2 * border * dlat
        dlon_view = dlat_view * aspect
        assert dlon_view >= dlon
        margin = (dlon_view - dlon) / 2
        x0 = lon0 - margin
        x1 = lon1 + margin

    return x0, x1, y0, y1

def link_axes(figs, axis='xy'):
    """
    link axes in list of figure objects, either 'x', 'y', or 'xy'
    """
    assert isinstance(figs, list)
    for fig in figs[1:]:
        if 'x' in axis:
            fig.x_range = figs[0].x_range
        if 'y' in axis:
            fig.y_range = figs[0].y_range

def format_axes(fig, xlabel='', ylabel='', title='', size=12):
    """
    update axis labels, title, and font sizes for a figure object
    """

    # axis labels
    fig.xaxis.axis_label = xlabel
    fig.yaxis.axis_label = ylabel
    fig.title.text = title

    # font sizes
    largefonts(fig, size)

def largefonts(fig, size=12):
    """
    update font sizes for a figure object
    """

    # font sizes based on 'size' argument
    s1 = '{}pt'.format(size - 1)
    s2 = '{}pt'.format(size)
    s3 = '{}pt'.format(size + 1)

    # title
    fig.title.text_font_size = s3

    # x-axis
    fig.xaxis.axis_label_text_font_size = s2
    fig.xaxis.major_label_text_font_size = s1

    # y-axis
    fig.yaxis.axis_label_text_font_size = s2
    fig.yaxis.major_label_text_font_size = s1

    # legend
    if fig.legend:
        fig.legend.label_text_font_size = s1
        fig.legend.border_line_width = 1
        fig.legend.border_line_color = 'grey'
        fig.legend.border_line_alpha = 0.6

def str_axis_labels(axis, labels):
    """
    set string labels on axis with FuncTickFormatter and custom JS
    - the 'ticker' attribute of axis must be a 'FixedTicker' with same num of ticks as items in 'labels'
    """

    # validate
    assert isinstance(axis.ticker, FixedTicker)
    assert isinstance(axis.ticker.ticks, list)
    assert isinstance(labels, np.ndarray)
    assert len(axis.ticker.ticks) == labels.size

    # create the FuncTickFormatter object
    label_dict = {a: b for (a, b) in zip(range(labels.size), labels)}
    axis.formatter = FuncTickFormatter(code=f"""var labels = {label_dict};return labels[tick];""")

def get_glyphs(fig):
    """
    return list of GlyphRenderer objects
    """
    return [x for x in fig.renderers if isinstance(x, GlyphRenderer)]

def autoload_static(model, script_path):
    """
    simplifies usage of bokeh.embed.autoload_static
    """

    from bokeh.embed import autoload_static
    from bokeh.resources import CDN

    script, tag = autoload_static(model=model, resources=CDN, script_path=script_path)
    with open(script_path, 'w') as fid:
        fid.write(script)

    return tag
