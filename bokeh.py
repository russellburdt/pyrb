
"""
common bokeh utils
"""

import numpy as np
from bokeh import palettes
from bokeh.plotting import figure
from bokeh.tile_providers import Vendors, get_provider
from bokeh.models import ColumnDataSource, DataRange1d, Range1d, GlyphRenderer, LabelSet, Legend, LegendItem, Band, Patches
from bokeh.models import PanTool, BoxZoomTool, WheelZoomTool, ResetTool, SaveTool, TapTool, HoverTool, CrosshairTool, RedoTool, UndoTool
from bokeh.models import Arrow, VeeHead, LabelSet
from bokeh.models import FuncTickFormatter, FixedTicker, LogScale, DatetimeTickFormatter
from pyrb.processing import gps_to_webm
from ipdb import set_trace

class MultiLineInterface:
    """
    interface for figure object with following glyphs
    - n circle/line glyphs, style set by palette and line_width args
    - x/line segment glyphs, fixed style
    - x position glyph, fixed style
    other configuration includes
    - pan/wheel/box/reset/save tools
    - pan/wheel tools according to dimensions arg
    - optional cross(always vertical)/hover/tap tools
    - x_range/y_range as Range1d objects according to manual_xlim/manual_ylim args
    - datetime as True sets xaxis.formatter as DatetimeTickFormatter
    - width/height/xlabel/ylabel/title/size
    - legend object initialized at legend_location arg, legend items must be added manually eg
      self.legend.items.append(LegendItem(label=..., renderers=self.renderers[...]))
    attributes created on init
    - fig - figure object
    - data_sources - one for each of n circle/line glyphs
    - segments/position - for other glyphs
    - legend/renderers - legend object and renderers for each of n circle/line glyphs
    methods
    - reset_interface
    """
    def __init__(self, width=700, height=300, xlabel='', ylabel='', title='', size=12, legend_location='bottom_right',
        hover=False, tap=False, cross=False, dimensions='both', n=20, palette='Category20_20', circle=True, line=True, line_width=1,
        manual_xlim=False, manual_ylim=False, datetime=False, box_dimensions='both'):

        # standard tools
        pan, wheel = PanTool(dimensions=dimensions), WheelZoomTool(dimensions=dimensions)
        box, reset, save = BoxZoomTool(dimensions=box_dimensions), ResetTool(), SaveTool()
        tools = [pan, wheel, box, reset, save]

        # crosshair tool
        self.cross = cross
        cross_tool = CrosshairTool(dimensions='height', line_width=3, line_color='black') if self.cross else None
        if self.cross:
            tools.append(cross_tool)
            self.cross = cross_tool

        # hover tool
        self.hover = hover
        hover_tool = HoverTool() if self.hover else None
        if self.hover:
            tools.append(hover_tool)
            self.hover = hover_tool

        # tap tool
        self.tap = tap
        tap_tool = TapTool() if self.tap else None
        if self.tap:
            tools.append(tap_tool)
            self.tap = tap_tool

        # figure object
        self.fig = figure(width=width, height=height, tools=tools, toolbar_location='right')
        self.fig.toolbar.logo = None
        if manual_xlim:
            self.fig.x_range = Range1d()
        if manual_ylim:
            self.fig.y_range = Range1d()
        if datetime:
            self.fig.xaxis.formatter = DatetimeTickFormatter()
        self.fig.toolbar.active_drag = pan
        self.fig.toolbar.active_scroll = wheel
        self.fig.toolbar.active_inspect = hover_tool if self.hover else cross_tool if self.cross else None
        self.fig.toolbar.active_tap = tap_tool

        # ColumnDataSource objects
        self.n = n
        self.data_sources = tuple([ColumnDataSource() for _ in range(n)])

        # circle and line glyphs for each data source
        assert circle or line
        colors = palettes.linear_palette(getattr(palettes, palette), n)
        for src, color in zip(self.data_sources, colors):
            if circle:
                self.fig.circle('x', 'y', source=src, size=6, color=color)
            if line:
                self.fig.line('x', 'y', source=src, line_width=line_width, color=color)
        if np.logical_xor(circle, line):
            self.renderers = np.expand_dims(np.array(self.fig.renderers), axis=1)
        else:
            self.renderers = np.array(self.fig.renderers).reshape(n, 2)

        # additional segment data source and glyphs (fixed formatting)
        self.segments = ColumnDataSource()
        self.fig.x('x', 'y', source=self.segments, size=3, color='violet', alpha=0.6)
        self.fig.line('x', 'y', source=self.segments, line_width=1, color='violet', alpha=0.6)

        # additional position data source and glyphs (fixed formatting)
        self.position = ColumnDataSource()
        self.fig.x('x', 'y', source=self.position, size=12, line_width=3, color='red')

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
        self.segments.data = {'x': np.array([]), 'y': np.array([])}
        self.position.data = {'x': np.array([]), 'y': np.array([])}
        self.legend.items = []
        if self.hover:
            self.hover.tooltips = []
            self.hover.formatters = {}
            self.hover.renderers = 'auto'
        if self.tap:
            self.tap.renderers = []

class MapInterface:
    """
    interface for map object supporting following glyphs
        - path - darkblue circle/line glyphs
        - segments - violet x/line glyphs
        - nodes - red x/patches glyphs
        - links - darkorange circle/line glyphs
        - position - lime asterik glyph
        - events - circle glyph, darkorange (nonselected) and red (selected)
    - arguments to init
        - width, height, map_vendor, size - map formatting
        - hover - include hover tool - not assigned any attributes, eg renderers, tooltips, formatters
        - tap - include tap tool - no tap tool callback is created
        - lon0, lon1, lat0, lat1 - default view
    - attributes created on init
        - fig - map as a bokeh figure object
        - ColumnDataSource objects for all glyphs
        - hover - HoverTool object if indicated, eg use as hover.renderers = [glyphs]
        - tap - TapTool object if indicated, eg use as <column-data-source>.selected.on_change('indices', callback))
    - methods
        - reset_map_view
        - reset_interface
    """
    def __init__(self, width=700, height=300, map_vendor='OSM', size=12, hover=False, tap=False,
        lon0=-110, lon1=-70, lat0=25, lat1=50):

        # standard tools
        pan, wheel, box, reset, redo, undo = PanTool(), WheelZoomTool(), BoxZoomTool(), ResetTool(), RedoTool(), UndoTool()
        tools = [pan, wheel, box, reset, redo, undo]

        # hover and tap tools
        if hover:
            self.hover = HoverTool()
            self.hover.tooltips = []
            self.hover.formatters = {}
            self.hover.renderers = []
            tools.append(self.hover)
        else:
            self.hover = None
        if tap:
            self.tap = TapTool()
            self.tap.renderers = []
            tools.append(self.tap)
        else:
            self.tap = None

        # interactive map as a figure object
        self.fig = figure(width=width, height=height, tools=tools, toolbar_location='right')
        self.fig.add_tile(tile_source=get_provider(getattr(Vendors, map_vendor)))
        self.fig.toolbar.logo = None
        self.fig.x_range = DataRange1d()
        self.fig.y_range = DataRange1d()
        self.fig.toolbar.active_drag = pan
        self.fig.toolbar.active_scroll = wheel
        self.fig.toolbar.active_inspect = self.hover
        self.fig.toolbar.active_tap = self.tap
        self.fig.outline_line_width = 2
        self.fig.xaxis.visible = False
        self.fig.yaxis.visible = False
        self.fig.grid.visible = False
        self.fig.x_range = Range1d()
        self.fig.y_range = Range1d()
        largefonts(self.fig, size)

        # path data source and glyphs
        self.path = ColumnDataSource()
        self.fig.circle('lon', 'lat', source=self.path, size=6, color='darkblue', name='path')
        self.fig.line('lon', 'lat', source=self.path, line_width=2, color='darkblue', name='path')

        # segments data source and glyphs
        self.segments = ColumnDataSource()
        self.fig.x('lon', 'lat', source=self.segments, size=3, color='violet', alpha=0.6, name='segments')
        self.fig.line('lon', 'lat', source=self.segments, line_width=1, color='violet', alpha=0.6, name='segments')

        # nodes data source and glyphs
        self.nodes = ColumnDataSource()
        self.fig.x('lon_centroids', 'lat_centroids', source=self.nodes, size=8, color='red', line_width=2, name='nodes')
        self.fig.patches('lon', 'lat', source=self.nodes, line_color='red', line_width=4, fill_color='red', fill_alpha=0.2, name='nodes')

        # links data source and glyphs
        self.links = ColumnDataSource()
        self.fig.circle('lon', 'lat', source=self.links, size=4, color='darkorange', name='links')
        self.fig.line('lon', 'lat', source=self.links, line_width=1, color='darkorange', name='links')

        # position data source and glyph
        self.position = ColumnDataSource()
        self.fig.asterisk('lon', 'lat', source=self.position, size=10, color='lime', line_color='lime', line_width=2, name='position')

        # events data source and glyph
        self.events = ColumnDataSource()
        self.fig.circle('lon', 'lat', source=self.events, size=8, color='darkorange',
            # visual properties for selected
            selection_color='red',
            # visual properties for non-selected
            nonselection_fill_alpha=1,
            nonselection_fill_color='darkorange',
            name='events')

        # default view
        assert lon0 < lon1
        assert lat0 < lat1
        self.lon0 = lon0
        self.lon1 = lon1
        self.lat0 = lat0
        self.lat1 = lat1

        # reset map view and interface
        self.reset_interface()
        self.reset_map_view()

    def reset_map_view(self, lon0=None, lon1=None, lat0=None, lat1=None, convert=True):
        """
        reset map view
        """

        # use default view if any coords not provided as args
        if (lon0 is None) or (lon1 is None) or (lat0 is None) or (lat1 is None):
            lon0, lon1 = self.lon0, self.lon1
            lat0, lat1 = self.lat0, self.lat1

        # validate and convert
        assert lon0 < lon1
        assert lat0 < lat1
        if convert:
            lon0, lat0 = gps_to_webm(lon=lon0, lat=lat0)
            lon1, lat1 = gps_to_webm(lon=lon1, lat=lat1)

        # update coords to maintain aspect ratio
        lon0, lon1, lat0, lat1 = get_map_ranges(lon0, lon1, lat0, lat1, aspect=self.fig.width / self.fig.height)

        # update coords on map
        self.fig.x_range.start = lon0
        self.fig.x_range.end = lon1
        self.fig.y_range.start = lat0
        self.fig.y_range.end = lat1

    def reset_interface(self):
        """
        reset data sources
        """
        self.path.data = {'lon': np.array([]), 'lat': np.array([])}
        self.segments.data = {'lon': np.array([]), 'lat': np.array([])}
        self.position.data = {'lon': np.array([]), 'lat': np.array([])}
        self.events.data = {'lon': np.array([]), 'lat': np.array([])}
        self.nodes.data = {'lon': [], 'lat': [], 'lon_centroids': np.array([]), 'lat_centroids': np.array([])}
        self.links.data = {'lon': np.array([]), 'lat': np.array([])}

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
        - include_nums - numbers as labels at end of horizontal bars
    - attributes created on init
        - fig - figure object
        - data - ColumnDataSource
    - methods
        - reset_interface
    """
    def __init__(self, width=700, height=300, xlabel='', ylabel='', title='', size=12,
        bar_color='darkblue', alpha=0.8, include_nums=False, pan_dimensions='both'):

        # standard tools
        pan, wheel, reset, save = PanTool(dimensions=pan_dimensions), WheelZoomTool(dimensions='height'), ResetTool(), SaveTool()
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

        # numbers at end of horizontal bars
        # https://stackoverflow.com/questions/39401481/how-to-add-data-labels-to-a-bar-chart-in-bokeh
        self.include_nums = include_nums
        if self.include_nums:
            self.label_source = ColumnDataSource()
            self.labels = LabelSet(x='x', y='y', text='text', source=self.label_source, render_mode='canvas', level='glyph',
                text_font_size='14px', text_align='left', text_baseline='middle')
            self.fig.add_layout(self.labels)

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
        if self.include_nums:
            self.label_source.data = {'x': np.array([]), 'y': np.array([]), 'text': np.array([])}
        self.fig.yaxis.ticker.ticks = []

class MetricDistributionInterface:
    """
    interface for single distribution
    - arguments to init
        - width, height, xlabel, ylabel, title, size - figure formatting
        - bar_color, alpha - bar formatting
        - logscale - log-scale for y-axis
    - attributes created on init
        - fig - figure object
        - data - ColumnDataSource
    - methods
        - reset_interface
    """
    def __init__(self, width=700, height=300, xlabel='', ylabel='', title='', size=12, cross=False,
            bar_color='darkblue', alpha=0.8, logscale=False, fixed_ticker=False, dimensions='height'):

        # standard tools
        pan, wheel, box, reset, save = PanTool(dimensions=dimensions), WheelZoomTool(dimensions=dimensions), BoxZoomTool(), ResetTool(), SaveTool()
        tools = [pan, wheel, box, reset, save]

        # crosshair tool
        self.cross = cross
        cross_tool = CrosshairTool(dimensions='height', line_width=3, line_color='black') if self.cross else None
        if self.cross:
            tools.append(cross_tool)
            self.cross = cross_tool

        # figure object
        self.fig = figure(width=width, height=height, tools=tools, toolbar_location='right')
        self.fig.toolbar.logo = None
        self.fig.x_range = Range1d()
        if fixed_ticker:
            self.fig.xaxis.ticker = FixedTicker()
        self.fig.y_range = Range1d()
        self.fig.toolbar.active_drag = pan
        self.fig.toolbar.active_scroll = wheel
        if logscale:
            # not currently supported by bokeh, see
            # https://github.com/bokeh/bokeh/issues/6536
            # self.fig.y_scale = LogScale()
            pass
        format_axes(self.fig, xlabel, ylabel, title, size)

        # ColumnDataSource object
        self.data = ColumnDataSource()

        # vertical bar glyph
        self.fig.vbar(x='x', width='width', top='top', source=self.data, line_color=bar_color, fill_color=bar_color, fill_alpha=alpha)

        # reset interface
        self.reset_interface()

    def reset_interface(self):
        """
        reset data source
        """
        self.data.data = {'x': np.array([]), 'width': np.array([]), 'top': np.array([])}

class ShapValuesWaterfallInterface:
    """
    interface for a shap values waterfall chart
    """
    def __init__(self, width=400, height=600, xlabel='', ylabel='', title='', size=12):

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

        # ColumnDataSource objects
        self.base = ColumnDataSource()
        self.positive = ColumnDataSource()
        self.negative = ColumnDataSource()

        # line glyph and arrow glyphs
        self.fig.line('x', 'y', source=self.base, line_width=4, color='black', line_dash='dashed')
        self.fig.add_layout(Arrow(end=VeeHead(size=8, line_color='green', fill_color='green', line_width=2),
            line_color='green', line_width=2, source=self.positive, x_start='x0', y_start='y0', x_end='x1', y_end='y1'))
        self.fig.add_layout(Arrow(end=VeeHead(size=8, line_color='red', fill_color='red', line_width=2),
            line_color='red', line_width=2, source=self.negative, x_start='x0', y_start='y0', x_end='x1', y_end='y1'))

        # numeric labels at ends of each arrow
        self.fig.add_layout(LabelSet(x='x1', y='y1', text='labels', source=self.positive,
            text_align='left', text_baseline='middle', text_font_size='9pt', text_color='green'))
        self.fig.add_layout(LabelSet(x='x1', y='y1', text='labels', source=self.negative,
            text_align='right', text_baseline='middle', text_font_size='9pt', text_color='red'))

        # reset interface
        self.reset_interface()

    def reset_interface(self):
        """
        reset data sources
        """
        self.base.data = {'x': np.array([]), 'y': np.array([])}
        self.positive.data = {'x0': np.array([]), 'x1': np.array([]), 'y0': np.array([]), 'y1': np.array([]), 'labels': np.array([])}
        self.negative.data = {'x0': np.array([]), 'x1': np.array([]), 'y0': np.array([]), 'y1': np.array([]), 'labels': np.array([])}
        self.fig.yaxis.ticker.ticks = []

def console_str_objects(width):
    """
    return strings to open and close a div object representing a 'console' in a bokeh application
    """
    c0 = f"""<div style="background-color:#072A49;width:{width}px;color:white;border:0;font-family':Helvetica,arial,sans-serif">"""
    c1 = """</div>"""
    return c0, c1

def update_fig_range1d(range1d, data):
    """
    update the range1d object according to limits in data, handle corner cases
    """

    # validate
    assert isinstance(range1d, Range1d)
    assert isinstance(data, np.ndarray)
    assert data.size > 0
    data = data[~np.isnan(data)]
    assert data.size > 0

    # nominal case
    if np.unique(data).size > 1:
        range1d.start = data.min()
        range1d.end = data.max()
        return

    # corner cases
    if np.unique(data).size == 1:
        value = np.unique(data)[0]
        if value == 0:
            range1d.start = -1
            range1d.end = 1
        elif value > 0:
            range1d.start = 0
            range1d.end = value
        elif value < 0:
            range1d.start = value
            range1d.end = 0
        return

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
    s1 = '{}pt'.format(size - 2)
    s2 = '{}pt'.format(size)
    s3 = '{}pt'.format(size + 2)

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
