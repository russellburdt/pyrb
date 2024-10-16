"""
common bokeh utils
"""

import os
import numpy as np
import xyzservices
from bokeh import palettes
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, DataRange1d, Range1d, GlyphRenderer, LabelSet, Legend, LegendItem, Band, Patches, WMTSTileSource
from bokeh.models import PanTool, BoxZoomTool, WheelZoomTool, ResetTool, SaveTool, TapTool, HoverTool, CrosshairTool, RedoTool, UndoTool
from bokeh.models import Arrow, VeeHead, LabelSet, CustomJSTickFormatter, FixedTicker, LogScale, DatetimeTickFormatter, TileRenderer, CustomJS
from bokeh.models import HTMLTemplateFormatter, StringFormatter, DataTable, TableColumn, Div, Button, LabelSet
from pyrb.processing import gps_to_webm
from ipdb import set_trace


class FigureInterface:
    """
    interface for figure object, supports one or more circle/line glyphs, one position glyph, one highlight glyph
    - arguments to init
        - num - number of circle/line glyphs
        - colors - numpy array of hex colors, same size as num
        - circle, line - boolean to include circle and/or line glyphs, at least one must be True
        - circle_size, line_width - formatting for circle/line glyphs
        - width, height, size - formatting
        - hover - boolean to include hover tool - attributes initialized as
            - renderers=[], tooltips=[], formatters={}
        - tap - boolean to include tap tool - attributes initialized as
            - renderers=[]
            - associate callback - <ColumnDataSource>.selected.on_change('indices', callback)
        - crosshair - boolean to include vertical crosshair tool - no attributes, associate callbacks
            - <figure>.on_event(MouseMove, move_callback)
            - <figure>.on_event(MouseLeave, leave_callback)
        - title - text for title
        - legend_location, legend_layout_location, legend_title - legend position and title
        - legend_fig - another bokeh figure object where figure is added as layout
        - position_marker, position_size, position_color, position_line_width - position glyph formatting
        - manual_xlim, manual_ylim - boolean to use Range1d for x_range/y_range
            (keeping DataRange1d works best with ResetTool, as the default view gets updated with ColumnDataSource objects)
        - datetime - boolean to use datetime x-axis formatter
    - attributes created on init
        - fig - map as a bokeh figure object
        - data - tuple of ColumnDataSource objects for all circle/line glyphs
        - position - ColumnDataSource object for position glyph
        - highlight - ColumnDataSource object for highlight glyph
        - highlight_glyph - renderer for highlight glyph
        - hover, tap, cross - tool objects
    - methods
        - reset_interface
    """

    def __init__(self, num=1, colors=np.array(['#0000ff']), circle=True, line=True, line_width=1, circle_size=6,
        width=1200, height=300, size=12, hover=False, tap=False, cross=False, standard_tools=True, title=None, xlabel=None, ylabel=None,
        legend_location='top_left', legend_layout_location='center', legend_title=None, legend_fig=None,
        position_marker='asterisk', position_size=20, position_color='red', position_line_width=3, line_alpha=1.0,
        manual_xlim=False, manual_ylim=False, datetime=False):

        # standard tools
        if standard_tools:
            pan, wheel, box, save  = PanTool(), WheelZoomTool(), BoxZoomTool(), SaveTool()
            tools = [pan, wheel, box, save]
            self.pan = pan
            self.wheel = wheel
            self.box = box
        else:
            tools = [SaveTool()]

        # hover tool
        if hover:
            self.hover = HoverTool()
            self.hover.tooltips = []
            self.hover.formatters = {}
            tools.append(self.hover)
        else:
            self.hover = None

        # tap tool
        if tap:
            self.tap = TapTool()
            tools.append(self.tap)
        else:
            self.tap = None

        # cross tool
        if cross:
            self.cross = CrosshairTool(dimensions='height', line_width=3, line_color='black')
            tools.append(self.cross)
        else:
            self.cross = None

        # interactive map as a figure object
        self.fig = figure(width=width, height=height, tools=tools, toolbar_location='right')
        self.fig.toolbar.logo = None
        if standard_tools:
            self.fig.toolbar.active_drag = pan
            self.fig.toolbar.active_scroll = wheel
        if title is not None:
            self.fig.title.text = title
            self.fig.title.align = 'center'
        if xlabel is not None:
            self.fig.xaxis.axis_label = xlabel
        if ylabel is not None:
            self.fig.yaxis.axis_label = ylabel

        # x/y axis options
        if manual_xlim:
            self.fig.x_range = Range1d()
        if manual_ylim:
            self.fig.y_range = Range1d()
        if datetime:
            self.fig.xaxis.formatter = DatetimeTickFormatter()

        # ColumnDataSource objects
        self.num = num
        self.data = tuple([ColumnDataSource() for _ in range(num)])

        # circle and line glyphs for each data source
        assert circle or line
        assert colors.size == num
        for src, color in zip(self.data, colors):
            if circle:
                self.fig.scatter('x', 'y', marker='circle', source=src, size=circle_size, color=color, name='circle',
                    # visual properties for selected
                    selection_color='red',
                    # visual properties for non-selected
                    nonselection_fill_alpha=1,
                    nonselection_fill_color=color)
            if line:
                self.fig.line('x', 'y', source=src, line_width=line_width, color=color, line_alpha=line_alpha)
        if np.logical_xor(circle, line):
            self.renderers = np.expand_dims(np.array(self.fig.renderers), axis=1)
        else:
            self.renderers = np.array(self.fig.renderers).reshape(num, 2)

        # position data source and glyph
        self.position = ColumnDataSource()
        self.fig.scatter('x', 'y', source=self.position, marker=position_marker, size=position_size, color=position_color, line_color=position_color, line_width=position_line_width, name='position')

        # highlight data source and glyph
        self.highlight = ColumnDataSource()
        self.highlight_glyph = self.fig.line('x', 'y', source=self.highlight, line_width=12, line_color='black', alpha=0.4)

        # legend object
        self.legend = Legend(location=legend_location, glyph_width=20, glyph_height=20)
        if legend_fig is None:
            self.fig.add_layout(self.legend, legend_layout_location)
            if legend_title is not None:
                self.fig.legend.title = legend_title
                self.fig.legend.title_text_font_style = 'bold'
                self.fig.legend.title_text_font_size = f'{size + 1}px'
        else:
            legend_fig.add_layout(self.legend, legend_layout_location)
            legend_fig.legend.title = legend_title
            legend_fig.legend.title_text_font_style = 'bold'
            legend_fig.legend.title_text_font_size = f'{size + 2}px'
            legend_fig.renderers = self.fig.renderers
            largefonts(legend_fig, size)

        # reset map view and interface
        largefonts(self.fig, size)
        self.reset_interface()

    def reset_interface(self):
        """
        reset data sources
        """
        for x in range(self.num):
            self.data[x].data = {'x': np.array([]), 'y': np.array([])}
        self.position.data = {'x': np.array([]), 'y': np.array([])}
        self.highlight.data = {'x': np.array([]), 'y': np.array([])}
        self.legend.items = []
        if self.hover:
            self.hover.renderers = []
        if self.tap:
            self.tap.renderers = []

class MapInterface:
    """
    interface for map object, supports one or more circle/line glyphs, one position glyph, one label glyph, one highlight glyph
    - arguments to init
        - num - number of circle/line glyphs
        - colors - numpy array of hex colors, same size as num
        - circle, line - boolean to include circle and/or line glyphs, at least one must be True
        - circle_size, line_width - formatting for circle/line glyphs
        - width, height, map_vendor, size - map formatting
        - lon0, lon1, lat0, lat1 - default view
        - hover - boolean to include hover tool - attributes initialized as
            - renderers=[], tooltips=[], formatters={}
        - tap - boolean to include tap tool - attributes initialized as
            - renderers=[]
            - associate callback - <ColumnDataSource>.selected.on_change('indices', callback)
        - title - text for title
        - legend_location, legend_layout_location, legend_title - legend position and title
        - legend_fig - another bokeh figure object where figure is added as layout
        - position_marker, position_size, position_color, position_line_width - position glyph formatting
    - attributes created on init
        - fig - map as a bokeh figure object
        - data - tuple of ColumnDataSource objects for all circle/line glyphs
        - position - ColumnDataSource object for position glyph
        - labelset - LabelSet glyph
        - label - ColumnDataSource object for label
        - highlight - ColumnDataSource object for highlight glyph
        - hover, tap - tool objects
    - methods
        - reset_map_view
        - reset_interface
        - update_tile_source
    """

    def __init__(self, num=1, colors=np.array(['#0000ff']), circle=True, line=True, line_width=1, circle_size=6,
        width=1200, height=300, map_vendor='OSM', size=12, lon0=-110, lon1=-70, lat0=25, lat1=50, hover=False, tap=False,
        title='', legend_location='top_left', legend_layout_location='center', legend_title=None, legend_fig=None,
        position_marker='asterisk', position_size=16, position_color='black', position_line_width=3):

        # standard tools
        pan, wheel, box, reset, save = PanTool(), WheelZoomTool(), BoxZoomTool(), ResetTool(), SaveTool()
        tools = [pan, wheel, box, reset, save]

        # hover and tap tools
        if hover:
            self.hover = HoverTool()
            self.hover.tooltips = []
            self.hover.formatters = {}
            tools.append(self.hover)
        else:
            self.hover = None
        if tap:
            self.tap = TapTool()
            tools.append(self.tap)
        else:
            self.tap = None

        # interactive map as a figure object
        self.fig = figure(width=width, height=height, tools=tools, toolbar_location='right')
        self.fig.add_tile(tile_source=map_vendor)
        self.fig.toolbar.logo = None
        self.fig.toolbar.active_drag = pan
        self.fig.toolbar.active_scroll = wheel
        self.fig.outline_line_width = 2
        self.fig.xaxis.visible = False
        self.fig.yaxis.visible = False
        self.fig.grid.visible = False
        self.fig.x_range = Range1d()
        self.fig.y_range = Range1d()
        self.fig.title.text = title
        self.fig.title.align = 'center'

        # ColumnDataSource objects
        self.num = num
        self.data = tuple([ColumnDataSource() for _ in range(num)])

        # circle and line glyphs for each data source
        assert circle or line
        assert colors.size == num
        for src, color in zip(self.data, colors):
            if circle:
                self.fig.scatter('lon', 'lat', marker='circle', source=src, size=circle_size, color=color, name='circle',
                    # visual properties for selected
                    selection_color='red',
                    # visual properties for non-selected
                    nonselection_fill_alpha=1,
                    nonselection_fill_color=color)
            if line:
                self.fig.line('lon', 'lat', source=src, line_width=line_width, color=color)
        if np.logical_xor(circle, line):
            self.renderers = np.expand_dims(np.array(self.fig.renderers[1:]), axis=1)
        else:
            self.renderers = np.array(self.fig.renderers[1:]).reshape(num, 2)

        # position data source and glyph
        self.position = ColumnDataSource()
        self.fig.scatter('lon', 'lat', source=self.position, marker=position_marker, size=position_size, color=position_color, line_color=position_color, line_width=position_line_width, name='position')

        # label data source and glyph
        self.label = ColumnDataSource()
        self.labelset = LabelSet(x='lon', y='lat', text='text', x_offset=10, y_offset=0, x_units='data', y_units='data', source=self.label,
            text_align='left', text_baseline='middle', text_font_size=f'{size-2}px', border_line_color='black', border_line_width=1,
            text_line_height=1, text_font_style='bold', background_fill_color='white', background_fill_alpha=0.8)
        self.fig.add_layout(self.labelset)

        # highlight data source and glyph
        self.highlight = ColumnDataSource()
        self.fig.scatter('lon', 'lat', source=self.highlight, marker='x', size=position_size, color='black', line_color='black', line_width=8, alpha=0.6)
        self.fig.line('lon', 'lat', source=self.highlight, line_width=8, color='black', alpha=0.6)

        # default view
        assert lon0 < lon1
        assert lat0 < lat1
        self.lon0 = lon0
        self.lon1 = lon1
        self.lat0 = lat0
        self.lat1 = lat1

        # legend object
        self.legend = Legend(location=legend_location, glyph_width=20, glyph_height=20)
        if legend_fig is None:
            self.fig.add_layout(self.legend, legend_layout_location)
            if legend_title is not None:
                self.fig.legend.title = legend_title
                self.fig.legend.title_text_font_style = 'bold'
                self.fig.legend.title_text_font_size = f'{size + 2}px'
        else:
            legend_fig.add_layout(self.legend, legend_layout_location)
            legend_fig.legend.title = legend_title
            legend_fig.legend.title_text_font_style = 'bold'
            legend_fig.legend.title_text_font_size = f'{size + 2}px'
            legend_fig.renderers = self.fig.renderers[1:]
            largefonts(legend_fig, size)

        # reset map view and interface
        largefonts(self.fig, size)
        self.reset_interface()
        self.update_map_view()

    def update_map_view(self, lon0=None, lon1=None, lat0=None, lat1=None, convert=True):
        """
        update map view, use gps coords
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
        for x in range(self.num):
            self.data[x].data = {'lon': np.array([]), 'lat': np.array([])}
        self.position.data = {'lon': np.array([]), 'lat': np.array([])}
        self.label.data = {'lon': np.array([]), 'lat': np.array([]), 'text': np.array([])}
        self.highlight.data = {'lon': np.array([]), 'lat': np.array([])}
        self.legend.items = []
        if self.hover:
            self.hover.renderers = []
        if self.tap:
            self.tap.renderers = []

    def update_tile_source(self, provider='OSM'):
        assert provider in ['OSM', 'ESRI']
        tr = [x for x in self.fig.renderers if isinstance(x, TileRenderer)]
        assert len(tr) == 1
        tr = tr[0]
        name = 'openstreetmap_mapnik' if provider == 'OSM' else 'esri_worldimagery' if provider == 'ESRI' else None
        provider = xyzservices.providers.query_name(name)
        ts = WMTSTileSource(url=provider.build_url(), attribution=provider.html_attribution, min_zoom=provider.get('min_zoom', 0), max_zoom=provider.get('max_zoom', 30))
        tr.update(tile_source=ts)

class TableInterface:
    """
    interface for table object, supports a DataTable, Title, and export_to_csv objects
    - arguments to init
        - width, height, size - table size and formatting
        - config, tuple of lists, eg config = (['col1', width1], ['col2', width2], ...)
        - widths in config are relative (int or float) and get scaled to fill table width
        - title - boolean to include table title as a Div object
        - export_to_csv - boolean to include button object that exports table to csv
        - sortable - boolean (default is False)
    - attributes created on init
        - columns - TableColumn objects
        - data - ColumnDataSource object
        - table - DataTable object
        - title - Div object or None
        - download - Button object for export_to_csv or None
    - methods
        - update_table - update table based on provided dataframe
        - reset_interface - clear data in table
    """
    def __init__(self, config, width=800, height=200, size=12, title=None, export_to_csv=False, sortable=False):

        # validate config object
        assert isinstance(config, tuple)
        assert all([isinstance(x, list) and (len(x) == 3) and isinstance(x[0], str) and isinstance(x[1], (int, float)) for x in config])

        # scale widths in config to table width
        ts = sum([x[1] for x in config])
        scale = width / ts
        for rx in config:
            rx[1] = int(rx[1] * scale)
        ts = sum([x[1] for x in config])
        config[-1][1] += width - ts
        assert sum([x[1] for x in config]) == width

        # column objects for data-table
        self.columns = []
        for x in config:
            if x[2] is None:
                formatter = HTMLTemplateFormatter(template=f"""<div style="font-size:{size}px;text-align:left"><%=value%></div>""")
            else:
                formatter = x[2]
            self.columns.append(TableColumn(field=x[0], width=x[1], formatter=formatter,
                title=f"""<div style="font-size:{size}px;font-weight:bold;text-align:left">{x[0]}</div>"""))

        # data-table object
        self.data = ColumnDataSource()
        self.table = DataTable(width=width, height=height, source=self.data, columns=self.columns, index_position=None,
            sortable=sortable, reorderable=False, header_row=True)

        # title as a Div object
        if title is not None:
            self.title = Div(text=f"""<strong style="font-size:{size + 2}px;background-color:#baf1ef">&emsp;{title}&emsp;</strong>""")
        else:
            self.title = None

        # export to csv Button
        if export_to_csv:
            fn = os.path.join(os.getcwd(), 'download.js')
            assert os.path.isfile(fn)
            with open(fn, 'r') as fid:
                code = fid.read()
            self.download = Button(label='download to csv', button_type='success', width=100)
            self.download.js_on_event("button_click", CustomJS(args=dict(source=self.data), code=code))

    def update_table(self, df, others=False):
        """
        update table based on provided dataframe
        - others=True - columns from provided dataframe in ColumnDataSource
        - others=False - columns from DataTable in ColumnDataSource
        """
        fields = [x.field for x in self.columns]
        assert all([field in df.columns for field in fields])
        fs = [x.field for x in self.columns if isinstance(x.formatter, HTMLTemplateFormatter)]
        assert df[fs].isnull().values.sum() == 0

        if others:
            self.data.data = {field: df[field].tolist() for field in df.columns}
        else:
            self.data.data = {field: df[field].tolist() for field in fields}

    def reset_interface(self):
        """
        clear data in table
        """
        self.data.selected.indices = []
        self.data.data = {}

class SeriesInterface:
    """
    interface to display a pandas Series as a table object, supports a DataTable (configured as a Series), Title, and export_to_csv objects
    - arguments to init
        - formatter for values column, eg NumberFormatter(format=f'0,0.0')
        - width, height, size - table size and formatting
        - midpoint - relative location from left edge for column border
        - title - boolean to include table title as a Div object
        - export_to_csv - boolean to include button object that exports table to csv
    - attributes created on init
        - data - ColumnDataSource object
        - table - DataTable object
        - title - Div object or None
        - download - Button object for export_to_csv or None
    - methods
        - update_table - update table based on provided pandas Series
        - reset_interface - clear data in table
    """
    def __init__(self, formatter, width=800, height=200, size=12, midpoint=0.6, title=None, export_to_csv=False):

        # width for table columns
        w0 = int(midpoint * width)
        w1 = width - w0

        # column objects for data-table
        c0 = TableColumn(field='index', width=w0, title=None, formatter=
            HTMLTemplateFormatter(template=f"""<div style="font-size:{size}px;text-align:left"><%=value%></div>"""))
        c1 = TableColumn(field='values', width=w1, title=None, formatter=formatter)

        # data-table object
        self.data = ColumnDataSource()
        self.table = DataTable(width=width, height=height, source=self.data, columns=[c0, c1], index_position=None,
            sortable=False, reorderable=False, header_row=False)

        # title as a Div object
        if title is not None:
            self.title = Div(text=f"""<strong style="font-size:{size + 2}px;background-color:#baf1ef">&emsp;{title}&emsp;</strong>""")
        else:
            self.title = None

        # export to csv Button
        if export_to_csv:
            fn = os.path.join(os.getcwd(), 'download.js')
            assert os.path.isfile(fn)
            with open(fn, 'r') as fid:
                code = fid.read()
            self.download = Button(label='download to csv', button_type='success', width=100)
            self.download.js_on_event("button_click", CustomJS(args=dict(source=self.data), code=code))

    def update_table(self, ds):
        """
        update table based on provided dataframe
        """
        self.data.data = {'index': ds.index.values, 'values': ds.to_numpy()}

    def reset_interface(self):
        """
        clear data in table
        """
        self.data.selected.indices = []
        self.data.data = {}

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
        pan, wheel, save = PanTool(dimensions=pan_dimensions), WheelZoomTool(dimensions='height'), SaveTool()
        tools = [pan, wheel, save]

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
            self.labels = LabelSet(x='x', y='y', text='text', x_offset=2, y_offset=0, x_units='data', y_units='data', source=self.label_source,
                text_font_size=f'{size}px', text_font_style='bold', text_align='left', text_baseline='middle')
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
        fig.legend.border_line_width = 2
        fig.legend.border_line_color = 'grey'
        fig.legend.border_line_alpha = 0.6

def str_axis_labels(axis, labels):
    """
    set string labels on axis with CustomJSTickFormatter and custom JS
    - the 'ticker' attribute of axis must be a 'FixedTicker' with same num of ticks as items in 'labels'
    """

    # validate
    assert isinstance(axis.ticker, FixedTicker)
    assert isinstance(axis.ticker.ticks, list)
    assert isinstance(labels, np.ndarray)
    assert len(axis.ticker.ticks) == labels.size

    # create the CustomJSTickFormatter object
    label_dict = {a: b for (a, b) in zip(range(labels.size), labels)}
    axis.formatter = CustomJSTickFormatter(code=f"""var labels = {label_dict};return labels[tick];""")

def empty_figure(width=300, height=300):
    """
    return 'empty' bokeh figure object
    - useful to hold a legend outside of other figures
    """
    fig = figure(width=width, height=height, toolbar_location=None)
    fig.x_range = Range1d(start=-2, end=-1)
    fig.y_range = Range1d(start=-2, end=-1)
    fig.axis.visible = False
    fig.grid.visible = False
    fig.outline_line_color = None
    return fig

def console_str_objects(width):
    """
    return strings to open and close a div object representing a 'console' in a bokeh application
    """
    c0 = f"""<div style="background-color:#072A49;width:{width}px;color:white;border:0;font-family':Helvetica,arial,sans-serif">"""
    c1 = """</div>"""
    return c0, c1

class FrameInterface:
    """
    interface for figure object to render a static png image
    - may be blinking when image url source is updated
    - https://github.com/bokeh/bokeh/issues/13157
    """
    def __init__(self, width=600, height=330, size=10, title=''):

        # figure object
        self.fig = figure(width=width, height=height, tools='pan,box_zoom,wheel_zoom,save,reset', toolbar_location='left')
        self.fig.toolbar.logo = None
        self.fig.grid.visible = False
        self.fig.axis.visible = False
        self.fig.outline_line_color = None
        self.fig.x_range.start = 0
        self.fig.y_range.start = 0
        self.fig.x_range.end = width
        self.fig.y_range.end = height
        self.fig.title.text = title

        # data-source and glyph
        self.frame = ColumnDataSource()
        self.fig.image_url(url='frame', source=self.frame, x=0, y=0, w=width, h=height, anchor='bottom_left')
        largefonts(self.fig, size)

        # reset interface
        self.reset_interface()

    def reset_interface(self):
        """
        reset data source and yaxis ticks
        """
        self.frame.data = {'frame': np.array([])}

class MultiLineInterface1:
    """
    interface for figure object with following glyphs
    - n circle/line glyphs, style set by palette and line_width args
    - circle glyphs support tap tool
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
    def __init__(self, width=700, height=300, xlabel='', ylabel='', title='', size=12, legend_location='bottom_right', legend_layout_location='center',
        hover=False, tap=False, cross=False, dimensions='both', n=20, palette='Category20_20', circle=True, line=True, line_width=1,
        manual_xlim=False, manual_ylim=False, datetime=False, box_dimensions='both', toolbar_location='right'):

        # standard tools
        pan, wheel = PanTool(dimensions=dimensions), WheelZoomTool(dimensions=dimensions)
        box, reset, undo, save = BoxZoomTool(dimensions=box_dimensions), ResetTool(), UndoTool(), SaveTool()
        tools = [pan, wheel, box, reset, undo, save]

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
        self.fig = figure(width=width, height=height, tools=tools, toolbar_location=toolbar_location)
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
                self.fig.circle('x', 'y', source=src, size=6, color=color, name='circles',
                    # visual properties for selected
                    selection_color='red',
                    # visual properties for non-selected
                    nonselection_fill_alpha=1,
                    nonselection_fill_color=color)
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
        self.fig.add_layout(self.legend, legend_layout_location)

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

class MapInterface1:
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

        # assert bokeh.__version__ == '2.4.3'

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
        if bokeh.__version__ == '2.4.3':
            self.fig.add_tile(tile_source=get_provider(getattr(Vendors, map_vendor)))
        else:
            self.fig.add_tile(tile_source=map_vendor)
        self.fig.toolbar.logo = None
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
        self.fig.line('lon', 'lat', source=self.path, line_width=1, color='darkblue', name='path', alpha=0.4)

        # segments data source and glyphs
        self.segments = ColumnDataSource()
        self.fig.x('lon', 'lat', source=self.segments, size=6, color='violet', alpha=1, name='segments')
        self.fig.line('lon', 'lat', source=self.segments, line_width=2, color='violet', alpha=1, name='segments')

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

class MapInterface2:
    """
    interface for map object supporting following glyphs
        - path1 - darkblue circle/line glyphs
        - path2 - darkorange circle/line glyphs
        - position1 - darkblue asterik glyph
        - position2 - darkorange asterik glyph
        - event marker - red x glyph
    - arguments to init
        - width, height, map_vendor, size - map formatting
        - lon0, lon1, lat0, lat1 - default view
    - attributes created on init
        - fig - map as a bokeh figure object
        - ColumnDataSource objects for all glyphs
    - methods
        - reset_map_view
        - reset_interface
        - update_tile_source
    """

    def __init__(self, width=1200, height=300, map_vendor='OSM', size=12, lon0=-110, lon1=-70, lat0=25, lat1=50,
        legend_location='top_left', legend_layout_location='center'):

        # standard tools
        pan, wheel, box, reset, redo, undo = PanTool(), WheelZoomTool(), BoxZoomTool(), ResetTool(), RedoTool(), UndoTool()
        tools = [pan, wheel, box, reset, redo, undo]

        # interactive map as a figure object
        self.fig = figure(width=width, height=height, tools=tools, toolbar_location='right')
        self.fig.add_tile(tile_source=map_vendor)
        self.fig.toolbar.logo = None
        self.fig.toolbar.active_drag = pan
        self.fig.toolbar.active_scroll = wheel
        self.fig.outline_line_width = 2
        self.fig.xaxis.visible = False
        self.fig.yaxis.visible = False
        self.fig.grid.visible = False
        self.fig.x_range = Range1d()
        self.fig.y_range = Range1d()

        # path1 / position1 data source and glyphs
        self.path1 = ColumnDataSource()
        self.position1 = ColumnDataSource()
        self.event = ColumnDataSource()
        self.fig.circle('lon', 'lat', source=self.path1, size=6, color='darkblue', name='path1')
        self.fig.line('lon', 'lat', source=self.path1, line_width=1, color='darkblue', name='path1', alpha=0.4)
        self.fig.asterisk('lon', 'lat', source=self.position1, size=22, color='darkblue', line_color='darkblue', line_width=2, name='position1')

        # path2 / position2 data source and glyphs
        self.path2 = ColumnDataSource()
        self.position2 = ColumnDataSource()
        self.fig.circle('lon', 'lat', source=self.path2, size=6, color='green', name='path2')
        self.fig.line('lon', 'lat', source=self.path2, line_width=1, color='green', name='path2', alpha=0.4)
        self.fig.asterisk('lon', 'lat', source=self.position2, size=22, color='green', line_color='green', line_width=2, name='position2')

        # event glyph
        self.fig.x('lon', 'lat', source=self.event, size=22, color='red', line_color='red', line_width=3, name='event')

        # default view
        assert lon0 < lon1
        assert lat0 < lat1
        self.lon0 = lon0
        self.lon1 = lon1
        self.lat0 = lat0
        self.lat1 = lat1

        # legend object
        self.legend = Legend(location=legend_location, glyph_width=20, glyph_height=20)
        self.fig.add_layout(self.legend, legend_layout_location)

        # reset map view and interface
        largefonts(self.fig, size)
        self.reset_interface()
        self.reset_map_view()

    def reset_map_view(self, lon0=None, lon1=None, lat0=None, lat1=None, convert=True):
        """
        reset map view, use gps coords
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
        self.path1.data = {'lon': np.array([]), 'lat': np.array([])}
        self.position1.data = {'lon': np.array([]), 'lat': np.array([])}
        self.path2.data = {'lon': np.array([]), 'lat': np.array([])}
        self.position2.data = {'lon': np.array([]), 'lat': np.array([])}
        self.event.data = {'lon': np.array([]), 'lat': np.array([])}
        self.legend.items = []

    def update_tile_source(self, provider='OSM'):
        assert provider in ['OSM', 'ESRI']
        tr = [x for x in self.fig.renderers if isinstance(x, TileRenderer)]
        assert len(tr) == 1
        tr = tr[0]
        name = 'openstreetmap_mapnik' if provider == 'OSM' else 'esri_worldimagery' if provider == 'ESRI' else None
        provider = xyzservices.providers.query_name(name)
        ts = WMTSTileSource(url=provider.build_url(), attribution=provider.html_attribution, min_zoom=provider.get('min_zoom', 0), max_zoom=provider.get('max_zoom', 30))
        tr.update(tile_source=ts)

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
        self.fig.add_layout(Arrow(end=VeeHead(size=10, line_color='green', fill_color='green', line_width=2),
            line_color='green', line_width=2, source=self.positive, x_start='x0', y_start='y0', x_end='x1', y_end='y1'))
        self.fig.add_layout(Arrow(end=VeeHead(size=10, line_color='red', fill_color='red', line_width=2),
            line_color='red', line_width=2, source=self.negative, x_start='x0', y_start='y0', x_end='x1', y_end='y1'))

        # numeric labels at ends of each arrow
        self.fig.add_layout(LabelSet(x='x1', y='y1', text='labels', source=self.positive,
            text_align='left', text_baseline='middle', text_font_size='11pt', text_color='green'))
        self.fig.add_layout(LabelSet(x='x1', y='y1', text='labels', source=self.negative,
            text_align='right', text_baseline='middle', text_font_size='11pt', text_color='red'))

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
