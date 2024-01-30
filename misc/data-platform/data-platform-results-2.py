
"""
validation of new data platform based on extracted data
- based on raw trackpoints data for individual vehicles on a single day
"""

import os
import pickle
import pandas as pd
from bokeh.plotting import figure, show, output_file
from bokeh.tile_providers import Vendors, get_provider
from bokeh.models import PanTool, WheelZoomTool, ResetTool
from pyrb.bokeh import largefonts
from tqdm import tqdm
from ipdb import set_trace


# read results based on hadoop and new data platform
fn0 = r'c:/Users/russell.burdt/Data/data-platform/results-hadoop.p'
fn1 = r'c:/Users/russell.burdt/Data/data-platform/results-new-data-platform.p'
with open(fn0, 'rb') as fid:
    r0 = pickle.load(fid)
with open(fn1, 'rb') as fid:
    r1 = pickle.load(fid)
assert r0.keys() == r1.keys()

# scan over cases
for case in tqdm(r0.keys(), desc='scanning cases'):

    # validate same case, extract trackpoints data and convert units
    for key in ['vid', 'time0', 'time1']:
        assert r0[case][key] == r1[case][key]
        assert r0[case][key] == r1[case][key]
    df0 = r0[case]['data']
    df1 = r1[case]['data']
    df0.sort_values('gpsdatetime').reset_index(drop=True)
    df1.sort_values('gpsdatetime').reset_index(drop=True)

    # tools, fig, and map provider objects
    title = 'all trackpoints, {}, {} to {}'.format(r0[case]['vid'], r0[case]['time0'], r0[case]['time1'])
    pan, wheel, reset = PanTool(), WheelZoomTool(), ResetTool()
    fig = figure(
        title=title, width=800, height=400, toolbar_location= 'above', tools=[pan, wheel, reset],
        x_axis_type='mercator', y_axis_type='mercator', x_range=[-10e6, 0], y_range=[0, 8e6])
    fig.add_tile(tile_source=get_provider('OSM'))
    fig.toolbar.active_drag = pan
    fig.toolbar.active_scroll = wheel
    fig.toolbar.logo = None
    fig.outline_line_width = 2
    fig.xaxis.visible = False
    fig.yaxis.visible = False
    fig.grid.visible = False

    # trackpoint glyphs for hadoop and new data platform
    fig.circle(df0['longitude'], df0['latitude'], size=10, color='darkblue', legend_label='hadoop platform')
    fig.line(df0['longitude'], df0['latitude'], line_width=2, line_dash='solid', color='darkblue', legend_label='hadoop platform')
    fig.triangle(df1['longitude'], df1['latitude'], size=10, color='darkgreen', legend_label='AWS data platform')
    fig.line(df1['longitude'], df1['latitude'], line_width=2, line_dash='dashed', color='darkgreen', legend_label='AWS data platform')

    # clean up and show
    largefonts(fig, 10)
    output_file(os.path.join(r'c:/Users/russell.burdt/Downloads', f'{case}.html'))
    show(fig)
