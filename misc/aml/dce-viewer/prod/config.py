
"""
dce viewer prod config
"""

# app directories
SDIR = '/mnt/home/russell.burdt/Lytx.AML.RussellB/aml/dce-viewer/prod/static'
VDIR = 'http://10.144.240.35/dce-viewer/prod'
RBIN = '/mnt/home/russell.burdt/rbin'

# map style (aspect ratios should be appx the same)
MAP_WIDTH = 450
MAP_HEIGHT = 240
MAP_X_RANGE = [-15e6, -3.75e6]
MAP_Y_RANGE = [1e6, 7e6]
GPS_VIEW_SCALE = 1

# video style
VIDEO_WIDTH = 900
VIDEO_HEIGHT = 250
VIDEO_STYLE0 = f"""<div style="background-color:#000000;width:{VIDEO_WIDTH}px;height:{VIDEO_HEIGHT}px;color:white;border:0">"""
VIDEO_STYLE1 = """</div>"""
