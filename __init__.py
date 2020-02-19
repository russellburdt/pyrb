
"""
define hooks to some functions in other modules,
so existing tools referenced to pyrb continue to work
"""

# hooks to pyrb.mpl
from pyrb import mpl
addspecline = mpl.addspecline
format_axes = mpl.format_axes
get_current_figs = mpl.get_current_figs
get_random_closed_data_marker = mpl.get_random_closed_data_marker
largefonts = mpl.largefonts
maximize_figs = mpl.maximize_figs
open_figure = mpl.open_figure
save_pngs = mpl.save_pngs
subplots_adjust = mpl.subplots_adjust
thicklines = mpl.thicklines
update_figs = mpl.update_figs

# hooks to pyrb.processing
from pyrb import processing
get_bounds_of_data_within_interval = processing.get_bounds_of_data_within_interval
arange = processing.arange
is_datetime_week_number_begin = processing.is_datetime_week_number_begin
is_datetime_week_number_end = processing.is_datetime_week_number_end
datetime_to_week_number = processing.datetime_to_week_number
week_number_to_datetime = processing.week_number_to_datetime
create_filename = processing.create_filename
get_unambiguous_tz = processing.get_unambiguous_tz
get_utc_times = processing.get_utc_times
is_writeable = processing.is_writeable
linspace = processing.linspace
loadmat = processing.loadmat
longest_substring_from_list = processing.longest_substring_from_list
matlab2datetime = processing.matlab2datetime
memoized = processing.memoized
movingaverage = processing.movingaverage
movingstd = processing.movingstd
movingsum = processing.movingsum
numpy_datetime64_to_datetime = processing.numpy_datetime64_to_datetime
pngs2ppt = processing.pngs2ppt
get_aliased_freq = processing.get_aliased_freq
return_dict = processing.return_dict
run_notch_filter_example = processing.run_notch_filter_example
search_tree = processing.search_tree
