
"""
short bokeh utility methods
"""


from ipdb import set_trace


def linkfigs(figs):

    set_trace()


    # at this point I am going to need a utility function to link plots
    # e.g. pyrb.linkfigs(figs, axis='x'), it will do:
    # figs[1].x_range = figs[0].x_range
    # figs[2].x_range = figs[0].x_range
    # ..., and same thing for axis='y' or axis='xy'
    # otherwise this all must be managed in figure method calls
