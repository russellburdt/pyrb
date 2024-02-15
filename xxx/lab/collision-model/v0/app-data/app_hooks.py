
"""
bokeh app hooks
"""
import os

def on_session_destroyed(session_context):
    """
    executes when the server closes a session
    """

    # remove video files
    for fn in session_context.vids:
        if os.path.isfile(fn):
            os.remove(fn)
