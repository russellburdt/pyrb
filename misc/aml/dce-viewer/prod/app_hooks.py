
"""
dce viewer app hooks
"""

import os
import config
from glob import glob


def on_session_destroyed(session_context):
    """
    executes when the server closes a session
    """

    # get session id
    sid = session_context.session.id

    # remove temp dir for session id
    rdir = os.path.join(config.RBIN, sid)
    if os.path.isdir(rdir):
        cmd = 'rm -r ' + rdir
        os.system(cmd)

    # remove video files for session id
    for fn in glob(os.path.join(config.SDIR, f'video_{sid}_*.mp4')):
        os.remove(fn)
