
import os
from shutil import rmtree

def on_session_destroyed(session_context):
    """
    executes when the server closes a session
    """
    for x in session_context.remove:
        if os.path.isfile(x):
            os.remove(x)
            print(f'deleted {os.path.split(x)[1]}')
        if os.path.isdir(x):
            rmtree(x)
            print(f'deleted {os.path.split(x)[1]}')
