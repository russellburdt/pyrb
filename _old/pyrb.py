
"""
short Python utilities
Russell Burdt, v0.41
1 Nov 2016
"""


class PBar():
    """ console progressbar based on typical usage of other progressbar module
        includes percentage, estimated time, and bar widgets
        typical usage:
        pbar = pyrb.PBar('Running', len(items))
        for i, item in enumerate(items):
            <do stuff>
            pbar.update(i)
        pbar.finish() """

    # initialize the class
    def __init__(self, title, maxval=1):

        # import other progressbar module
        import progressbar

        # set up the progressbar configuration
        self.widgets = [title + ' ',
                        progressbar.Percentage(),
                        progressbar.Bar(),
                        progressbar.ETA()]

        # create progressbar
        self.pbar = progressbar.ProgressBar(\
                    widgets = self.widgets, maxval = maxval).start()

    # update progressbar
    def update(self, value):
        self.pbar.update(value + 1)

    # force progressbar to maxval
    def finish(self):
        self.pbar.finish()

def set_trace(gui=False):
    """ set debug breakpoint, with flag for use in PyQt4 GUI debugging
    (the flag is not necessary for mac, but used for windows, don't know why """

    import sys

    if gui:
        from PyQt4.QtCore import pyqtRemoveInputHook
        pyqtRemoveInputHook()

    from IPython.core.debugger import Pdb
    Pdb(color_scheme='Linux').set_trace(sys._getframe().f_back)

def set_itrace(gui=False):
    """ set ipython debug breakpoint, with flag for use in PyQt4 GUI debugging
    (the flag is not necessary for mac, but used for windows, don't know why """

    if gui:
        from PyQt4.QtCore import pyqtRemoveInputHook
        pyqtRemoveInputHook()

    from IPython.terminal.embed import InteractiveShellEmbed
    from IPython.config.loader import Config
    from inspect import currentframe

    # Configure the prompt so that I know I am in a nested (embedded) shell
    cfg = Config()
    prompt_config = cfg.PromptManager
    prompt_config.in_template = 'N.In <\\#>: '
    prompt_config.in2_template = '   .\\D.: '
    prompt_config.out_template = 'N.Out<\\#>: '

    # Messages displayed when I drop into and exit the shell.
    banner_msg = ("\n**Nested Interpreter:\n"
    "Hit Ctrl-D to exit interpreter and continue program.\n"
    "Note that if you use %kill_embedded, you can fully deactivate\n"
    "This embedded instance so it will never turn on again")
    exit_msg = '**Leaving Nested interpreter'

    # Put ipshell() anywhere in your code where you want it to open.
    ipshell = InteractiveShellEmbed(config=cfg, banner1=banner_msg, exit_msg=exit_msg)
    frame = currentframe().f_back
    msg = 'Stopped at {0.f_code.co_filename} and line {0.f_lineno}'.format(frame)
    ipshell(msg,stack_depth=2) # Go back one level!

def debug(f, *args, **kwargs):
    """ debug function """
    from IPython.core.debugger import Pdb
    pdb = Pdb(color_scheme = 'Linux')
    return pdb.runcall(f, *args, **kwargs)
