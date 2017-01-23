
"""
custom Qt Dialog widgets
Russell Burdt, v0.2
Aug 2015
"""

from PyQt4 import QtGui, QtCore
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4 import NavigationToolbar2QT as NavigationToolbar
from matplotlib.widgets import MultiCursor
import matplotlib.pyplot as plt
import os.path
import time
import dateutil
import numpy as np
import sys
from pyrb import set_trace, set_itrace


def pbar(title, range):
    """ supposed to be a wrapper around class ProgressBar, but unworking """

    app = QtGui.QApplication(sys.argv)
    bar = ProgressBar(title, range)
    bar.show()
    return bar

class ProgressBar(QtGui.QWidget):
    """ GUI progressbar for use with a python script
    does not work very well, e.g., freezes if you move the progressbar
    a PyQt4 GUI progressbar is not working outside of a PyQt4 gui
    example usage:
    app = QtGui.QApplication(sys.argv)
    bar = rbDialogs.ProgressBar('Running', 200)
    bar.show()
    for i in range(200):
        bar.update(i)
        time.sleep(0.001)
    bar.close()
    """

    def __init__(self, title='', total=20):
        super().__init__()
        self.progressbar = QtGui.QProgressBar()
        self.progressbar.setMaximum(total)
        self.progressbar.setWindowTitle(title)
        layout = QtGui.QVBoxLayout()
        layout.addWidget(self.progressbar)
        self.setLayout(layout)

    def update(self, value):
        self.progressbar.setValue(value)

class MultiCursor2(MultiCursor):
    """ MutliCursor2 is a class inherited from matplotlib.widgets.MultiCursor that snaps
    to data points and prints a value to the axes
    ax is a list of axes, x is shared x-data for all axes, y is y-data for each axes and
    must be an iterable with same length as ax

    example:
    import matplotlib.pyplot as plt
    import numpy as np
    import rbDialogs

    x1a = np.linspace(0, 5*np.pi, 5e1)
    y1a = np.sin(x1a)
    x1b = np.linspace(0, 5*np.pi, 5e2)
    y1b = 3 + np.cos(x1b)
    x2 = np.linspace(0, 5*np.pi, 2e2)
    y2 = np.sin(x2)
    x3 = np.linspace(0, 5*np.pi, 1e3)
    y3 = np.cos(x3**2)

    fig, ax = plt.subplots(3, 1, sharex=True, figsize=(16, 9))
    ax[0].plot(x1a, y1a, '.-')
    ax[0].plot(x1b, y1b, '.-')
    ax[1].plot(x2, y2, '.-')
    ax[2].plot(x3, y3, '.-')
    data = [[(x1a, y1a), (x1b, y1b)], [(x2, y2)], [(x3, y3)]]
    cursor = rbDialogs.MultiCursor2(fig.canvas, ax, data, color='r', lw=3, after_decimal=4)
    plt.show()
    """

    def __init__(self, canvas, ax, data, color='m', lw=2, after_decimal=1):
        super().__init__(canvas, ax, useblit=True, horizOn=False, vertOn=True, color=color, lw=lw)
        self.canvas = canvas
        self.ax = ax
        self.data = data
        self.after_decimal = after_decimal

    def onmove(self, event):
        """ overloaded onmove method """

        # run MultiCursor onmove method to update cursor
        MultiCursor.onmove(self, event)

        # return for no event.xdata or if toolbar is activated
        if event.xdata is None:
            return

#        # clear current labels during any toolbar action
#        if self.canvas.manager.toolbar._active is not None:
#            if not hasattr(self, 'text'):
#                return
#            else:
#                for ax_tmp in self.ax:
#                    for text in ax_tmp.texts:
#                        text.remove()

        # handle case of initial text objects not yet created
        if not hasattr(self, 'text'):
            self.text = []

            # scan over axes then datasets on each axes
            for ax, data_ax in zip(self.ax, self.data):
                for dataset in data_ax:

                    # get temporary x, y data
                    x_tmp = dataset[0]
                    y_tmp = dataset[1]

                    # find nearest x-data point to the event x-data
                    x_idx = np.argmin(np.abs(x_tmp - event.xdata))

                    # get x, y coords of datapoint
                    x_tmp = x_tmp[x_idx]
                    y_tmp = y_tmp[x_idx]

                    # create and append the initial text object to self.text list
                    self.text.append(ax.text(x_tmp, y_tmp, '{a:.{b}f}'.\
                        format(a=y_tmp, b=self.after_decimal),
                        horizontalalignment='left', verticalalignment='bottom'))

        # handle case of initial text objects already created
        else:

            # scan over axes then datasets on each axes
            counter = 0
            for ax, data_ax in zip(self.ax, self.data):
                for dataset in data_ax:

                    # get temporary x, y data
                    x_tmp = dataset[0]
                    y_tmp = dataset[1]

                    # find nearest x-data point to the event x-data
                    x_idx = np.argmin(np.abs(x_tmp - event.xdata))

                    # get x, y coords of datapoint
                    x_tmp = x_tmp[x_idx]
                    y_tmp = y_tmp[x_idx]

                    # update text object and increment counter
                    self.text[counter].set_position((x_tmp, y_tmp))
                    self.text[counter].set_text('{a:.{b}f}'.\
                        format(a=y_tmp, b=self.after_decimal))
                    counter += 1

        # update everything
        self.canvas.draw_idle()
#        counter = 0
#        for ax, data_ax in zip(self.ax, self.data):
#                for _ in range(len(data_ax)):
#                    text = self.text[counter]
#                    ax.draw_artist(text)
#                    self.canvas.blit(text.get_window_extent())
#                    counter += 1
