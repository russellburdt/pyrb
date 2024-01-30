
"""
short PyQt4 dialog classes and methods
"""


from PyQt4 import QtGui, QtCore
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4 import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt
import os.path
import time
import dateutil
import numpy as np


def progressbar(title, range, parent, fontsize=14):
    """ GUI ProgressDialog for use within instantiated PyQt GUI
    where parent is the main PyQt class
    example usage:
    from rbDialogs import progressbar
    pbar = progressbar(pbar_title, len(fnames), parent)
    for i, ... in enumerate(...):
        ...
        pbar.setValue(i+1)

    pbar.setValue(len(...))
    pbar.close()
    """
    pbar = QtGui.QProgressDialog(title, '', 0, range, parent)
    pbar.setWindowModality(QtCore.Qt.WindowModal)
    pbar.setMinimumDuration(0)
    pbar.setFont(QtGui.QFont('Calibri', fontsize))
    [x for x in pbar.children() if isinstance(x, type(QtGui.QPushButton()))][0].setVisible(False)
    pbar.show()
    pbar.setValue(0.005 * range)
    return pbar

def center(self):
    """
    center widget method
    """
    qr = self.frameGeometry()
    cp = QtGui.QDesktopWidget().availableGeometry().center()
    qr.moveCenter(cp)
    self.move(qr.topLeft())

class MultiLabelTextDialog(QtGui.QDialog):
    """
    dialog used to set values of a dictionary, where the keys are QLabels
    displayed in the left columns of the dialog, and the values are QLineEdit
    boxes with an empty of default value in them. Closing the dialog with the
    ok button returns a dict with key-value pairs in the dialog
    """

    # initialize method
    def __init__(self, fontsize, window_title, keys, defaults, key_text,
                 value_text, parent=None):

        # set up inheritance and layouts
        super(MultiLabelTextDialog, self).__init__(parent)
        v_layout = QtGui.QVBoxLayout(self)
        h_layout = QtGui.QHBoxLayout()
        v_layout.addLayout(h_layout)
        vbox_left = QtGui.QVBoxLayout()
        h_layout.addLayout(vbox_left)
        vbox_right = QtGui.QVBoxLayout()
        h_layout.addLayout(vbox_right)

        key_label = QtGui.QLabel(key_text)
        vbox_left.addWidget(key_label)
        value_label = QtGui.QLabel(value_text)
        vbox_right.addWidget(value_label)

        # set title and icon
        self.setWindowTitle(window_title)
        self.setWindowIcon(QtGui.QIcon('wolf.png'))

        # create a QLabel and a QLineEdit for every item in keys
        self.edit_list = []
        for key, default in zip(keys, defaults):
            label_tmp = QtGui.QLabel(key)
            label_tmp.setFont(QtGui.QFont('Calibri', fontsize-2))
            vbox_left.addWidget(label_tmp)
            edit_tmp = QtGui.QLineEdit()
            edit_tmp.setFont(QtGui.QFont('Calibri', fontsize-2))
            edit_tmp.setText(default)
            vbox_right.addWidget(edit_tmp)
            self.edit_list.append(edit_tmp)

        # create OK and cancel buttons, add to layout in a groupbox with title
        okbox = QtGui.QGroupBox('Accept Selections or Cancel')
        v_layout.addWidget(okbox)
        box_layout = QtGui.QHBoxLayout(okbox)
        buttons = QtGui.QDialogButtonBox(
            QtGui.QDialogButtonBox.Ok | QtGui.QDialogButtonBox.Cancel,
            QtCore.Qt.Horizontal, self)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        box_layout.addWidget(buttons)

        # adjust fontsizes
        okbox.setFont(QtGui.QFont('Calibri', fontsize-2))
        buttons.setFont(QtGui.QFont('Calibri', fontsize-2))

        # adjust fontsizes
        key_label.setFont(QtGui.QFont('Calibri', fontsize))
        value_label.setFont(QtGui.QFont('Calibri', fontsize))

    # this method is called when this dialog is created
    def getItems(parent=None, fontsize=12, keys=['a', 'b'], defaults=['0', '0'],
                 window_title='Select Values For Each Item', key_label='keys',
                 value_label='values'):

        # create the dialog and run it
        dialog = MultiLabelTextDialog(fontsize, window_title, keys, defaults,
                                      key_label, value_label)
        result = dialog.exec_()

        # get the values from the QLineEdits
        values = [x.text() for x in dialog.edit_list]

        # return the values and the result of the dialog interaction
        return values, result == QtGui.QDialog.Accepted

class ListSelectFilterDialog(QtGui.QDialog):
    """
    dialog made up of multiple ListSelectFilterWidgets, with a method called
    getItems to return results with an ok status.  inputs to __init__ are
    provided in the getItems method.
    kwargs for getItems include:
    *fontsize* standard fontsize argument, adjust as necessary, default=16
    *window_title* title for the dialog window
    *titles* a list of titles for each ListSelectFilterWidget
    *items* a list of lists, where each list is the default items in a ListSelectFilterWidget

    --- example usage ---
    import rbDialogs
    from PyQt4 import QtGui
    import sys
    from pyrb import set_trace

    class main(QtGui.QWidget):
        def __init__(self):
            super().__init__()
            layout = QtGui.QVBoxLayout(self)
            pb = QtGui.QPushButton('&Run Dialog')
            layout.addWidget(pb)
            pb.clicked.connect(self.pb_press)

        def pb_press(self):
            self.setVisible(False)
            titles = [
                'Select Signals from 1 MShot Log',
                'Select Events from FEW Log']
            items = [
                ['EMo Min', 'BW E95 Mean', 'EPa Max', 'ELaser Mean', 'EPa Min'],
                ['335', '336', '006', '041', '042', '040', '540', '957']]
            result, ok = rbDialogs.ListSelectFilterDialog.getItems(
                fontsize=14, window_title='Select Data For This Axes',
                titles=titles, items=items)
            print(result)
            print(ok)
            self.setVisible(True)
    """

    # initialize method
    def __init__(self, fontsize, window_title, titles, items, filter_select, parent=None):

        # set up inheritance and layouts
        super(ListSelectFilterDialog, self).__init__(parent)
        v_layout = QtGui.QVBoxLayout(self)

        # set title and icon
        self.setWindowTitle(window_title)
        self.setWindowIcon(QtGui.QIcon('wolf.png'))

        # set the number of widgets per row, initialize a row counter
        widgets_per_row = 4
        rows = 0

        # add ListSelectFilterWidgets to the horizontal layout
        self.ListSelectFilterWidgets = []
        for i, (title, item, filter_ok) in enumerate(zip(titles, items, filter_select)):

            # create the ListSelectFilterWidget
            self.ListSelectFilterWidgets.append(ListSelectFilterWidget(
                title=title, items=item, fontsize=fontsize,
                filter_select=filter_ok))

            # decide whether or not to create a new HLayout for widgets, and add the
            # ListSelectFilterWidget that was just created
            if np.mod(i, widgets_per_row) == 0:
                h_layout = QtGui.QHBoxLayout()
                v_layout.addLayout(h_layout)
                rows += 1
            h_layout.addWidget(self.ListSelectFilterWidgets[-1])

        # add some empty widgets to the bottom row of multi-row dialogs for scaling
        missing = widgets_per_row - np.mod(len(self.ListSelectFilterWidgets), widgets_per_row)
        if rows > 1 and missing < widgets_per_row:
            [h_layout.addWidget(QtGui.QWidget()) for _ in range(missing)]

        # create OK and cancel buttons, add to layout in a groupbox with title
        okbox = QtGui.QGroupBox('Accept Selections or Cancel')
        v_layout.addWidget(okbox)
        box_layout = QtGui.QHBoxLayout(okbox)
        buttons = QtGui.QDialogButtonBox(
            QtGui.QDialogButtonBox.Ok | QtGui.QDialogButtonBox.Cancel,
            QtCore.Qt.Horizontal, self)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        box_layout.addWidget(buttons)

        # adjust fontsizes
        okbox.setFont(QtGui.QFont('Calibri', fontsize-2))
        buttons.setFont(QtGui.QFont('Calibri', fontsize-2))

    # this method is called when this dialog is created
    def getItems(parent=None, fontsize=16, window_title='Select Data',
                 titles=[], items=[], filter_select=[]):

        # create the dialog and run it
        dialog = ListSelectFilterDialog(fontsize, window_title, titles, items,
                                        filter_select, parent)
        result = dialog.exec_()

        # get a list of lists, where each list is the selected items in a ListSelectFilterWidget
        items_selected = [x.return_selected_items() for x in dialog.ListSelectFilterWidgets]

        # return the items selected and the result of the dialog interaction
        return items_selected, result == QtGui.QDialog.Accepted

class ListSelectFilterWidget(QtGui.QWidget):
    """
    widget to select multiple items from a long list with a filter and
    buttons to select all or select none
    """

    # initialize method, title, items, and fontsize passed as input
    def __init__(self, title='Select Signals', items=[], fontsize=14, filter_select=False):

        # define the inheritance, add the initial list of items to self data
        super().__init__()
        self.all_items = items

        # create Grid Layout with self as the parent
        layout = QtGui.QGridLayout(self)

        # add the title, add the filter line edit and wire to filter_updated
        self.title = QtGui.QLabel(title)
        layout.addWidget(self.title, 0, 0, 1, 6)
        self.filter = QtGui.QLabel('Filter')
        layout.addWidget(self.filter, 1, 0, 1, 1)
        self.line_edit = QtGui.QLineEdit()

        # add the filter line-edit and a '+' box if filter_select is True
        if filter_select:
            layout.addWidget(self.line_edit, 1, 1, 1, 4)
            add_filter = QtGui.QPushButton('+')
            add_filter.setFixedWidth(20)
            add_filter.setFont(QtGui.QFont('Calibri', 12))
            layout.addWidget(add_filter, 1, 5, 1, 1)
            add_filter.clicked.connect(self.add_the_filter)
        else:
            layout.addWidget(self.line_edit, 1, 1, 1, 5)

        # keep track of the filter connected to its method
        self.filter_on = True
        self.line_edit.textChanged.connect(self.filter_updated)

        # create the QListWidget, add to the layout, populate with items
        self.list_widget = QtGui.QListWidget()
        self.list_widget.addItems(items)
        self.list_widget.setSelectionMode(2)
        layout.addWidget(self.list_widget, 2, 0, 3, 6)

        # create buttons to select all and select none from the current list
        self.select_all_button = QtGui.QPushButton('Select &All Items')
        self.select_all_button.clicked.connect(self.select_all_action)
        layout.addWidget(self.select_all_button, 5, 0, 1, 3)

        self.select_none_button = QtGui.QPushButton('Select &None')
        self.select_none_button.clicked.connect(self.select_none_action)
        layout.addWidget(self.select_none_button, 5, 3, 1, 3)

        # adjust font size for all widgets
        self.title.setFont(QtGui.QFont('Calibri', fontsize))
        self.filter.setFont(QtGui.QFont('Calibri', fontsize))
        self.line_edit.setFont(QtGui.QFont('Calibri', fontsize))
        self.list_widget.setFont(QtGui.QFont('Calibri', fontsize-2))
        self.select_all_button.setFont(QtGui.QFont('Calibri', fontsize))
        self.select_none_button.setFont(QtGui.QFont('Calibri', fontsize))

    # text in the filter can be added to the list and selected only if the
    # current filter does not match anything, in this case the normal filtering
    # is disabled and the filter box is used as an add-text box - this feature
    # is kind of a hidden feature
    def add_the_filter(self):

        # first make sure nothing is in the list, if so disable the filter box
        if self.filter_on:
            if self.list_widget.count() > 0:
                return
            else:
                self.line_edit.textChanged.disconnect(self.filter_updated)
                self.filter_on = False

        # add the filter to the list box in a selected state, then clear
        self.list_widget.addItem(self.line_edit.text())
        self.line_edit.clear()

    # return the items selected in the list widget
    def return_selected_items(self):
        selected_items = [x.text() for x in self.list_widget.selectedItems()]
        return selected_items

    # action to select all items in the list widget
    def select_all_action(self):
        [self.list_widget.item(x).setSelected(True) for x in range(self.list_widget.count())]

    # action to select no items from the list widget
    def select_none_action(self):
        [self.list_widget.item(x).setSelected(False) for x in range(self.list_widget.count())]

    # method runs on any text update to the filter line-edit box
    def filter_updated(self):

        # get the filter from the line edit, clear the list widget
        filter = self.line_edit.text().lower()
        self.list_widget.clear()

        # get a lowercase list of all_items
        all_items = [x.lower() for x in self.all_items]

        # get the filtered items as a subset of all_items containing the filter
        # note: filtering is done on the list of lowercase items
        filtered_items = [a for a, b in zip(self.all_items, all_items) if filter in b]

        # add the filtered items to the list widget
        self.list_widget.addItems(filtered_items)

class AddStrToListWidget(QtGui.QWidget):
    """
    create widget to add text or date strings to a modifieable list
    """

    # initialize method
    def __init__(self, **kwargs):

        # extract data passed to class, default is 2nd arg of pop
        self.text0 = kwargs.pop('text0', '')        # initial title
        self.unique = kwargs.pop('unique', True)    # enforce unique entries
        self.isDate = kwargs.pop('isDate', False)   # is it a date?
        self.format = kwargs.pop('format',          # format for dates
                                 '%d %B %Y')
        self.Nlimit = kwargs.pop('Nlimit', None)    # length limit for text input
        self.Nexact = kwargs.pop('Nexact', None)    # exact length requirement
        self.title = kwargs.pop('title', None)      # static text above QListView
        self.upper = kwargs.pop('upper', False)     # if True, make entry uppercase

        # parent_method is a method in the calling object that will run on actions
        self.parent_method = kwargs.pop('parent_method', None)

        # define the inheritance
        super().__init__()

        # set text0 as current date if isDate and no initial date given
        if self.isDate and self.text0 == '':
            self.text0 = 'Enter Date, {}'.format(time.strftime(self.format))

        # create Grid Layout, add a label above the QLineEdit and QPushButton
        layout = QtGui.QGridLayout(self)
        if self.title is not None:
            layout.addWidget(QtGui.QLabel(self.title), 0, 0, 1, 2)
            n = 1
        else:
            n = 0

        # define line-edit and pushbutton, add to layout
        self.line_edit = QtGui.QLineEdit(self.text0)
        layout.addWidget(self.line_edit, n, 0)
        self.add_button = QtGui.QPushButton('Add')
        layout.addWidget(self.add_button, n, 1)

        # wire the pushbutton to add item to groupbox
        self.add_button.clicked.connect(self.add_press)

        # create the list widget and add to the layout
        self.items_view = QtGui.QListView()
        self.items_model = QtGui.QStandardItemModel(self.items_view)
        self.items_view.setModel(self.items_model)
        self.items_view.setSpacing(3)
        self.items_view.setSelectionMode(False)
        layout.addWidget(self.items_view, n + 1, 0, 1, 2)

        # wire the list models to parent_method if provided
        if self.parent_method is not None:
            self.items_model.itemChanged.connect(self.parent_method)

    # define action for add pushbutton
    def add_press(self):

        # get text from the QLineEdit, and the current list from the QList
        text = self.line_edit.text()
        list0 = [self.items_model.item(x).text() for x in range(self.items_model.rowCount())]

        # make text uppercase if requested
        if self.upper:
            text = text.upper()

        # get criteria for decision whether to ignore text entry (not isDate)
        # c0 -> if nothing is there
        # c1 -> if text0 has not been changed
        # c2 -> if text is already in the list and unique entries are set
        # c3 -> if the text length is greater than Nlimit and Nlimit is set
        # c4 -> if the text length is not equal to Nexact (and it's set)
        if not self.isDate:
            c0 = text == ''
            c1 = text == self.text0
            c2 = (text in list0) and self.unique
            c3 = (self.Nlimit is not None) and len(text) > self.Nlimit
            c4 = (self.Nexact is not None) and len(text) != self.Nexact
            if c0 or c1 or c2 or c3 or c4:
                self.line_edit.setText(self.text0)
                return
        # different criteria are used for isDate True with a date entry
        else:
            try:
                # assert the condition that the text entry is not empty
                assert text != ''
                # check if text entry is parseable as a date string
                time_tuple = dateutil.parser.parse(text)
                text = time_tuple.strftime(self.format)     # put it in self.format
                # return if the date is already there and unique dates are enforced
                if (text in list0) and self.unique:
                    self.line_edit.setText(self.text0)
                    return

            # placeholder for date widget that returns a date in self.format
            # in case text was empty or entry was unparseable
            except (ValueError, AssertionError):
                print('placeholder for date widget')
                self.line_edit.setText(self.text0)
                return

        # add text to the groupbox, initialize state
        text = QtGui.QStandardItem(text)
        text.setEditable(False)
        text.setSelectable(True)
        text.setCheckable(True)
        text.setCheckState(2)
        text.setEnabled(True)
        self.items_model.appendRow(text)

        # call parent_method if passed as a kwarg
        if self.parent_method is not None:
            self.parent_method()

class SimplePlotWidget(QtGui.QWidget):
    """
    create simple widget with a figure canvas on a VBox layout
    """

    # initialize method
    def __init__(self, window_title=None, addNavBar=True):

        # define the inheritance
        super().__init__()

        # create VBox layout
        layout = QtGui.QVBoxLayout(self)

        # set up a figure and a canvas on the figure
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        if window_title is not None:
            self.canvas.setWindowTitle(window_title)

        # set up a navigation toolbar and add to the layout, if requested
        if addNavBar:
            self.navbar = NavigationToolbar(self.canvas, self)
            layout.addWidget(self.navbar)

        # add the canvas to the layout and set the facecolor
        layout.addWidget(self.canvas)
        self.figure.set_facecolor('w')

class DataFolder(QtGui.QDialog):
    """
    create dialog to add a data folder in a text box with a search pushbutton
    """

    def __init__(self, dir0, window_title, fontsize, parent=None):

        # set up inheritance, create Grid layout
        super(DataFolder, self).__init__(parent)
        layout = QtGui.QGridLayout(self)
        self.dir0 = dir0

        # resize, set title and icon
        self.resize(600, 100)
        self.setWindowTitle(window_title)
        self.setWindowIcon(QtGui.QIcon('wolf.png'))

        # set up directory enter box
        self.dirlabel = QtGui.QLabel(
            'Enter Folder Location, or Search For It')
        layout.addWidget(self.dirlabel, 0, 0)
        self.dirbox = QtGui.QLineEdit()
        layout.addWidget(self.dirbox, 1, 0, 1, 4)
        self.searchbutton = QtGui.QPushButton('&Search')
        self.searchbutton.resize(self.searchbutton.sizeHint())
        layout.addWidget(self.searchbutton, 1, 4)

        # wire up directory box and search
        self.dirbox.textChanged.connect(self.checkText)
        self.searchbutton.clicked.connect(self.searchpress)

        # create OK and cancel buttons, add to layout in a group box
        self.okbox = QtGui.QGroupBox('Accept Location or Cancel')
        layout.addWidget(self.okbox, 2, 0)
        boxlayout = QtGui.QVBoxLayout(self.okbox)
        self.buttons = QtGui.QDialogButtonBox(
            QtGui.QDialogButtonBox.Ok | QtGui.QDialogButtonBox.Cancel,
            QtCore.Qt.Horizontal, self.okbox)
        boxlayout.addWidget(self.buttons)
        self.okbox.setEnabled(False)

        # wire up buttons
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)

        # adjust fonts
        self.dirbox.setFont(QtGui.QFont('Calibri', fontsize))
        self.searchbutton.setFont(QtGui.QFont('Calibri', fontsize))
        self.okbox.setFont(QtGui.QFont('Calibri', fontsize))
        self.dirlabel.setFont(QtGui.QFont('Calibri', fontsize))

    # method to check the text in the dirbox
    def checkText(self):

        # check if what is entered in dirbox is an actual directory
        if os.path.isdir(self.dirbox.text()):

            # if it is a directory enable the rest of the GUI
            self.okbox.setEnabled(True)
            self.dirlabel.setText(
                'Enter Data Folder Location, or Search For It')

        else:

            # if it's not a directory, disable the rest of the GUI and print red message
            self.okbox.setEnabled(False)
            self.dirlabel.setText(
                'Enter Data Folder Location, or Search For It (**not a directory**)')

    # method to open a directory search dialog and populate the dirbox
    def searchpress(self):
        dir = QtGui.QFileDialog.getExistingDirectory(
            self, 'Get Data Directory', self.dir0)
        self.dirbox.setText(os.path.join(dir, ''))

    # method to create the dialog and return (folder, accepted) tuple
    def getValues(parent=None, dir0=os.path.expanduser('~'),
                  window_title='Set Folder', fontsize=12):
        dialog = DataFolder(dir0, window_title, fontsize)
        result = dialog.exec_()
        dataLoc = dialog.dirbox.text()
        return dataLoc, result == QtGui.QDialog.Accepted

class DataFolderAndLegend(QtGui.QDialog):
    """
    create dialog to add a data folder and create editable default legend
    """

    def __init__(self, dir0, window_title, fontsize, parent=None):

        # set up inheritance, create Grid layout
        super(DataFolderAndLegend, self).__init__(parent)
        layout = QtGui.QGridLayout(self)
        self.dir0 = dir0

        # resize, set title and icon
        self.resize(600, 150)
        self.setWindowTitle(window_title)
        self.setWindowIcon(QtGui.QIcon('wolf.png'))

        # set up directory enter box
        self.dirlabel = QtGui.QLabel(
            'Enter Data Folder Location, or Search For It')
        layout.addWidget(self.dirlabel, 0, 0)
        self.dirbox = QtGui.QLineEdit()
        layout.addWidget(self.dirbox, 1, 0, 1, 4)
        self.searchbutton = QtGui.QPushButton('&Search')
        self.searchbutton.resize(self.searchbutton.sizeHint())
        layout.addWidget(self.searchbutton, 1, 4)

        # wire up directory box and search
        self.dirbox.textChanged.connect(self.checkText)
        self.searchbutton.clicked.connect(self.searchpress)

        # set up legend enter box, initial state is disabled
        label = QtGui.QLabel('Edit Default Legend for Plots (not required)')
        layout.addWidget(label, 2, 0)
        self.legbox = QtGui.QLineEdit()
        layout.addWidget(self.legbox, 3, 0, 1, 5)
        self.legbox.setEnabled(False)

        # create OK and cancel buttons, add to layout in a group box
        self.okbox = QtGui.QGroupBox('Accept Data Location/Legend or Cancel')
        layout.addWidget(self.okbox, 4, 0)
        boxlayout = QtGui.QVBoxLayout(self.okbox)
        self.buttons = QtGui.QDialogButtonBox(
            QtGui.QDialogButtonBox.Ok | QtGui.QDialogButtonBox.Cancel,
            QtCore.Qt.Horizontal, self.okbox)
        boxlayout.addWidget(self.buttons)
        self.okbox.setEnabled(False)

        # wire up buttons
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)

        # adjust fonts
        self.dirbox.setFont(QtGui.QFont('Calibri', fontsize))
        self.legbox.setFont(QtGui.QFont('Calibri', fontsize))
        self.searchbutton.setFont(QtGui.QFont('Calibri', fontsize))
        self.okbox.setFont(QtGui.QFont('Calibri', fontsize))
        self.dirlabel.setFont(QtGui.QFont('Calibri', fontsize))
        label.setFont(QtGui.QFont('Calibri', fontsize))

    # method to check the text in the dirbox
    def checkText(self):

        # check if what is entered in dirbox is an actual directory
        if os.path.isdir(self.dirbox.text()):

            # if it is a directory enable the rest of the GUI and create a default legend
            legDefault = os.path.split(os.path.split(self.dirbox.text())[0])[1]
            self.legbox.setText(legDefault)
            self.legbox.setEnabled(True)
            self.okbox.setEnabled(True)
            self.dirlabel.setText(
                'Enter Data Folder Location, or Search For It')

        else:

            # if it's not a directory, disable the rest of the GUI and print red message
            self.legbox.setText('')
            self.legbox.setEnabled(False)
            self.okbox.setEnabled(False)
            self.dirlabel.setText(
                'Enter Data Folder Location, or Search For It (**not a directory**)')

    # method to open a directory search dialog and populate the dirbox
    def searchpress(self):
        dir = QtGui.QFileDialog.getExistingDirectory(
            self, 'Get Data Directory', self.dir0)
        self.dirbox.setText(os.path.join(dir, ''))

    # method to create the dialog and return (folder, legend, accepted) tuple
    def getValues(parent=None, dir0=os.path.expanduser('~'),
                  window_title='Set Folder', fontsize=12):
        dialog = DataFolderAndLegend(dir0, window_title, fontsize)
        result = dialog.exec_()
        dataLoc = dialog.dirbox.text()
        legend = dialog.legbox.text()
        return dataLoc, legend, result == QtGui.QDialog.Accepted

class DateDialog(QtGui.QDialog):
    """
    create dialog to get PyQt4.QtCore.QTime object from interactive calendar
    """

    def __init__(self, parent=None):

        # set up inheritance, create VBox layout
        super(DateDialog, self).__init__(parent)
        layout = QtGui.QVBoxLayout(self)

        # create nice widget for getting date and time, add to layout
        self.datetime = QtGui.QDateTimeEdit(self)
        self.datetime.setCalendarPopup(True)
        self.datetime.setDateTime(QtCore.QDateTime.currentDateTime())
        layout.addWidget(self.datetime)

        # create OK and cancel buttons, add to layout
        self.buttons = QtGui.QDialogButtonBox(
            QtGui.QDialogButtonBox.Ok | QtGui.QDialogButtonBox.Cancel,
            QtCore.Qt.Horizontal, self)
        layout.addWidget(self.buttons)

        # wire up buttons
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)

    # get current date and time from the dialog
    def dateTime(self):
        return self.datetime.dateTime()

    # static method to create the dialog and return (date, time, accepted)
    # @staticmethod - don't know the meaning of this line, code executes the same without it
    def getDateTime(parent=None):
        dialog = DateDialog(parent)
        result = dialog.exec_()
        date = dialog.dateTime()
        return (date.date(), date.time(), result == QtGui.QDialog.Accepted)
