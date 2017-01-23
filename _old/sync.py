
"""
custom Python synchronization tool
R Burdt, v0.02
17 Sep 2015
"""

import os
import shutil
import filecmp
from rbDialogs import ProgressBar
from PyQt4.QtGui import QApplication
import sys

# sync everything in dir1 to dir2
dir1 = r'c:\Users\rburdt\Documents'
dir2 = r'w:\ENG\System\Personal Folders\RussellB\Documents Backup'

# shallow=True compares size, shallow=False compares bytes
# verbose=True to print messages for all src to dest overwrites, False otherwise
# careful=True to ask user if a dest file modified later in time than a src file should
# be replaced by the src file (src and dest have same filename, but are different)
shallow = True
verbose = True
careful = False

# copies src to dest and prints verbose message if requested
def copy_src_dest(src, dest, verbose=False, location=None):
    from shutil import copyfile
    import os

    copyfile(src, dest)
    if verbose:
        print('\nReplaced destination file with source file at: {}'.\
              format(os.path.join(*location)))

# get total number of directories and files in dir1, for use with a progressbar later
count = 0
for root, dirs, files in os.walk(dir1):
    for dir in dirs:
        count += 1
    for file in files:
        count += 1

# initialize a gui progressbar
app = QApplication(sys.argv)
pbar = ProgressBar('Syncing Files...', count)
pbar.show()
progress = 0

# walk thru dir1 and sync with dir2
for root, dirs, files in os.walk(dir1):

    # synchronize base path of dir2 with dir1
    extra = [x for x in root.split(os.path.sep) if x not in dir1.split(os.path.sep)]
    extra.insert(0, dir2)
    dir2_root = os.path.join(*extra)

    # scan thru dirs in dir1 tree, ensure the same directory structure in dir2
    for dir in dirs:
        pbar.update(progress)
        progress += 1
        target = os.path.join(dir2_root, dir)
        if not os.path.isdir(target):
            os.mkdir(target)

    # scan thru file in files:
    for file in files:
        pbar.update(progress)
        progress += 1
        src = os.path.join(root, file)
        dest = os.path.join(dir2_root, file)

        # if the source file is not in destination copy it there and continue
        if not os.path.isfile(dest):
            shutil.copyfile(src, dest)
            continue

        # continue if files are compared to be the same
        if filecmp.cmp(src, dest, shallow=shallow):
            continue

        # anytime src file modified later than dest file, replace dest with src
        if os.path.getmtime(src) > os.path.getmtime(dest):
            copy_src_dest(src, dest, verbose, extra[1:] + [file])

        # case where dest file modified later than src file
        elif os.path.getmtime(dest) > os.path.getmtime(src):

            # if careful=True, ask the user if replacement is really requested
            if careful:
                ans = None
                while ans not in ['y', '', 'n']:
                    ans = input('\nShould dest file modified more recently than src ' +
                                'file be overwritten ([y]/n)? ')
                if ans == '' or ans == 'y':
                    copy_src_dest(src, dest, verbose, extra[1:] + [file])
                elif ans == 'n':
                    continue
            # otherwise replace dest with src
            else:
                copy_src_dest(src, dest, verbose, extra[1:] + [file])

# close the progressbar
pbar.close()
