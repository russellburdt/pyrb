
""" count lines of code in all .py files in all subdirectories of parent_dir
R Burdt, v0.01
12 Oct 2015
"""

import os
from glob import glob

parent_dir = r'c:\Users\rburdt\Documents\Python'

# walk thru parent_dir and all subdirs of parent_dir, save matching files with glob
pfiles = []
for pydir in os.walk(parent_dir):
    for pfile in glob(os.path.join(pydir[0], '*.py')):
        pfiles.append(pfile)

# loop thru all pfiles, count all lines that are not newlines (comments count!)
lines = 0
for pfile in pfiles:
    with open(pfile) as fid:
        for i, line in enumerate(fid):
            if line != '\n':
                lines += 1
