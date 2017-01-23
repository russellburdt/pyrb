
""" count total songs in all itunes playlists, print result to screen
create a backup of the library to itunes_dir in a format, e.g.,
iTunes Library 26 Oct 2015, 20, 327.itl, where '20' is the number of playlists
and '327' is the number of songs in all playlists; overwrite any existing
file of the same name
*** all of this is to prevent ITunes from messing with my playlists ***
this is a Python 2 script due to pyItunes dependency
"""

import pyItunes
import os
import time
from shutil import copyfile

def get_summary_info(fname, verbose=True):
    """ method to print and return playlist information """

    # get the playlist information, ignore certain playlists in the list
    library = pyItunes.Library(fname)
    not_a_playlist = ['Recently Added', 'Purchased', 'Spanish', 'Audiobooks', u'iTunes\xa0U']
    playlists = [x for x in library.getPlaylistNames() if x not in not_a_playlist]
    n_songs = [len(library.getPlaylist(x).tracks) for x in playlists]

    # print information to the console if requested, and return the info
    if verbose:
        print 'Number of iTunes playlists: %d' % len(n_songs)
        print 'Number of songs in all iTunes playlists: %d' % sum(n_songs)
    return len(n_songs), sum(n_songs)

def save_library_backup(fname, nlists, nsongs):
    """ save a backup of the ITunes .itl library file to a subfolder called
    Previous iTunes Libraries, which is a default library backup folder created
    by an ITunes installation (but only used once per install). Create this subfolder
    if necessary. Copy the current library .itl file to the backup folder. Use the
    filename format described in above docstring, and replace if necessary.
    """

    # get path the the backup folder
    backup_dir = os.path.join(os.path.split(fname)[0], r'Previous iTunes Libraries')

    # create a filename for the library backup, join it to the backup folder
    backup = os.path.join(backup_dir, '%s %s, %d, %d.itl' %
                         (os.path.split(fname)[1].rpartition('.')[0],
                          time.strftime('%d %b %Y'), nlists, nsongs))

    # get source filename of the current ITunes library
    source = fname.rpartition('.')[0] + '.itl'

    # copy source file to backup file, default behavior is backup exists is okay
    copyfile(source, backup)

# enter name of iTunes xml library 0 pyItunes needs this file
fname = r'/Users/rburdt/Music/iTunes/iTunes Library.xml'
nlists, nsongs = get_summary_info(fname, verbose=True)
save_library_backup(fname, nlists, nsongs)
