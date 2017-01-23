
"""
short data processing utility methods
"""


def return_aliased_freq(f, fs):
    """
    return the aliased frequency of f sampled at fs
    """
    import numpy as np

    fs = fs / 2
    if np.ceil(f / fs) % 2:
        return np.remainder(f, fs)
    else:
        return fs - np.remainder(f, fs)

def matlab2datetime(mat_dnums):
    """
    convert a 1d numpy array of float matlab datenums (e.g., 735976.001453)
    to a 1d numpy array of datetime objects
    """
    from datetime import datetime, timedelta
    import numpy as np

    day = np.array([datetime.fromordinal(int(x)) for x in mat_dnums])
    dayfrac = np.array([timedelta(days=(x % 1)) - timedelta(days=366) for x in mat_dnums])
    return day + dayfrac

def streaming_timestamp_to_datetime(timestamp):
    """
    convert streaming-data timestamp (e.g. '2016_05_19_13_41_53') to a datetime object
    """

    from datetime import datetime

    return datetime.strptime(timestamp, r'%Y_%m_%d_%H_%M_%S')

def loadmat(fname):
    """
    this function should be called instead of directly calling scipy.io.loadmat
    as it solves the problem of not properly recovering python dictionaries
    """

    import scipy.io as spio

    def check_keys(dict):
        """
        checks if entries in dictionary are mat-objects. If yes
        todict is called to change them to nested dictionaries
        """
        for key in dict:
            if isinstance(dict[key], spio.matlab.mio5_params.mat_struct):
                dict[key] = todict(dict[key])
        return dict

    def todict(matobj):
        """
        A recursive function which constructs from matobjects nested dictionaries
        """
        dict = {}
        for strg in matobj._fieldnames:
            elem = matobj.__dict__[strg]
            if isinstance(elem, spio.matlab.mio5_params.mat_struct):
                dict[strg] = todict(elem)
            else:
                dict[strg] = elem
        return dict

    data = spio.loadmat(fname, struct_as_record=False, squeeze_me=True)
    return check_keys(data)

def run_notch_filter_example():
    """
    run a notch filter example
    """

    import matplotlib.pyplot as plt
    import numpy as np
    import pyrb
    from scipy import signal
    from pyrb.mpl import open_figure, format_axes, largefonts
    plt.style.use('bmh')

    # define a sampling rate, fs, and N data points
    fs = 6000
    N = 1e5

    # calculate a time array based on fs and N
    dt = 1 / fs
    time = np.arange(0, N*dt, dt)

    # define y(time) data to includes freqs at mags, plus some baseline noise
    mags = [1, 2, 4, 2, 5, 3, 1]
    freqs = [250, 1200, 1917, 711, 2356, 2100, 8209]
    y = 0
    for mag, freq in zip(mags, freqs):
        y += mag * np.sin(2 * np.pi * freq * time)
    y += np.random.normal(0, 1, y.size)

    # calculate the psd of y data
    freq, psd = signal.welch(y, fs=fs, nperseg=512)

    # update freqs for aliasing, as any freq greater than fs/2 will alias to some other freq less than fs/2
    freqs = [return_aliased_freq(x, fs) for x in freqs]

    # select a random 'freqs' to filter, mapped to 0 to 1 scale where fs/2 maps to 1
    wf = np.random.choice(freqs) / (fs/2)

    # prepare the 0 to 1 mapped wp (pass-band) and ws (stop-band) edge frequencies
    wd = 25 / (fs/2)
    ws = [wf - wd, wf + wd]
    wp = [wf - 2 * wd, wf + 2 * wd]
    gpass, gstop = 3, 40

    # create the bandstop filter
    N, Wn = signal.cheb2ord(wp=wp, ws=ws, gpass=gpass, gstop=gstop)
    b, a = signal.iirfilter(N=N, Wn=Wn, rp=gpass, rs=gstop, btype='bandstop', ftype='cheby2')

    # apply the filter to y, get the psd of the filtered signal
    yf = signal.lfilter(b, a, y)
    freq_f, psd_f = signal.welch(yf, fs=fs, nperseg=512)

    # calculate filter response, create a results plot
    w, h = signal.freqz(b, a)
    wHz = w * fs / (2 * np.pi)
    fig, ax = open_figure('Notch Filter Example', 1, 2, figsize=(16, 6), sharex=True)
    ax[0].plot(wHz, 20 * np.log10(abs(h)), '-', lw=3)
    ax[1].semilogy(freq, psd, '.-', label='unfiltered')
    ax[1].semilogy(freq_f, psd_f, '.-', label='filtered')
    ax[1].legend(loc='upper left', bbox_to_anchor=(1, 1), shadow=True, numpoints=3)
    format_axes('freq, Hz', 'dB', 'Chebyshev II Bandstop Filter Response', ax[0])
    format_axes('freq, Hz', 'arb', axes=ax[1],
        title='Synthetic data\ntone at {}Hz should be filtered'.format(int(wf * fs / 2)))
    largefonts(16)
    fig.tight_layout()
    fig.subplots_adjust(right=0.8)
    plt.show()

def detect_peaks(x, mph=None, mpd=1, threshold=0, edge='rising',
                 kpsh=False, valley=False, show=False, ax=None):

    """
    Detect peaks in data based on their amplitude and other features.
    http://nbviewer.jupyter.org/github/demotu/BMC/blob/master/notebooks/DetectPeaks.ipynb

    Parameters
    ----------
    x : 1D array_like
        data.
    mph : {None, number}, optional (default = None)
        detect peaks that are greater than minimum peak height.
    mpd : positive integer, optional (default = 1)
        detect peaks that are at least separated by minimum peak distance (in
        number of data).
    threshold : positive number, optional (default = 0)
        detect peaks (valleys) that are greater (smaller) than `threshold`
        in relation to their immediate neighbors.
    edge : {None, 'rising', 'falling', 'both'}, optional (default = 'rising')
        for a flat peak, keep only the rising edge ('rising'), only the
        falling edge ('falling'), both edges ('both'), or don't detect a
        flat peak (None).
    kpsh : bool, optional (default = False)
        keep peaks with same height even if they are closer than `mpd`.
    valley : bool, optional (default = False)
        if True (1), detect valleys (local minima) instead of peaks.
    show : bool, optional (default = False)
        if True (1), plot data in matplotlib figure.
    ax : a matplotlib.axes.Axes instance, optional (default = None).

    Returns
    -------
    ind : 1D array_like
        indeces of the peaks in `x`.

    Notes
    -----
    The detection of valleys instead of peaks is performed internally by simply
    negating the data: `ind_valleys = detect_peaks(-x)`

    The function can handle NaN's

    See this IPython Notebook [1]_.

    References
    ----------
    .. [1] http://nbviewer.ipython.org/github/demotu/BMC/blob/master/notebooks/DetectPeaks.ipynb

    Examples
    --------
    >>> from detect_peaks import detect_peaks
    >>> x = np.random.randn(100)
    >>> x[60:81] = np.nan
    >>> # detect all peaks and plot data
    >>> ind = detect_peaks(x, show=True)
    >>> print(ind)

    >>> x = np.sin(2*np.pi*5*np.linspace(0, 1, 200)) + np.random.randn(200)/5
    >>> # set minimum peak height = 0 and minimum peak distance = 20
    >>> detect_peaks(x, mph=0, mpd=20, show=True)

    >>> x = [0, 1, 0, 2, 0, 3, 0, 2, 0, 1, 0]
    >>> # set minimum peak distance = 2
    >>> detect_peaks(x, mpd=2, show=True)

    >>> x = np.sin(2*np.pi*5*np.linspace(0, 1, 200)) + np.random.randn(200)/5
    >>> # detection of valleys instead of peaks
    >>> detect_peaks(x, mph=0, mpd=20, valley=True, show=True)

    >>> x = [0, 1, 1, 0, 1, 1, 0]
    >>> # detect both edges
    >>> detect_peaks(x, edge='both', show=True)

    >>> x = [-2, 1, -2, 2, 1, 1, 3, 0]
    >>> # set threshold = 2
    >>> detect_peaks(x, threshold = 2, show=True)
    """

    import numpy as np

    x = np.atleast_1d(x).astype('float64')
    if x.size < 3:
        return np.array([], dtype=int)
    if valley:
        x = -x
    # find indices of all peaks
    dx = x[1:] - x[:-1]
    # handle NaN's
    indnan = np.where(np.isnan(x))[0]
    if indnan.size:
        x[indnan] = np.inf
        dx[np.where(np.isnan(dx))[0]] = np.inf
    ine, ire, ife = np.array([[], [], []], dtype=int)
    if not edge:
        ine = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) > 0))[0]
    else:
        if edge.lower() in ['rising', 'both']:
            ire = np.where((np.hstack((dx, 0)) <= 0) & (np.hstack((0, dx)) > 0))[0]
        if edge.lower() in ['falling', 'both']:
            ife = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) >= 0))[0]
    ind = np.unique(np.hstack((ine, ire, ife)))
    # handle NaN's
    if ind.size and indnan.size:
        # NaN's and values close to NaN's cannot be peaks
        ind = ind[np.in1d(ind, np.unique(np.hstack((indnan, indnan-1, indnan+1))), invert=True)]
    # first and last values of x cannot be peaks
    if ind.size and ind[0] == 0:
        ind = ind[1:]
    if ind.size and ind[-1] == x.size-1:
        ind = ind[:-1]
    # remove peaks < minimum peak height
    if ind.size and mph is not None:
        ind = ind[x[ind] >= mph]
    # remove peaks - neighbors < threshold
    if ind.size and threshold > 0:
        dx = np.min(np.vstack([x[ind]-x[ind-1], x[ind]-x[ind+1]]), axis=0)
        ind = np.delete(ind, np.where(dx < threshold)[0])
    # detect small peaks closer than minimum peak distance
    if ind.size and mpd > 1:
        ind = ind[np.argsort(x[ind])][::-1]  # sort ind by peak height
        idel = np.zeros(ind.size, dtype=bool)
        for i in range(ind.size):
            if not idel[i]:
                # keep peaks with the same height if kpsh is True
                idel = idel | (ind >= ind[i] - mpd) & (ind <= ind[i] + mpd) \
                    & (x[ind[i]] > x[ind] if kpsh else True)
                idel[i] = 0  # Keep current peak
        # remove the small peaks and sort back the indices by their occurrence
        ind = np.sort(ind[~idel])

    if show:
        if indnan.size:
            x[indnan] = np.nan
        if valley:
            x = -x
        _plot(x, mph, mpd, threshold, edge, valley, ax, ind)

    return ind

def movingstd(x, k):
    """
    1d or 2d moving standard deviation in a rectangular window
    """

    from pandas import DataFrame
    from numpy import array

    return array(DataFrame(x).rolling(window=k, center=False).std())

def movingaverage(x, k):
    """
    1d or 2d moving average in a rectangular window
    """

    from pandas import DataFrame
    from numpy import array

    return array(DataFrame(x).rolling(window=k, center=False).mean())

def is_writeable(path):
    """
    check if a path is writeable
    """
    import os

    try:
        file = os.path.join(path, r'delete_me.txt')
        fid = open(file, 'w')
        fid.close()
        os.remove(file)
        return True
    except PermissionError:
        return False

def return_dict(x):
    """
    recursive function that converts all defaultdicts to regular dicts
    this is useful as pickle sometimes has problems with defaultdicts
    """

    from collections import defaultdict
    for key in x.keys():
        if isinstance(x[key], defaultdict) or isinstance(x[key], dict):
            x[key] = return_dict(x[key])
    return dict(x)

def search_tree(datadir, pattern, use_pbar=False):
    """
    recursive search in datadir for pattern, with or without a progressbar
    """

    import os
    from glob import glob

    # 1 line solution with no progressbar
    if not use_pbar:
        return glob(os.path.join(datadir, '**', pattern), recursive=True)

    # otherwise must use os.walk to accomodate a progressbar
    from tqdm import tqdm
    from itertools import chain

    # get the toplevel directory list, initialize a 'found' list, counter, and progressbar
    toplevel = [os.path.join(datadir, x) for x in os.listdir(datadir)]
    toplevel = [x for x in toplevel if os.path.isdir(x)]
    found = []
    pbar = tqdm(desc='scanning datadir', total=len(toplevel))

    # walk thru data dir
    for root, _, _ in os.walk(datadir):

        # update the progressbar only for advances in toplevel directories, which could lead
        # to an inaccurate progressbar if the toplevel dirs do not have similar content
        if root in toplevel:
            pbar.update()

        # use glob to find search within root for pattern
        inside = glob(os.path.join(root, pattern))
        if inside:
            found.append(inside)

    # return a flattend list of lists
    return list(chain(*found))

def TIR(data):
    """
    return the total-inclusive range of data, works for 1d data without nans
    """

    import numpy as np

    return np.max(data) - np.min(data)

def longest_substring_from_list(data):
    """
    return the longest common substring from a list of strings, inefficiently
    """

    def is_substr(find, data):
        """ used with longest_substring_from_list """
        if len(data) < 1 and len(find) < 1:
            return False
        for i in range(len(data)):
            if find not in data[i]:
                return False
        return True

    substr = ''
    if len(data) > 1 and len(data[0]) > 0:
        for i in range(len(data[0])):
            for j in range(len(data[0])-i+1):
                if j > len(substr) and is_substr(data[0][i:i+j], data):
                    substr = data[0][i:i+j]
    return substr

def memoized(f):
    """
    Memoization decorator for functions taking one or more arguments
    """
    class memodict(dict):
        def __init__(self, f):
            self.f = f
        def __call__(self, *args):
            return self[args]
        def __missing__(self, key):
            ret = self[key] = self.f(*key)
            return ret
    return memodict(f)

def pngs2ppt(pngs_dir, template=r'c:\pngs2ppt_template.pptx',
             fname=None, author='Author', date=None, title=None,
             add_exec_summary=False, img_width=9, takeaways=None):

    """
    create powerpoint presentation from folder of png image files in pngs_dir
    supported keyword arguments:
    -- template, must point to the template file pngs2ppt_template.pptx
        get this file from the author if you do not have it
    -- fname, filename for output .pptx file, the default is to use the a filename of
        pngs2ppt.pptx in the same folder as pngs_dir
    -- author, name to include on the title page
    -- date, date to include on the title page, default is current date
    -- title, title for the title page, default is pngs2ppt
    -- add_exec_summary, True to add an 'Executive Summary' page to the presentation
    -- img_width, width of png images in the report, default is 9 inches
    -- takeaways, this is text added to each image slide, default is None
        if None, 'Takeaway Message' is added to each image slide
        if provided as a string, this same string is used for each slide
        if provided as a list of strings with length equal to number of png images,
        the corresponding string is added to each slide
    """

    import pptx
    from datetime import datetime
    from PIL import Image
    import os
    from glob import glob

    # set the fname, date, and title, if not provided as keyword arguments
    if fname is None:
        fname = os.path.join(pngs_dir, 'pngs2ppt.pptx')
    if date is None:
        date = datetime.now().strftime(r'%d %b %Y')
    if title is None:
        title = 'Created with python-pptx {}'.format(pptx.__version__)

    # get all png files in a list, return if empty
    pngs = glob(os.path.join(pngs_dir, r'*.png'))
    if len(pngs) == 0:
        print('No png images found, returning without creating a presentation')
        return

    # open a presentation object from the template file, save as fname before proceeding
    prs = pptx.Presentation(template)
    prs.save(fname)

    # add a title slide with title, author, date
    title_slide = prs.slides.add_slide(prs.slide_layouts[0])
    title_slide.shapes.title.text = title
    title_slide.placeholders[1].text = '{}\n\n{}'.format(author, date)

    # add an executive summary if requested
    if add_exec_summary:
        exec_slide = prs.slides.add_slide(prs.slide_layouts[1])
        exec_slide.shapes.title.text = 'Executive Summary'
        exec_slide.placeholders[1].text = '...'
        exec_slide.placeholders[13].text = 'Takeaway'

    # get the width, height of a slide, get the native unit corresponding to 1 inch
    width, height = prs.slide_width, prs.slide_height
    inch = pptx.util.Inches(1)

    # process the takeaways keyword argument at this point
    if isinstance(takeaways, str):
        takeaways = [takeaways for _ in range(len(pngs))]
    elif isinstance(takeaways, list) and len(takeaways) == len(pngs):
        pass
    else:
        takeaways = ['Takeaway Message' for _ in range(len(pngs))]

    # add an image slide for each png file
    for png, takeaway in zip(pngs, takeaways):

        # get size of the png image
        png_width_nom, png_height_nom = Image.open(png).size

        # set the desired image width, and height to preserve the original aspect ratio
        png_width_actual = img_width * inch
        png_height_actual = png_height_nom * png_width_actual / png_width_nom

        # set the 'top' and 'left' keyword arguments to center the image
        top = int((height - png_height_actual) / 2)
        left = int((width - png_width_actual) / 2)

        # add the slide with the centered png image
        png_slide = prs.slides.add_slide(prs.slide_layouts[2])
        png_slide.shapes.title.text = os.path.split(png)[1][:-4]
        png_slide.shapes.add_picture(png, left=left, top=top, width=png_width_actual)

        # add the takeaway message
        png_slide.placeholders[13].text = takeaway
        #set_trace()

    # final save, print save location
    prs.save(fname)
    print('Presentation saved to {}'.format(fname))
