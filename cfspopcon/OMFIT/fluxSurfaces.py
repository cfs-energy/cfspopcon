# try:
#     # framework is running
#     from .startup_choice import *
# except ImportError as _excp:
#     # class is imported by itself
#     if (
#         'attempted relative import with no known parent package' in str(_excp)
#         or 'No module named \'omfit_classes\'' in str(_excp)
#         or "No module named '__main__.startup_choice'" in str(_excp)
#     ):
#         from startup_choice import *
#     else:
#         raise

#import omfit_classes.utils_math
#from omfit_classes.utils_math import (
    #parabola,
    #paraboloid,
    #parabolaMaxCycle,
    #contourPaths,
    #reverse_enumerate,
    #RectBivariateSplineNaN,
    #deriv,
    #line_intersect,
    #interp1e,
    #centroid,
    #pack_points,
#)
import time
import copy
import functools
def dynaLoad(f):
    """
    Decorator which calls `dynaLoader` method

    :param f: function to decorate

    :return: decorated function
    """

    @functools.wraps(f)
    def dynamicLoading(self, *args, **kw):
        dynaLoader(self, f)
        return f(self, *args, **kw)

    return dynamicLoading

class _OMFITauxiliary(dict):
    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls, *args, **kwargs)
        instance.defaults = {}
        return instance

    def __init__(self):
        self.defaults['lastUserError'] = ['']
        self.defaults['lastReportedUserError'] = ['']

        self.defaults['lastBrowsedDirectory'] = ''
        self.defaults['lastBrowsed'] = {}

        self.defaults['GUI'] = None
        self.defaults['rootGUI'] = None
        self.defaults['treeGUI'] = None
        self.defaults['console'] = None
        self.defaults['virtualKeys'] = False

        self.defaults['hardLinks'] = False

        self.defaults['quickPlot'] = {}

        self.defaults['prun_process'] = []
        self.defaults['prun_nprocs'] = []
        self.defaults['pythonRunWindows'] = []
        self.defaults['haltWindow'] = None

        self.defaults['MDSserverReachable'] = {}
        self.defaults['RDBserverReachable'] = {}
        self.defaults['batch_js'] = {}
        self.defaults['sysinfo'] = {}
        self.defaults['sshTunnel'] = {}

        self.defaults['lastActivity'] = time.time()

        self.defaults['noCopyToCWD'] = False

        self.defaults['lastRunModule'] = ''
        self.defaults['moduleSkeletonCache'] = None

        self.defaults['debug'] = 0

        self.defaults['dynaLoad_switch'] = True

        self.update(copy.deepcopy(self.defaults))

    def __getitem__(self, key):
        # if 'lastBrowsedDirectory' does not exisits recurse directory backwards to find valid directory root
        if key == 'lastBrowsedDirectory' and key in self:
            tmp = super().__getitem__(key)
            tmp = os.path.abspath(os.path.expandvars(os.path.expanduser(tmp)))
            for k in range(len(tmp)):
                if os.path.exists(tmp):
                    return tmp
                tmp = os.path.split(tmp)[0]
            return os.environ['HOME']
        return super().__getitem__(key)


OMFITaux = _OMFITauxiliary()

def dynaLoader(self, f=None):
    """
    Call `load` function if object has `dynaLoad` attribute set to True
    After calling `load` function the `dynaLoad` attribute is set to False
    """
    if OMFITaux['dynaLoad_switch'] and self.dynaLoad:
        self.dynaLoad = False
        if f is None or f.__name__ != 'load':
            try:
                return self.load()
            except Exception as _excp_load:
                # If an error occurs during loading
                # Clear and reset the dynamic load switch to allow re-tries
                # note: errors could occur because user stops the process
                try:
                    self.clear()
                except Exception as _excp_clear:
                    # if clear() fails, then its exception should be printed but not raised.
                    # the user is interested in the original exception raised by load().
                    printe('The clear() method raised an exception: ' + repr(_excp_clear))
                self.dynaLoad = True
                raise
import types
import traceback

class SortedDict(dict):
    # originally inspired from django/trunk/django/utils/datastructures.py @ 17464
    r"""
    A dictionary that keeps its keys in the order in which they're inserted

    :param data: A dict object or list of (key,value) tuples from which to initialize the new SortedDict object

    :param \**kw: Optional keyword arguments given below

    kw:
        :param caseInsensitive: (bool)  If True, allows self['UPPER'] to yield self['upper'].

        :param sorted: (bool) If True, keep keys sorted alphabetically, instead of by insertion order.

        :param limit: (int) keep only the latest `limit` number of entries (useful for data cashes)

        :param dynaLoad: (bool) Not sure what this does
    """

    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls, *args, **kwargs)
        instance._OMFITkeyName = ''
        instance._OMFITparent = None
        instance.keyOrder = []
        instance.caseInsensitive = kwargs.pop('caseInsensitive', False)
        instance.sorted = kwargs.pop('sorted', False)
        instance.limit = kwargs.pop('limit', 0)
        instance.dynaLoad = False
        return instance

    def __init__(self, data=None, *args, **kwargs):
        self.clear()
        self.caseInsensitive = kwargs.pop('caseInsensitive', False)
        self.sorted = kwargs.pop('sorted', False)
        self.limit = kwargs.pop('limit', 0)
        self.dynaLoad = False

        if data is None:
            data = {}
        elif isinstance(data, types.GeneratorType):
            # Unfortunately we need to be able to read a generator twice.  Once
            # to get the data into self with our super().__init__ call and a
            # second time to setup keyOrder correctly
            data = list(data)

        if isinstance(data, dict):
            for key in list(data.keys()):
                self[key] = data[key]
        else:
            for key, value in data:
                self[key] = value

        if self.sorted:
            self.sort()

        self.dynaLoad = kwargs.pop('dynaLoad', False)

        # keep track of what classes have been loaded
        from omfit_classes.utils_base import _loaded_classes

        _loaded_classes.add(self.__class__.__name__)

    def __getattr__(self, attr):
        if attr.startswith('_OMFIT') or attr.startswith('OMFIT'):
            raise AttributeError('bad attribute `%s`' % attr)
        if (
            OMFITaux['dynaLoad_switch']
            and self.dynaLoad
            and attr not in ['__save_kw__', '__tree_repr__', 'modifyOriginal', 'readOnly']
            and 'getattr_infiniteloop_block' not in self.__dict__
        ):
            try:
                if os.environ['USER'] == 'meneghini':
                    print('%s dynaloading because %s attribute was requested' % (self.__class__.__name__, attr), file=sys.__stderr__)
                    traceback.print_stack(file=sys.__stderr__)
                self.__dict__['getattr_infiniteloop_block'] = True
                dynaLoader(self)
                return getattr(self, attr)
            finally:
                del self.__dict__['getattr_infiniteloop_block']

        raise AttributeError('bad attribute `%s`' % attr)


from scipy import interpolate,integrate
import datetime
class interp1e(interpolate.interp1d):
    """
    Shortcut for scipy.interpolate.interp1d with fill_value='extrapolate' and bounds_error=False as defaults
    """

    __doc__ += (
        interpolate.interp1d.__doc__.replace('\n    ----------\n', ':\n')
        .replace('\n    -------\n', ':\n')
        .replace('\n    --------\n', ':\n')
    )

    def __init__(self, x, y, *args, **kw):
        kw.setdefault('fill_value', 'extrapolate')
        kw.setdefault('bounds_error', False)
        interpolate.interp1d.__init__(self, x, y, *args, **kw)

def _seg_intersect(a1, a2, b1, b2):
    if not _intersect(a1, a2, b1, b2):
        return None
    da = a2 - a1
    db = b2 - b1
    dp = a1 - b1
    dap = _perp(da)
    denom = np.dot(dap, db)
    num = np.dot(dap, dp)
    return (num / denom) * db + b1

def line_intersect(path1, path2, return_indices=False):
    """
    intersection of two 2D paths

    :param path1: array of (x,y) coordinates defining the first path

    :param path2: array of (x,y) coordinates defining the second path

    :param return_indices: return indices of segments where intersection occurred

    :return: array of intersection points (x,y)
    """

    warnings.filterwarnings('ignore', category=RuntimeWarning, message='invalid value.*')

    ret = []
    index = []

    path1 = np.array(path1).astype(float)
    path2 = np.array(path2).astype(float)

    switch = False
    if len(path1[:, 0]) > len(path2[:, 0]):
        path1, path2 = path2, path1
        switch = True

    m1 = (path1[:-1, :] + path1[1:, :]) / 2.0
    m2 = (path2[:-1, :] + path2[1:, :]) / 2.0
    l1 = np.sqrt(np.diff(path1[:, 0]) ** 2 + np.diff(path1[:, 1]) ** 2)
    l2 = np.sqrt(np.diff(path2[:, 0]) ** 2 + np.diff(path2[:, 1]) ** 2)

    for k1 in range(len(path1) - 1):
        d = np.sqrt((m2[:, 0] - m1[k1, 0]) ** 2 + (m2[:, 1] - m1[k1, 1]) ** 2)
        for k2 in np.where(d <= (l2 + l1[k1]) / 2.0)[0]:
            tmp = _seg_intersect(path1[k1], path1[k1 + 1], path2[k2], path2[k2 + 1])
            if tmp is not None:
                index.append([k1, k2])
                ret.append(tmp)

    ret = np.array(ret)

    if not len(ret):
        return []

    if switch:
        index = np.array(index)
        i = np.argsort(index[:, 1])
        index = [index[i, 1], index[i, 0]]
        index = np.array(index).T.astype(int).tolist()
        ret = ret[i]

    if return_indices:
        return ret, index

    return ret

import scipy
def contourPaths(x, y, Z, levels, remove_boundary_points=False, smooth_factor=1):
    """
    :param x: 1D x coordinate

    :param y: 1D y coordinate

    :param Z: 2D data

    :param levels: levels to trace

    :param remove_boundary_points: remove traces at the boundary

    :param smooth_factor: smooth contours by cranking up grid resolution

    :return: list of segments
    """
    import matplotlib
    import contourpy as _contour

    sf = int(round(smooth_factor))
    if sf > 1:
        x = scipy.ndimage.zoom(x, sf)
        y = scipy.ndimage.zoom(y, sf)
        Z = scipy.ndimage.zoom(Z, sf)

    [X, Y] = np.meshgrid(x, y)
    contour_generator = _contour.contour_generator(X, Y, Z, corner_mask=None, chunk_count=0)

    mx = min(x)
    Mx = max(x)
    my = min(y)
    My = max(y)

    allsegs = []
    for level in levels:
        verts = contour_generator.create_contour(level)# Is this right?
        # if compare_version(matplotlib.__version__, '3.4.3') > 0:
        #     """After MPL ver 3.4.3 the return type is vertices and codes as a list. Prior it was just vertices"""
        #     segs = verts[0]
        # else:
        segs = verts
        if not remove_boundary_points:
            segs_ = segs
        else:
            segs_ = []
            for segarray in segs:
                segarray = np.array(segarray)
                x_ = segarray[:, 0]
                y_ = segarray[:, 1]
                valid = []
                for i in range(len(x_) - 1):
                    if np.isclose(x_[i], x_[i + 1]) and (np.isclose(x_[i], Mx) or np.isclose(x_[i], mx)):
                        continue
                    if np.isclose(y_[i], y_[i + 1]) and (np.isclose(y_[i], My) or np.isclose(y_[i], my)):
                        continue
                    valid.append((x_[i], y_[i]))
                    if i == len(x_):
                        valid.append(x_[i + 1], y_[i + 1])
                if len(valid):
                    segs_.append(np.array(valid))

        segs = list(map(matplotlib.path.Path, segs_))
        allsegs.append(segs)
    return allsegs

import warnings
def parabola(x, y):
    """
    y = a*x^2 + b*x + c

    :return: a,b,c
    """
    if np.any(np.isnan(x)) or np.any(np.isnan(y)):
        raise np.linalg.LinAlgError('parabola could not be fit with x=%s y=%s' % (x, y))
    # A=np.matrix([[x[0]**2,x[0],1],[x[1]**2,x[1],1],[x[2]**2,x[2],1]])
    # a,b,c=np.array(A.I*np.matrix(y[0:3]).T).flatten().tolist()
    # polyfit is equally fast but more robust when x values are similar, and matrix A becomes singular, eg.
    # x,y=[-1E-16,1,1+1E-16],[1,1,1]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', np.RankWarning)
        a, b, c = np.polyfit(x, y, 2)
    return a, b, c

def parabolaMax(x, y, bounded=False):
    """
    Calculate a parabola through x,y, then return the extremum point of
    the parabola

    :param x: At least three abcissae points
    :param y: The corresponding ordinate points
    :param bounded: False, 'max', or 'min'
          - False: The extremum is returned regardless of location relative to x (default)
          - 'max' ('min'): The extremum location must be within the bounds of x, and if not return the location and value of max(y) (min(y))
    :return: x_max,y_max - The location and value of the extremum
    """
    a, b, c = parabola(x, y)
    x_max = -b / (2 * a)
    y_max = a * x_max**2 + b * x_max + c
    if bounded and (x_max > max(x) or x_max < min(x)):
        printe('ParabolaMax found the maximum outside the input abcissae; using arg%s(y) instead' % bounded)
        iy = eval('arg%s(y)' % bounded)
        x_max = x[iy]
        y_max = y[iy]
    return x_max, y_max

def parabolaCycle(X, Y, ix):
    ix = np.array([-1, 0, 1]) + ix

    if (X[0] - X[-1]) < 1e-15:
        if ix[0] < 0:
            ix[0] = len(X) - 1 - 1
        if ix[-1] > (len(X) - 1):
            ix[-1] = 0 + 1
    else:
        if ix[0] < 0:
            ix[0] = len(X) - 1
        if ix[-1] > (len(X) - 1):
            ix[-1] = 0

    return parabola(X[ix], Y[ix])


def parabolaMaxCycle(X, Y, ix, bounded=False):
    """
    Calculate a parabola through X[ix-1:ix+2],Y[ix-1:ix+2], with proper
    wrapping of indices, then return the extremum point of the parabola

    :param X: The abcissae points: an iterable to be treated as periodic
    :param y: The corresponding ordinate points
    :param ix: The index of X about which to find the extremum
    :param bounded: False, 'max', or 'min'
          - False: The extremum is returned regardless of location relative to x (default)
          - 'max' ('min'): The extremum location must be within the bounds of x, and if not return the location and value of max(y) (min(y))
    :return: x_max,y_max - The location and value of the extremum
    """
    ix = np.array([-1, 0, 1]) + ix

    if (X[0] - X[-1]) < 1e-15:
        if ix[0] < 0:
            ix[0] = len(X) - 1 - 1
        if ix[-1] > (len(X) - 1):
            ix[-1] = 0 + 1
    else:
        if ix[0] < 0:
            ix[0] = len(X) - 1
        if ix[-1] > (len(X) - 1):
            ix[-1] = 0

    try:
        return parabolaMax(X[ix], Y[ix], bounded=bounded)
    except Exception:
        printe([ix, X[ix], Y[ix]])
        return X[ix[1]], Y[ix[1]]

class RectBivariateSplineNaN:
    def __init__(self, Z, R, Q, *args, **kw):
        tmp = Q.copy()
        bad = np.isnan(tmp)
        self.thereAreNaN = False
        if bad.any():
            self.thereAreNaN = True
            tmp[bad] = 0
            self.mask = interpolate.RectBivariateSpline(Z, R, bad, kx=1, ky=1)
        self.spline = interpolate.RectBivariateSpline(Z, R, tmp, *args, **kw)

    def __call__(self, *args):
        tmp = self.spline(*args)
        if self.thereAreNaN:
            mask = self.mask(*args)
            tmp[mask > 0.01] = np.nan
        return tmp

    def ev(self, *args):
        tmp = self.spline.ev(*args)
        if self.thereAreNaN:
            mask = self.mask.ev(*args)
            tmp[mask > 0.01] = np.nan
        return tmp

def reverse_enumerate(l):
    import itertools

    return zip(range(len(l) - 1, -1, -1), reversed(l))

def centroid(x, y):
    """
    Calculate centroid of polygon

    :param x: x coordinates of the polygon

    :param y: y coordinates of the polygon

    :return: tuple with x and y coordinates of centroid
    """
    x = np.array(x)
    y = np.array(y)
    dy = np.diff(y)
    dx = np.diff(x)
    x0 = (x[1:] + x[:-1]) * 0.5
    y0 = (y[1:] + y[:-1]) * 0.5
    A = np.sum(dy * x0)
    x_c = -np.sum(dx * y0 * x0) / A
    y_c = np.sum(dy * x0 * y0) / A
    return x_c, y_c

#from omfit_classes.utils_base import printw
import os
import sys
from io import StringIO
class _Streams(dict):
    tags = {
        'STDOUT': ('black', 'RESET'),
        'STDERR': ('red3', 'RED'),
        'DEBUG': ('gold4', 'YELLOW'),
        'PROGRAM_OUT': ('blue', 'BLUE'),
        'PROGRAM_ERR': ('purple', 'MAGENTA'),
        'INFO': ('forest green', 'GREEN'),
        'WARNING': ('DarkOrange2', 'YELLOW'),
        'HIST': ('dark slate gray', 'CYAN'),
        'HELP': ('PaleGreen4', 'CYAN'),
    }

    def __new__(cls):
        return dict.__new__(cls)

    def __init__(self):
        self.stderr = sys.stderr
        self.stdout = sys.stdout
        self.setDefaults()
        self.bkpStack = []

    def setDefaults(self):
        for k in self.tags:
            if 'ERR' in k:
                self[k] = self.stderr
            else:
                self[k] = self.stdout

    def backup(self):
        self.bkpStack.append(list(self.items()))

    def restore(self):
        for k, v in self.bkpStack.pop():
            self[k] = v
        sys.stdout = self['STDOUT']
        sys.stderr = self['STDERR']


_streams = _Streams()

class console_color:
    @staticmethod
    def BLACK(x=''):
        return '\033[30m' + str(x) + '\033[0m'

    @staticmethod
    def RED(x=''):
        return '\033[31m' + str(x) + '\033[0m'

    @staticmethod
    def GREEN(x=''):
        return '\033[32m' + str(x) + '\033[0m'

    @staticmethod
    def YELLOW(x=''):
        return '\033[33m' + str(x) + '\033[0m'

    @staticmethod
    def BLUE(x=''):
        return '\033[34m' + str(x) + '\033[0m'

    @staticmethod
    def MAGENTA(x=''):
        return '\033[35m' + str(x) + '\033[0m'

    @staticmethod
    def CYAN(x=''):
        return '\033[36m' + str(x) + '\033[0m'

    @staticmethod
    def WHITE(x=''):
        return '\033[37m' + str(x) + '\033[0m'

    @staticmethod
    def UNDERLINE(x=''):
        return '\033[4m' + str(x) + '\033[0m'

    @staticmethod
    def RESET(x=''):
        return '\033[0m' + str(x) + '\033[0m'
    
def tag_print(*objects, **kw):
    """
    Works like the print function, but used to print to GUI (if GUI is available).
    The coloring of the GUI print is determined by the `tag` parameter.

    :param \*objects: string/objects to be printed

    :param sep: separator (default: ' ')

    :param sep: new line character (default: '\\\\n')

    :param tag: one of the following:

        * 'STDOUT'
        * 'STDERR'
        * 'DEBUG'
        * 'PROGRAM_OUT'
        * 'PROGRAM_ERR'
        * 'INFO'
        * 'WARNING'
        * 'HIST'
        * 'HELP'
    """
    tag = kw.get('tag', '')
    tag_override = os.environ.get('OMFIT_TAG_PRINT_STREAM_OVERRIDE', '')
    if tag_override in _streams:
        tag = tag_override
    if sys.stdout is not sys.__stdout__ and tag in _streams:  # <--- check if the stdout is redirected and tag is recognized
        return print(*objects, sep=kw.pop('sep', ' '), end=kw.pop('end', '\n'), file=_streams[tag])
    else:
        file = sys.__stderr__ if 'ERR' in tag else sys.__stdout__
        # colored terminal
        if sys.stdout.isatty():
            tmp = StringIO()
            print(*objects, sep=kw.pop('sep', ' '), end=kw.pop('end', '\n'), file=tmp)
            return print(getattr(console_color, _streams.tags[tag][1])(tmp.getvalue()), sep='', end='', file=file)
        # standard terminal
        return print(*objects, sep=kw.pop('sep', ' '), end=kw.pop('end', '\n'), file=file)
    
def printw(*objects, **kw):
    """
    Function to print with `WARNING` style

    :param \*objects: what to print

    :param \**kw: keywords passed to the `print` function

    :return: return from `print` function
    """
    kw['tag'] = 'WARNING'
    if int(os.environ.get('OMFIT_VISUAL_CUES', '0')):
        objects = ['@' + '\n@'.join(str(x).splitlines()) for x in objects]
    return tag_print(*objects, **kw)

import omas
import numpy as np
from matplotlib.path import Path
from scipy import integrate, interpolate, constants
from omas import ODS, omas_environment

__all__ = [
    'fluxSurfaces',
    'fluxSurfaceTraces',
    'boundaryShape',
    'BoundaryShape',
    'fluxGeo',
    'rz_miller',
    'miller_derived',
]


class fluxSurfaceTraces(SortedDict):
    def deploy(self, filename=None, frm='arrays'):
        import omfit_classes.namelist

        if filename is None:
            printi('Specify filename to deploy fluxSurfaceTraces')
            return
        if frm.lower() == 'namelist':
            tmp = NamelistFile()
            tmp.filename = filename
            nn = namelist.NamelistName()
            nn['Nsurf'] = len(self)
            npts = []
            for k in self:
                npts.append(len(self[k]['R']))
            nn['Npoints'] = npts
            tmp['dimensions'] = nn
            for k in self:
                nn = namelist.NamelistName()
                nn['psi'] = self[k]['psi']
                nn['q'] = self[k]['q']
                nn['P'] = self[k]['P']
                nn['R'] = self[k]['R']
                nn['Z'] = self[k]['Z']
                nn['l'] = np.hstack((0, np.sqrt(np.diff(self[k]['R']) ** 2 + np.diff(self[k]['Z']) ** 2)))
                tmp['flux_' + str(k)] = nn
            tmp.save()
        else:
            R = np.array([])
            Z = np.array([])
            l = np.array([])
            psi = np.array([])
            q = np.array([])
            P = np.array([])
            npts = np.array([])
            for k in self:
                npts = np.hstack((npts, len(self[k]['R'])))
                psi = np.hstack((psi, self[k]['psi']))
                q = np.hstack((q, self[k]['q']))
                P = np.hstack((P, self[k]['P']))
                l = np.hstack((l, np.hstack((0, np.sqrt(np.diff(self[k]['R']) ** 2 + np.diff(self[k]['Z']) ** 2)))))
                Z = np.hstack((Z, self[k]['Z']))
                R = np.hstack((R, self[k]['R']))
            with open(filename, 'w') as f:
                f.write(
                    '''
#line (1) number of flux surfaces
#line (2) number of points per flux surface
#line (3) psi for each flux surface
#line (4) q for each flux surface
#line (5) pressure for each flux surface
#line (6) R for each flux surface
#line (7) Z for each flux surface
#line (8) l for each flux surface
'''.strip()
                    + '\n'
                )
                savetxt(f, np.array([len(self)]), fmt='%d')
                savetxt(f, np.reshape(npts, (1, -1)), fmt='%d')
                savetxt(f, np.reshape(psi, (1, -1)))
                savetxt(f, np.reshape(q, (1, -1)))
                savetxt(f, np.reshape(P, (1, -1)))
                savetxt(f, np.vstack((R, Z, l)))

    def load(self, filename):
        with open(filename, 'r') as f:
            lines = f.readlines()
        k = 9
        n = np.fromstring(lines[k], dtype=int, sep=' ')
        k += 1
        psi = np.fromstring(lines[k], dtype=float, sep=' ')
        k += 1
        q = np.fromstring(lines[k], dtype=float, sep=' ')
        k += 1
        P = np.fromstring(lines[k], dtype=float, sep=' ')
        k += 1
        r = np.fromstring(lines[k], dtype=float, sep=' ')
        k += 1
        z = np.fromstring(lines[k], dtype=float, sep=' ')
        k += 1
        l = np.fromstring(lines[k], dtype=float, sep=' ')
        k = 0
        for start, stop in zip(*(np.cumsum(n) - n, np.cumsum(n))):
            self[k] = SortedDict()
            self[k]['psi'] = psi[k]
            self[k]['q'] = q[k]
            self[k]['P'] = P[k]
            self[k]['R'] = r[start:stop]
            self[k]['Z'] = z[start:stop]
            self[k]['l'] = l[start:stop]
            k = k + 1
        return self


def fluxGeo(inputR, inputZ, lcfs=False, doPlot=False):
    '''
    Calculate geometric properties of a single flux surface

    :param inputR: R points

    :param inputZ: Z points

    :param lcfs: whether this is the last closed flux surface (for sharp feature of x-points)

    :param doPlot: plot geometric measurements

    :return: dictionary with geometric quantities
    '''
    # Cast as arrays
    inputR = np.array(inputR)
    inputZ = np.array(inputZ)

    # Make sure the flux surfaces close
    if inputR[0] != inputR[1]:
        inputRclose = np.hstack((inputR, inputR[0]))
        inputZclose = np.hstack((inputZ, inputZ[0]))
    else:
        inputRclose = inputR
        inputZclose = inputZ
        inputR = inputR[:-1]
        inputR = inputZ[:-1]

    # This is the result
    geo = SortedDict()

    # These are the extrema indices
    imaxr = np.argmax(inputR)
    iminr = np.argmin(inputR)
    imaxz = np.argmax(inputZ)
    iminz = np.argmin(inputZ)

    # Test for lower null
    lonull = False
    # Find the slope on either side of min(z)
    ind1 = (iminz + 1) % len(inputZ)
    ind2 = (iminz + 2) % len(inputZ)
    loslope1 = (np.diff(inputZ[[ind1, ind2]]) / np.diff(inputR[[ind1, ind2]]))[0]
    ind1 = (iminz - 1) % len(inputZ)
    ind2 = (iminz - 2) % len(inputZ)
    loslope2 = (np.diff(inputZ[[ind1, ind2]]) / np.diff(inputR[[ind1, ind2]]))[0]
    # If loslope1*loslope2==-1, then it is a perfect x-point, test against -0.5 as a threshold
    if loslope1 * loslope2 < -0.5 and len(inputZ) > 20:
        lcfs = lonull = True
    geo['lonull'] = lonull

    # Test for upper null
    upnull = False
    # Find the slope on either side of max(z)
    ind1 = (imaxz + 1) % len(inputZ)
    ind2 = (imaxz + 2) % len(inputZ)
    upslope1 = (np.diff(inputZ[[ind1, ind2]]) / np.diff(inputR[[ind1, ind2]]))[0]
    ind1 = (imaxz - 1) % len(inputZ)
    ind2 = (imaxz - 2) % len(inputZ)
    upslope2 = (np.diff(inputZ[[ind1, ind2]]) / np.diff(inputR[[ind1, ind2]]))[0]
    # If upslope1*upslope2==-1, then it is a perfect x-point, test against -0.5 as a threshold
    if upslope1 * upslope2 < -0.5 and len(inputZ) > 20:
        lcfs = upnull = True
    geo['upnull'] = upnull

    # Find the extrema points
    if lcfs:
        r_at_max_z, max_z = inputR[imaxz], inputZ[imaxz]
        r_at_min_z, min_z = inputR[iminz], inputZ[iminz]
        z_at_max_r, max_r = inputZ[imaxr], inputR[imaxr]
        z_at_min_r, min_r = inputZ[iminr], inputR[iminr]
    else:
        r_at_max_z, max_z = parabolaMaxCycle(inputR, inputZ, imaxz, bounded='max')
        r_at_min_z, min_z = parabolaMaxCycle(inputR, inputZ, iminz, bounded='min')
        z_at_max_r, max_r = parabolaMaxCycle(inputZ, inputR, imaxr, bounded='max')
        z_at_min_r, min_r = parabolaMaxCycle(inputZ, inputR, iminr, bounded='min')

    dl = np.sqrt(np.ediff1d(inputR, to_begin=0) ** 2 + np.ediff1d(inputZ, to_begin=0) ** 2)
    geo['R'] = 0.5 * (max_r + min_r)
    geo['Z'] = 0.5 * (max_z + min_z)
    geo['a'] = 0.5 * (max_r - min_r)
    geo['eps'] = geo['a'] / geo['R']
    geo['per'] = np.sum(dl)
    geo['surfArea'] = 2 * np.pi * np.sum(inputR * dl)
    geo['kap'] = 0.5 * ((max_z - min_z) / geo['a'])
    geo['kapu'] = (max_z - z_at_max_r) / geo['a']
    geo['kapl'] = (z_at_max_r - min_z) / geo['a']
    geo['delu'] = (geo['R'] - r_at_max_z) / geo['a']
    geo['dell'] = (geo['R'] - r_at_min_z) / geo['a']
    geo['delta'] = 0.5 * (geo['dell'] + geo['delu'])
    geo['zoffset'] = z_at_max_r

    # squareness (Luce,T.C. PPCF 55.9 2013 095009)
    for quadrant in ['zetaou', 'zetaol', 'zetaiu', 'zetail']:
        if quadrant == 'zetaou':
            R0 = r_at_max_z
            Z0 = z_at_max_r
            r = inputR - R0
            z = inputZ - Z0
            A = max_r - R0
            B = max_z - Z0
            th = 3 * np.pi / 2.0 + np.arctan(B / A)
            i = np.where((r <= 0) | (z <= 0))
        elif quadrant == 'zetaol':
            R0 = r_at_min_z
            Z0 = z_at_max_r
            r = inputR - R0
            z = inputZ - Z0
            A = max_r - R0
            B = -(Z0 - min_z)
            th = 3 * np.pi / 2.0 + np.arctan(B / A)
            i = np.where((r <= 0) | (z >= 0))
        elif quadrant == 'zetaiu':
            R0 = r_at_max_z
            Z0 = z_at_max_r
            r = inputR - R0
            z = inputZ - Z0
            A = -(R0 - min_r)
            B = max_z - Z0
            th = 1 * np.pi / 2.0 + np.arctan(B / A)
            i = np.where((r >= 0) | (z <= 0))
        elif quadrant == 'zetail':
            R0 = r_at_min_z
            Z0 = z_at_max_r
            r = inputR - R0
            z = inputZ - Z0
            A = -(R0 - min_r)
            B = -(Z0 - min_z)
            th = 1 * np.pi / 2.0 + np.arctan(B / A)
            i = np.where((r >= 0) | (z >= 0))

        # remove points in other quadrants
        r[i] = np.nan
        z[i] = np.nan
        if np.sum(~np.isnan(r)) < 2 or np.sum(~np.isnan(z)) < 2:
            continue

        # rotate points by 45 degrees
        r1 = r * np.cos(th) + z * np.sin(th)
        z1 = -r * np.sin(th) + z * np.cos(th)

        # remove points in lower half midplane
        r1[np.where(z1 < 0)] = np.nan
        z1[np.where(z1 < 0)] = np.nan
        if np.sum(~np.isnan(r1)) < 2 or np.sum(~np.isnan(z1)) < 2:
            continue

        # fit a parabola to find location of max shifted-rotated z
        ix = np.nanargmin(abs(r1))
        try:
            ixp = np.mod(np.array([-1, 0, 1]) + ix, len(r1))
            a, b, D = parabola(r1[ixp], z1[ixp])
        except np.linalg.LinAlgError:
            D = interp1e(r1[~np.isnan(r1)], z1[~np.isnan(z1)])(0)

        # ellipsis
        C = np.sqrt((A * np.cos(np.pi / 4.0)) ** 2 + (B * np.sin(np.pi / 4.0)) ** 2)
        # square
        E = np.sqrt(A**2 + B**2)
        # squareness
        geo[quadrant] = (D - C) / (E - C)

        # Testing
        if doPlot:
            t = arctan(z / r * np.sign(A) * np.sign(B))
            pyplot.plot(R0 + r, Z0 + z)
            pyplot.gca().set_aspect('equal')
            pyplot.plot(R0 + D * np.sin(-th), Z0 + D * np.cos(-th), 'o')
            pyplot.plot(R0 + np.array([0, A]), Z0 + np.array([0, B]))
            C = np.sqrt((np.cos(-th) / B) ** 2 + (np.sin(-th) / A) ** 2)
            pyplot.plot(R0 + A * np.cos(t), Z0 + B * np.sin(t), '--')
            pyplot.plot(R0 + A, Z0 + B, 'd')
            pyplot.plot(R0 + A * np.cos(np.pi / 4.0), Z0 + B * np.sin(np.pi / 4.0), 's')
            pyplot.plot(R0 + np.array([0, A, A, 0, 0]), Z0 + np.array([0, 0, B, B, 0]), 'k')
            pyplot.gca().set_frame_on(False)

    def rmin_rmax_at_z(r, z, z_c):
        tmp = line_intersect(np.array([r, z]).T, np.array([[np.min(r), np.max(r)], [z_c, z_c]]).T)
        return tmp[0, 0], tmp[1, 0]

    r_c, z_c = centroid(inputR, inputZ)
    r_m, r_p = rmin_rmax_at_z(inputRclose, inputZclose, z_c)
    geo['R_centroid'] = r_c
    geo['Z_centroid'] = z_c
    geo['Rmin_centroid'] = r_m
    geo['Rmax_centroid'] = r_p

    rmaj = (r_p + r_m) / 2.0
    rmin = (r_p - r_m) / 2.0
    r_s = rmaj + rmin * np.cos(0.25 * np.pi + np.arcsin(geo['delta']) / np.sqrt(2.0))
    z_p, z_m = rmin_rmax_at_z(inputZ, inputR, r_s)
    geo['zeta'] = -0.25 * np.pi + 0.5 * (np.arcsin((z_p - z_c) / (geo['kap'] * rmin)) + np.arcsin((z_c - z_m) / (geo['kap'] * rmin)))

    return geo


class fluxSurfaces(SortedDict):
    """
    Trace flux surfaces and calculate flux-surface averaged and geometric quantities
    Inputs can be tables of PSI and Bt or an OMFITgeqdsk file
    """

    def __init__(
        self,
        Rin=None,
        Zin=None,
        PSIin=None,
        Btin=None,
        Rcenter=None,
        F=None,
        P=None,
        rlim=None,
        zlim=None,
        gEQDSK=None,
        resolution=0,
        forceFindSeparatrix=True,
        levels=None,
        map=None,
        maxPSI=1.0,
        calculateAvgGeo=True,
        cocosin=None,
        quiet=False,
        **kw,
    ):
        r"""
        :param Rin: (ignored if gEQDSK!=None) array of the R grid mesh

        :param Zin: (ignored if gEQDSK!=None) array of the Z grid mesh

        :param PSIin: (ignored if gEQDSK!=None) PSI defined on the R/Z grid

        :param Btin: (ignored if gEQDSK!=None) Bt defined on the R/Z grid

        :param Rcenter: (ignored if gEQDSK!=None) Radial location where the vacuum field is defined ( B0 = F[-1] / Rcenter)

        :param F: (ignored if gEQDSK!=None) F-poloidal

        :param P: (ignored if gEQDSK!=None) pressure

        :param rlim: (ignored if gEQDSK!=None) array of limiter r points (used for SOL)

        :param zlim: (ignored if gEQDSK!=None) array of limiter z points (used for SOL)

        :param gEQDSK: OMFITgeqdsk file or ODS

        :param resolution: if `int` the original equilibrium grid will be multiplied by (resolution+1), if `float` the original equilibrium grid is interpolated to that resolution (in meters)

        :param forceFindSeparatrix: force finding of separatrix even though this may be already available in the gEQDSK file

        :param levels: levels in normalized psi. Can be an array ranging from 0 to 1, or the number of flux surfaces

        :param map: array ranging from 0 to 1 which will be used to set the levels, or 'rho' if flux surfaces are generated based on gEQDSK

        :param maxPSI: (default 0.9999)

        :param calculateAvgGeo: Boolean which sets whether flux-surface averaged and geometric quantities are automatically calculated

        :param quiet: Verbosity level

        :param \**kw: overwrite key entries

        >> OMFIT['test']=OMFITgeqdsk(OMFITsrc+'/../samples/g133221.01000')
        >> # copy the original flux surfaces
        >> flx=copy.deepcopy(OMFIT['test']['fluxSurfaces'])
        >> # to use PSI
        >> mapping=None
        >> # to use RHO instead of PSI
        >> mapping=OMFIT['test']['RHOVN']
        >> # trace flux surfaces
        >> flx.findSurfaces(np.linspace(0,1,100),mapping=map)
        >> # to increase the accuracy of the flux surface tracing (higher numbers --> smoother surfaces, more time, more memory)
        >> flx.changeResolution(2)
        >> # plot
        >> flx.plot()
        """

        # initialization by ODS
        if isinstance(gEQDSK, ODS):
            return self.from_omas(gEQDSK)

        from omas import define_cocos

        self.quiet = quiet
        self.forceFindSeparatrix = forceFindSeparatrix
        self.calculateAvgGeo = calculateAvgGeo
        self.levels = levels
        self.resolution = resolution

        if gEQDSK is None:
            if not self.quiet:
                printi('Flux surfaces from tables')
            if cocosin is None:
                # Eventually this should raise an error but just warn now
                #                raise ValueError("Define cocosin for input equilibrium values")
                printw("cocosin not defined. Should be specified, but assuming COCOS 1 for now")
                self.cocosin = 1
            else:
                self.cocosin = cocosin
            self.Rin = Rin
            self.Zin = Zin
            self.PSIin = PSIin
            self.Btin = Btin
            self['RCENTR'] = Rcenter
            if F is not None:
                self.F = F
            if P is not None:
                self.P = P
            self.sep = None
            self.open_sep = None
            self.flx = None
            self.PSIaxis = None
            self.forceFindSeparatrix = True
            self.rlim = rlim
            self.zlim = zlim

        else:
            self.cocosin = gEQDSK.cocos
            if not self.quiet:
                printi('Flux surfaces from %dx%d gEQDSK' % (gEQDSK['NW'], gEQDSK['NH']))
            self.Rin = np.linspace(0, gEQDSK['RDIM'], gEQDSK['NW']) + gEQDSK['RLEFT']
            self.Zin = np.linspace(0, gEQDSK['ZDIM'], gEQDSK['NH']) - gEQDSK['ZDIM'] / 2.0 + gEQDSK['ZMID']
            self.PSIin = gEQDSK['PSIRZ']
            self.F = gEQDSK['FPOL']
            self.P = gEQDSK['PRES']
            self.FFPRIM = gEQDSK['FFPRIM']
            self.PPRIME = gEQDSK['PPRIME']
            self.Btin = gEQDSK['AuxQuantities']['Bt']
            self['R0'] = gEQDSK['RMAXIS']
            self['Z0'] = gEQDSK['ZMAXIS']
            self['RCENTR'] = gEQDSK['RCENTR']
            self.sep = np.vstack((gEQDSK['RBBBS'], gEQDSK['ZBBBS'])).T
            self.open_sep = None
            self.PSIaxis = gEQDSK['SIMAG']
            self.flx = gEQDSK['SIBRY']
            self.rlim = gEQDSK['RLIM']
            self.zlim = gEQDSK['ZLIM']

            # At the end of the following section:
            # forceFindSeparatrix = None means that existing separatrix is likely to be ok
            # forceFindSeparatrix = True means that new separtrix should be found
            # forceFindSeparatrix = False means that old separtrix should be left alone
            if forceFindSeparatrix is None:
                psiSep = RectBivariateSplineNaN(self.Zin, self.Rin, self.PSIin).ev(gEQDSK['ZBBBS'], gEQDSK['RBBBS'])
                cost = np.sqrt(np.var(psiSep)) / abs(self.flx)
                if cost > 1.0e-2 and self.forceFindSeparatrix is None:
                    self.forceFindSeparatrix = 'check'
                if cost > 5.0e-2:
                    printi("Normalized variance of PSI at separatrix ['RBBBS'],['ZBBBS'] points is " + format(cost * 100, '6.3g') + '%')
                    self.forceFindSeparatrix = True
                cost = abs((self.flx - np.mean(psiSep)) / (self.flx - self.PSIaxis))
                if cost > 1.0e-2 and self.forceFindSeparatrix is None:
                    self.forceFindSeparatrix = 'check'
                if cost > 5.0e-2:
                    printi(
                        "Relative error between ['SIBRY'] and average value of PSI at separatrix ['RBBBS'],['ZBBBS'] points is "
                        + format(cost * 100, '6.3g')
                        + '%'
                    )
                    self.forceFindSeparatrix = True
                if self.PSIaxis < self.flx:
                    self.flx = np.min(psiSep)
                if self.PSIaxis > self.flx:
                    self.flx = np.max(psiSep)

        self._cocos = define_cocos(self.cocosin)

        # setup level mapping
        if isinstance(map, str):
            if map == 'rho' and gEQDSK is not None:
                self.map = gEQDSK['RHOVN']
                if not self.quiet:
                    printi('Levels based on rho ...')
            else:
                self.map = None
                if not self.quiet:
                    printi('Levels based on psi ...')
        elif map is not None:
            if not self.quiet:
                printi('Levels based on user provided map ...')
            self.map = map
        else:
            self.map = None
            if not self.quiet:
                printi('Levels based on psi ...')

        self.maxPSI = maxPSI

        self.update(kw)

        self.dynaLoad = True

    @dynaLoad
    def load(self):
        self._changeResolution(self.resolution)

        self._crop()

        self._findAxis()

        if self.forceFindSeparatrix is not False:
            if self.forceFindSeparatrix == 'check':
                # check if the flux surface at PSI=1 (using the original PSI definition) is actually valid
                if not len(self._findSurfaces(levels=[1.0])):
                    printi('Forcing find of new separatrix!')
                    self.quiet = False
                    self.forceFindSeparatrix = True
            if self.forceFindSeparatrix is True:
                self._findSeparatrix()

        if self.levels is not None:
            self.findSurfaces(self.levels)

    def _findAxis(self):
        if not self.quiet:
            printi('Find magnetic axis ...')
        if not hasattr(self, 'R'):
            self._changeResolution(0)

        # limit search of the axis to a circle centered in the center of the domain
        if 'R0' not in self or 'Z0' not in self:
            Raxis = (self.R[0] + self.R[-1]) / 2.0
            Zaxis = (self.Z[0] + self.Z[-1]) / 2.0
            dmax = (self.R[-1] - self.R[0]) / 2.0
        else:
            Raxis = self['R0']
            Zaxis = self['Z0']
            dmax = (self.R[-1] - self.R[0]) / 2.0 / 1.5
        RR, ZZ = np.meshgrid(self.R - Raxis, self.Z - Zaxis)
        DD = np.sqrt(RR**2 + ZZ**2) < dmax
        tmp = self.PSI.copy()
        tmp[np.where(DD == 0)] = np.nan

        # figure out sign (fitting paraboloid going inward)
        ri0 = np.argmin(abs(self.R - Raxis))
        zi0 = np.argmin(abs(self.Z - Zaxis))
        n = np.min([ri0, zi0, len(self.R) - ri0, len(self.Z) - zi0])
        ax = 0
        for k in range(1, n)[::-1]:
            ri = (ri0 + np.array([-k, 0, +k, 0, 0])).astype(int)
            zi = (zi0 + np.array([0, 0, 0, +k, -k])).astype(int)
            try:
                ax, bx, ay, by, c = paraboloid(self.R[ri], self.Z[zi], tmp[zi, ri])
                break
            except np.linalg.LinAlgError:
                pass
        if ax > 0:
            # look for the minimum
            m = np.nanargmin(tmp)
        else:
            # look for the maximum
            m = np.nanargmax(tmp)
        Zmi = int(m / self.PSI.shape[1])
        Rmi = int(m - Zmi * self.PSI.shape[1])

        # pick center points based on the grid
        self.PSIaxis = self.PSI[Zmi, Rmi]
        self['R0'] = self.R[Rmi]
        self['Z0'] = self.Z[Zmi]

        # fit paraboloid in the vicinity of the grid-based center
        ri = (Rmi + np.array([-1, 0, +1, 0, 0])).astype(int)
        zi = (Zmi + np.array([0, 0, 0, +1, -1])).astype(int)
        ax, bx, ay, by, c = paraboloid(self.R[ri], self.Z[zi], self.PSI[zi, ri])
        self['R0_interp'] = -bx / (2 * ax)
        self['Z0_interp'] = -by / (2 * ay)

        # forceFindSeparatrix also controls whether PSI on axis should be redefined
        if self.forceFindSeparatrix is not False:
            # set as the center value of PSI (based on R0 and Z0)
            PSIaxis_found = RectBivariateSplineNaN(self.Z, self.R, self.PSI).ev(self['Z0_interp'], self['R0_interp'])
            if self.PSIaxis is None:
                self.PSIaxis = self.PSI[Zmi, Rmi]
            elif self.flx is not None:
                self.PSI = (self.PSI - PSIaxis_found) / abs(self.flx - PSIaxis_found) * abs(self.flx - self.PSIaxis) + self.PSIaxis

        # understand sign of the current based on curvature of psi
        if not hasattr(self, 'sign_Ip'):
            self.sign_Ip = np.sign(ax) * self._cocos['sigma_Bp']
        # tmp[np.isnan(tmp)]=5
        # contour(self.Rin,self.Zin,tmp)
        # pyplot.plot(self['R0'],self['Z0'],'or')
        # pyplot.plot(self['R0_interp'],self['Z0_interp'],'ob')

    def _findSurfaces(self, levels, usePSI=False):
        signTheta = self._cocos['sigma_RpZ'] * self._cocos['sigma_rhotp']  # +! is clockwise
        if not hasattr(self, 'PSI'):
            self._changeResolution(self.resolution)
        if usePSI:
            levels_psi = np.array(levels)
            levels = (levels_psi - self.PSIaxis) / ((self.flx - self.PSIaxis) * self.maxPSI)
        else:
            levels_psi = np.array(levels) * (self.flx - self.PSIaxis) * self.maxPSI + self.PSIaxis
        CS = contourPaths(self.R, self.Z, self.PSI, levels_psi)
        lines = SortedDict()
        for k, item1 in enumerate(CS):
            line = []
            if levels_psi[k] == self.PSIaxis:
                # axis
                line = np.array([self['R0'], self['Z0']]).reshape((1, 2))
            elif len(item1):
                # all others
                index = np.argsort([len(k1.vertices) for k1 in item1])
                if item1[-1] in index:
                    index.insert(0, index.pop(item1[-1]))
                for k1 in index:
                    path = item1[k1]
                    r = path.vertices[:, 0]
                    z = path.vertices[:, 1]
                    if np.any(np.isnan(r * z)):
                        continue
                    r[0] = r[-1] = (r[0] + r[-1]) * 0.5
                    z[0] = z[-1] = (z[0] + z[-1]) * 0.5
                    t = np.unwrap(np.arctan2(z - self['Z0'], r - self['R0']))
                    t -= np.mean(t)
                    l = np.linspace(0, 1, len(t))
                    X = max(abs(integrate.cumtrapz(integrate.cumtrapz(t, l, initial=0), l)))
                    if not len(line) or X > Xmax:
                        orientation = int(np.sign((z[0] - self['Z0']) * (r[1] - r[0]) - (r[0] - self['R0']) * (z[1] - z[0])))
                        if orientation != 0:
                            line = path.vertices[:: signTheta * orientation, :]
                        else:
                            line = path.vertices
                        Xmax = X

            lines[levels[k]] = line
        del self.PSI
        del self.R
        del self.Z
        return lines

    @dynaLoad
    def findSurfaces(self, levels=None, map=None):
        '''
        Find flux surfaces at levels

        :param levels: defines at which levels the flux surfaces will be traced

        * None: use levels defined in gFile

        * Integer: number of levels

        * list: list of levels to find surfaces at

        :param map: psi mapping on which levels are defined (e.g. rho as function of psi)
        '''
        signTheta = self._cocos['sigma_RpZ'] * self._cocos['sigma_rhotp']  # +! is clockwise
        t0 = datetime.datetime.now()
        if not hasattr(self, 'sep'):
            self._findSeparatrix()
        if np.iterable(levels):
            levels = list(levels)
            levels.sort()
        elif is_int(levels) and not isinstance(levels, bool):
            levels = list(np.linspace(0, 1, int(levels)))
        else:
            if 'levels' not in self:
                levels = list(np.linspace(0, 1, len(self.Rin)))
            else:
                levels = self['levels']

        if map is not None:
            self.map = map
        if self.map is not None:
            levels = scipy.interpolate.PchipInterpolator(self.map, np.linspace(0, 1, self.map.size), extrapolate=True)(levels).tolist()
            levels[0] = 0.0
            levels[-1] = 1.0

        if not self.quiet:
            printi('Tracing flux surfaces ...')

        levels_psi = np.array(levels) * (self.flx - self.PSIaxis) * self.maxPSI + self.PSIaxis
        lines = self._findSurfaces(levels)

        self['levels'] = np.zeros((len(levels)))
        self['flux'] = fluxSurfaceTraces()
        self.nc = 0

        for k, item1 in reverse_enumerate(lines.keys()):
            self['flux'][k] = SortedDict()
            self['flux'][k]['psi'] = levels_psi[k]
            self['levels'][k] = levels[k]

            line = lines[item1]

            if k == 0:
                # axis
                imaxr = np.argmax(self['flux'][k + 1]['R'])
                z_at_max_r, max_r = parabolaMaxCycle(self['flux'][k + 1]['Z'], self['flux'][k + 1]['R'], imaxr)

                iminr = np.argmin(self['flux'][k + 1]['R'])
                z_at_min_r, min_r = parabolaMaxCycle(self['flux'][k + 1]['Z'], self['flux'][k + 1]['R'], iminr)

                imaxz = np.argmax(self['flux'][k + 1]['Z'])
                r_at_max_z, max_z = parabolaMaxCycle(self['flux'][k + 1]['R'], self['flux'][k + 1]['Z'], imaxz)

                iminz = np.argmin(self['flux'][k + 1]['Z'])
                r_at_min_z, min_z = parabolaMaxCycle(self['flux'][k + 1]['R'], self['flux'][k + 1]['Z'], iminz)

                a = 0.5 * (max_r - min_r)
                kap = 0.5 * ((max_z - min_z) / a)

                simplePath = Path(np.vstack((self['flux'][k + 1]['R'], self['flux'][k + 1]['Z'])).T)
                if simplePath.contains_point((self['R0_interp'], self['Z0_interp'])):
                    self['R0'] = self['R0_interp']
                    self['Z0'] = self['Z0_interp']

                r = a * 1e-3
                t = np.linspace(0, 2 * np.pi, len(self['flux'][k + 1]['R']))

                self['flux'][k]['R'] = r * np.cos(t) + self['R0']
                self['flux'][k]['Z'] = -signTheta * kap * r * np.sin(t) + self['Z0']

            else:
                # all others
                if len(line):
                    self['flux'][k]['R'] = line[:, 0]
                    self['flux'][k]['Z'] = line[:, 1]
                elif hasattr(self, 'sep') and k == len(levels) - 1:
                    # If you are here, it is because the EFIT separatrix turned out to be not so precise
                    printi('Forcing find of new separatrix!')
                    self._findSeparatrix()
                    self['flux'][k]['R'] = self.sep[:, 0]
                    self['flux'][k]['Z'] = self.sep[:, 1]
                else:
                    printw(
                        'Bad flux surface! #' + str(k) + " This is likely to be a bad equilibrium, which does not satisfy Grad-Shafranov..."
                    )
                    # pyplot.plot(path.vertices[:,0],path.vertices[:,1],'b')
                    # pyplot.plot(self.sep[:,0],self.sep[:,1],'r')
                    self['flux'][k]['R'] = self['flux'][k + 1]['R'].copy()
                    self['flux'][k]['Z'] = self['flux'][k + 1]['Z'].copy()
            self.nc += 1

        self['flux'].sort()

        # if lcfs does not close it's likely because we trusted external value of flux surface flux (e.g. from gEQDSK)
        # so we need to force re-finding of flux surfaces
        if (self['flux'][self.nc - 1]['R'][0] != self['flux'][self.nc - 1]['R'][-1]) | (
            self['flux'][self.nc - 1]['Z'][0] != self['flux'][self.nc - 1]['Z'][-1]
        ):
            self.changeResolution(self.resolution)

        if not self.quiet:
            printi('  > Took {:}'.format(datetime.datetime.now() - t0))
        # These quantities can take a lot of space, depending on the resolution
        # They are deleted, and if necessary re-generated on demand
        self.surfAvg()

    def _changeResolution(self, resolution):
        # PSI is the table on which flux surfaces are generated.
        # By increasing its resolution, the contour path becomes more and more smooth.
        # This is much better than doing a spline interpolation through a rough path.
        self.resolution = resolution
        if self.resolution == 0:
            self.R = self.Rin
            self.Z = self.Zin
            self.PSI = self.PSIin
        elif is_int(self.resolution):
            if self.resolution > 0:
                if not self.quiet:
                    printi(f'Increasing tables resolution by factor of {2 ** (self.resolution)} ...')
                nr = len(self.Rin)
                nz = len(self.Zin)
                for k in range(self.resolution):
                    nr = nr + nr - 1
                    nz = nz + nz - 1
                self.R = np.linspace(min(self.Rin), max(self.Rin), nr)
                self.Z = np.linspace(min(self.Zin), max(self.Zin), nz)
                self.PSI = RectBivariateSplineNaN(self.Zin, self.Rin, self.PSIin)(self.Z, self.R)
            elif self.resolution < 0:
                if not self.quiet:
                    printi(f'Decreasing tables resolution by factor of {2 ** abs(self.resolution)} ...')
                resolution = 2 ** abs(self.resolution)
                self.R = self.Rin[::resolution]
                self.Z = self.Zin[::resolution]
                self.PSI = self.PSIin[::resolution, ::resolution]
        elif is_float(self.resolution):
            if not self.quiet:
                printi(f'Interpolating tables to {self.resolution} [m] resolution ...')
            self.R = np.linspace(min(self.Rin), max(self.Rin), int(np.ceil((max(self.Rin) - min(self.Rin)) / self.resolution)))
            self.Z = np.linspace(min(self.Zin), max(self.Zin), int(np.ceil((max(self.Zin) - min(self.Zin)) / self.resolution)))
            self.PSI = RectBivariateSplineNaN(self.Zin, self.Rin, self.PSIin)(self.Z, self.R)

        self.dd = np.sqrt((self.R[1] - self.R[0]) ** 2 + (self.Z[1] - self.Z[0]) ** 2)
        if not self.quiet:
            printi(f'Grid diagonal resolution: {self.dd} [m]')

    @dynaLoad
    def changeResolution(self, resolution):
        '''
        :param resolution: resolution to use when tracing flux surfaces

        * integer: multiplier of the original table

        * float: grid resolution in meters
        '''
        self._changeResolution(resolution)
        self._crop()
        self._findAxis()
        self._findSeparatrix()
        self.findSurfaces()

    def _findSeparatrix(self, accuracy=3):
        # separatrix is found by looking for the largest closed path enclosing the magnetic axis
        signTheta = self._cocos['sigma_RpZ'] * self._cocos['sigma_rhotp']  # +! is clockwise
        if not hasattr(self, 'PSI'):
            self._changeResolution(self.resolution)
        if not self.quiet:
            printi('Find separatrix ...')

        # use of self.PSIaxis is more robust than relying on
        # min an max, which can fail when coils are working hard
        # and poloidal flux from plasma is weak
        if self.sign_Ip * self._cocos['sigma_Bp'] > 0:
            # PSI increases with minor radius
            flxm = self.PSIaxis  # np.nanmin(self.PSI)
            flxM = np.nanmax(self.PSI)
        else:
            flxm = self.PSIaxis  # np.nanmax(self.PSI)
            flxM = np.nanmin(self.PSI)

        # pyplot.plot(self['R0'],self['Z0'],'or') #<< DEBUG axis
        kdbgmax = 50
        flx_found = None
        forbidden = []
        sep = None
        open_sep = None
        for kdbg in range(kdbgmax):
            flx = (flxM + flxm) / 2.0
            line = []
            paths = contourPaths(self.R, self.Z, self.PSI, [flx])[0]
            for path in paths:
                # if there is not Nan and the path closes
                if (
                    not np.isnan(path.vertices[:]).any()
                    and np.allclose(path.vertices[0, 0], path.vertices[-1, 0])
                    and np.allclose(path.vertices[0, 1], path.vertices[-1, 1])
                ):
                    path.vertices[0, 0] = path.vertices[-1, 0] = (path.vertices[0, 0] + path.vertices[-1, 0]) * 0.5
                    path.vertices[0, 1] = path.vertices[-1, 1] = (path.vertices[0, 1] + path.vertices[-1, 1]) * 0.5
                    simplePath = Path(path.vertices)
                    if (
                        np.max(simplePath.vertices[:, 0]) > self['R0']
                        and np.min(simplePath.vertices[:, 0]) < self['R0']
                        and np.max(simplePath.vertices[:, 1]) > self['Z0']
                        and min(simplePath.vertices[:, 1]) < self['Z0']
                        and simplePath.contains_point((self['R0'], self['Z0']))
                        and not any([simplePath.contains_point((Rf, Zf)) for Rf, Zf in forbidden])
                    ):
                        dR = path.vertices[1, 0] - path.vertices[0, 0]
                        dZ = path.vertices[1, 1] - path.vertices[0, 1]
                        orientation = int(np.sign((path.vertices[0, 1] - self['Z0']) * dR - (path.vertices[0, 0] - self['R0']) * dZ))
                        line = path.vertices[:: signTheta * orientation, :]
                    else:
                        Rf = np.mean(path.vertices[:, 0])
                        Zf = np.mean(path.vertices[:, 1])
                        # do not add redundant forbidden points
                        if any([simplePath.contains_point((Rf, Zf)) for Rf, Zf in forbidden]):
                            pass
                        # do not add forbidden that encompass the current separatrix
                        elif sep is not None and Path(sep).contains_point((Rf, Zf)):
                            pass
                        else:
                            # pyplot.plot(Rf,Zf,'xm') #<< DEBUG plot, showing additional forbiddent points
                            forbidden.append([Rf, Zf])

            if len(line):
                # pyplot.plot(line[:,0],line[:,1],'r') #<< DEBUG plot, showing convergence
                try:
                    # stop condition
                    np.testing.assert_array_almost_equal(sep / self['R0'], line / self['R0'], accuracy)
                    break
                except Exception:
                    pass
                finally:
                    sep = line
                    flx_found = flx
                    flxm = flx

            else:
                open_sep = paths
                flxM = flx
                # for path in paths:
                #    pyplot.plot(path.vertices[:,0],path.vertices[:,1],'b') #<< DEBUG plot, showing convergence

        # do not store new separatrix if it hits edges of computation domain
        if (
            (np.abs(np.min(sep[:, 0]) - np.min(self.R)) < 1e-3)
            or (np.abs(np.max(sep[:, 0]) - np.max(self.R)) < 1e-3)
            or (np.abs(np.min(sep[:, 1]) - np.min(self.Z)) < 1e-3)
            or (np.abs(np.max(sep[:, 1]) - np.max(self.Z)) < 1e-3)
        ):
            self.forceFindSeparatrix = False

        else:
            self.sep = sep
            self.open_sep = open_sep

            # pyplot.plot(sep[:,0],sep[:,1],'k',lw=2) #<< DEBUG plot, showing separatrix
            # printd(flxm,flxM,flx,flxm-flxM,kdbg) #<< DEBUG print, showing convergence

            if self.flx is None:
                self.flx = flx_found
            elif self.PSIaxis is not None:
                self.PSI = (self.PSI - self.PSIaxis) / abs(flx_found - self.PSIaxis) * abs(self.flx - self.PSIaxis) + self.PSIaxis

            if self.open_sep:
                outer_midplane_distance = []
                for k, path in enumerate(self.open_sep):
                    ix = np.where(path.vertices[:, 0] > self['R0'])[0]
                    if not len(ix):
                        outer_midplane_distance.append(np.inf)
                        continue
                    outer_midplane_distance.append(min(abs(path.vertices[ix, 1] - self['Z0'])))
                ix = np.argmin(outer_midplane_distance)
                # pyplot.plot(self.open_sep[ix].vertices[:,0],self.open_sep[ix].vertices[:,1],'m') #<< DEBUG plot, showing open field separatrix
                self.open_sep = self.open_sep[ix].vertices

        if kdbg == kdbgmax - 1:
            printw('Finding of last closed flux surface aborted after %d iterations!!!' % kdbgmax)

    def _calcBrBz(self):
        RR, ZZ = np.meshgrid(self.Rin, self.Zin)
        [dPSIdZ, dPSIdR] = np.gradient(self.PSIin, self.Zin[1] - self.Zin[0], self.Rin[1] - self.Rin[0])
        Br = self._cocos['sigma_RpZ'] * self._cocos['sigma_Bp'] * dPSIdZ / (RR * ((2 * np.pi) ** self._cocos['exp_Bp']))
        Bz = -self._cocos['sigma_RpZ'] * self._cocos['sigma_Bp'] * dPSIdR / (RR * ((2 * np.pi) ** self._cocos['exp_Bp']))
        return Br, Bz

    def _BrBzAndF(self):
        t0 = datetime.datetime.now()
        if not self.quiet:
            printi('Find Br, Bz, F on flux surfaces ...')

        RR, ZZ = np.meshgrid(self.Rin, self.Zin)

        # F=Bt*R is a flux surface quantity
        if not hasattr(self, 'F'):
            # if F is not present, find it through the Btin table
            F = self._surfMean(self.Btin * RR)
        else:
            # if F is present
            F = interpolate.InterpolatedUnivariateSpline(np.linspace(0, 1, self.F.size), self.F)(self['levels'])
        for k in range(self.nc):
            self['flux'][k]['F'] = F[k]

        if self['RCENTR'] is None:
            printw('Using magnetic axis as RCENTR of vacuum field ( BCENTR = Fpol[-1] / RCENTR)')
            self['RCENTR'] = self['R0']
        self['BCENTR'] = self['flux'][self.nc - 1]['F'] / self['RCENTR']

        # if P is present
        if hasattr(self, 'P'):
            P = interpolate.InterpolatedUnivariateSpline(np.linspace(0, 1, self.P.size), self.P)(self['levels'])
            for k in range(self.nc):
                self['flux'][k]['P'] = P[k]

        # if PPRIME is present
        if hasattr(self, 'PPRIME'):
            PPRIME = interpolate.InterpolatedUnivariateSpline(np.linspace(0, 1, self.PPRIME.size), self.PPRIME)(self['levels'])
            for k in range(self.nc):
                self['flux'][k]['PPRIME'] = PPRIME[k]

        # if FFPRIM is present
        if hasattr(self, 'FFPRIM'):
            FFPRIM = interpolate.InterpolatedUnivariateSpline(np.linspace(0, 1, self.FFPRIM.size), self.FFPRIM)(self['levels'])
            for k in range(self.nc):
                self['flux'][k]['FFPRIM'] = FFPRIM[k]

        # calculate Br and Bz magnetic fields
        Br, Bz = self._calcBrBz()

        [dBrdZ, dBrdR] = np.gradient(Br, self.Zin[2] - self.Zin[1], self.Rin[2] - self.Rin[1])
        [dBzdZ, dBzdR] = np.gradient(Bz, self.Zin[2] - self.Zin[1], self.Rin[2] - self.Rin[1])
        [dBtdZ, dBtdR] = np.gradient(self.Btin, self.Zin[2] - self.Zin[1], self.Rin[2] - self.Rin[1])
        [dRBtdZ, dRBtdR] = np.gradient(RR * self.Btin, self.Zin[2] - self.Zin[1], self.Rin[2] - self.Rin[1])
        Jt = self._cocos['sigma_RpZ'] * (dBrdZ - dBzdR) / (4 * np.pi * 1e-7)

        # calculate flux expansion
        Brfun = RectBivariateSplineNaN(self.Zin, self.Rin, Br)
        Bzfun = RectBivariateSplineNaN(self.Zin, self.Rin, Bz)
        Jtfun = RectBivariateSplineNaN(self.Zin, self.Rin, Jt)
        self.fluxexpansion = []
        self.dl = []
        self.fluxexpansion_dl = []
        self.int_fluxexpansion_dl = []
        for k in range(self.nc):
            self.dl.append(
                np.sqrt(np.ediff1d(self['flux'][k]['R'], to_begin=0.0) ** 2 + np.ediff1d(self['flux'][k]['Z'], to_begin=0.0) ** 2)
            )
            self['flux'][k]['Br'] = Brfun.ev(self['flux'][k]['Z'], self['flux'][k]['R'])
            self['flux'][k]['Bz'] = Bzfun.ev(self['flux'][k]['Z'], self['flux'][k]['R'])
            modBp = np.sqrt(self['flux'][k]['Br'] ** 2 + self['flux'][k]['Bz'] ** 2)
            self.fluxexpansion.append(1 / modBp)
            self.fluxexpansion_dl.append(self.fluxexpansion[k] * self.dl[k])
            self.int_fluxexpansion_dl.append(np.sum(self.fluxexpansion_dl[k]))
            self['flux'][k]['Jt'] = Jtfun.ev(self['flux'][k]['Z'], self['flux'][k]['R'])
        if not self.quiet:
            printi('  > Took {:}'.format(datetime.datetime.now() - t0))

    def _crop(self):
        '''
        Eliminate points on the PSI map that are outside of the limiter surface.
        '''
        if self.rlim is not None and self.zlim is not None and len(self.rlim) > 3 and len(self.zlim) > 3:
            if not self.quiet:
                printi('Cropping tables ...')
            if self.rlim is not None and self.zlim is not None:
                if np.any(np.isnan(self.rlim)) or np.any(np.isnan(self.zlim)):
                    printw('fluxsurfaces: rlim/zlim arrays contain NaNs')
                    return
            bbox = [min(self.rlim), max(self.rlim), min(self.zlim), max(self.zlim)]
            limits = [
                max([np.argmin(abs(self.R - bbox[0])) - 1, 0]),
                min([np.argmin(abs(self.R - bbox[1])) + 1, len(self.R) - 1]),
                max([np.argmin(abs(self.Z - bbox[2])) - 1, 0]),
                min([np.argmin(abs(self.Z - bbox[3])) + 1, len(self.Z) - 1]),
            ]
            self.PSI = self.PSI[limits[2] : limits[3], limits[0] : limits[1]]
            self.R = self.R[limits[0] : limits[1]]
            self.Z = self.Z[limits[2] : limits[3]]

    @dynaLoad
    def resample(self, npts=None, technique='uniform', phase='Xpoint'):
        '''
        resample number of points on flux surfaces

        :param npts: number of points

        :param technique: 'uniform','separatrix','pest'

        :param phase: float for poloidal angle or 'Xpoint'
        '''
        if npts is None or np.iterable(npts):
            npts = self.sep.size
        if technique == 'separatrix':
            self._resample()
        elif technique == 'uniform':
            self._resample(npts, phase)
        elif technique == 'pest':
            th = self._straightLineTheta(npts)
            self._resample(th, phase)
        self._BrBzAndF()
        self.surfAvg()

    def _resample(self, npts=None, phase='Xpoint'):
        """
        Resampling will lead inaccuracies of the order of the resolution
        on which the flux surface tracing was originally done
        """

        signTheta = self._cocos['sigma_RpZ'] * self._cocos['sigma_rhotp']  # +! is clockwise

        # pyplot.subplot(1, 1, 1, aspect='equal')
        if not self.quiet:
            printi('Resampling flux surfaces ...')
        if npts is None:
            # maintain angles, direction and angle of separatrix
            theta0 = -signTheta * np.arctan2(self.sep[:, 1] - self['Z0'], self.sep[:, 0] - self['R0'])
            thetaXpoint = theta0[0]
            per_ = 1
        else:
            if phase == 'Xpoint':
                # X-point as the location of minimum Bp along the separatrix
                # indexXpoint = np.argmin(
                #     RectBivariateSplineNaN(self.Zin, self.Rin, self.Bpin).ev(self.sep[:, 1], self.sep[:, 0])
                # )
                # thetaXpoint = np.arctan2(self.sep[indexXpoint, 1] - self['Z0'], self.sep[indexXpoint, 0] - self['R0'])
                # X-point as the location of minimum angle between two adjacent segments of the separatrix
                a = np.vstack((np.gradient(self.sep[:-1, 0]), np.gradient(self.sep[:-1, 1]))).T
                b = np.vstack(
                    (
                        np.gradient(np.hstack((self.sep[1:-1, 0], self.sep[0, 0]))),
                        np.gradient(np.hstack((self.sep[1:-1, 1], self.sep[0, 1]))),
                    )
                ).T
                gr = (a[:, 0] * b[:, 0] + a[:, 1] * b[:, 1]) / np.sqrt(np.sum(a**2, 1)) / np.sqrt(np.sum(b**2, 1))
                t = -signTheta * np.arctan2(self.sep[:, 1] - self['Z0'], self.sep[:, 0] - self['R0'])
                thetaXpoint = t[np.argmin(gr)]
                per_ = 0
            else:
                thetaXpoint = phase
                per_ = 1
            if not np.iterable(npts):
                theta0 = np.linspace(0, 2 * np.pi, npts) + thetaXpoint

        for k in range(self.nc):

            if self['levels'][k] == 1:
                per = per_
            else:
                per = 1

            t = np.unwrap(-signTheta * np.arctan2(self['flux'][k]['Z'] - self['Z0'], self['flux'][k]['R'] - self['R0']))

            if np.iterable(npts):
                if np.iterable(npts[0]):
                    theta0 = npts[k]
                else:
                    theta0 = npts

            # force 'X axis' to be monotonically increasing
            if t[0] > t[1]:
                t = -t
                theta = -theta0
            else:
                theta = theta0

            index = np.argsort((theta[:-1] + thetaXpoint) % (2 * np.pi) - thetaXpoint)
            index = np.hstack((index, index[0]))
            theta = np.unwrap(theta[index])

            index = np.argsort((t[:-1] + thetaXpoint) % (2 * np.pi) - thetaXpoint)
            index = np.hstack((index, index[0]))

            t = np.unwrap(t[index])
            for item in ['R', 'Z', 'Br', 'Bz']:
                if item in self['flux'][k]:
                    self['flux'][k][item] = self['flux'][k][item][index]
                    # These quantities are periodic (per=1);
                    # however per=0 can be used to allow discontinuity of derivatives at X-point
                    tckp = interpolate.splrep(t, self['flux'][k][item], k=3, per=1)  # per=per)
                    self['flux'][k][item] = interpolate.splev((theta - t[0]) % max(t - t[0]) + t[0], tckp, ext=0)
                    self['flux'][k][item][-1] = self['flux'][k][item][0]

    def _straightLineTheta(self, npts=None):
        signTheta = self._cocos['sigma_RpZ'] * self._cocos['sigma_rhotp']
        if not self.quiet:
            printi('Evaluating straight line theta ...')
        if 'avg' not in self:
            self.surfAvg()
        theta = []
        for k in range(self.nc):
            if npts is None:
                npts_ = self['flux'][k]['R'].size
            else:
                npts_ = npts

            # t_ is the uniformly sampled straight-line theta
            t_ = np.linspace(0, 2 * np.pi, npts_)

            if self['levels'][0] == 0 and k == 0:
                theta.append(t_)
            else:
                # thetaStraight is the calculated straigthline theta
                signBp = signTheta * np.sign(
                    (self['flux'][k]['Z'] - self['Z0']) * self['flux'][k]['Br']
                    - (self['flux'][k]['R'] - self['R0']) * self['flux'][k]['Bz']
                )
                Bp = signBp * np.sqrt(self['flux'][k]['Br'] ** 2 + self['flux'][k]['Bz'] ** 2)
                Bt = self['flux'][k]['F'] / self['flux'][k]['R']
                thetaStraight = np.unwrap(np.cumsum(self.dl[k] * Bt / (abs(Bp) * self['avg']['q'][k] * self['flux'][k]['R'])))
                # t is the real theta angle
                t = np.unwrap(-signTheta * np.arctan2(self['flux'][k]['Z'] - self['Z0'], self['flux'][k]['R'] - self['R0']))

                # enforce 'X axis' (thetaStraight) to be monotonically increasing
                if thetaStraight[0] > thetaStraight[1]:
                    thetaStraight = -thetaStraight
                if t[0] > t[1]:
                    t = -t

                # to use periodic spline I need to make my 'y' periodic, this is done by defining tDelta, that is the difference
                # between the straigthline and the real theta
                tDelta = (t - t[0]) - (thetaStraight - thetaStraight[0])

                # the interpolation should be strictly monotonic. Enforcing continuity to the fifth derivative (k=5) is very likely to ensure that.
                tckp = interpolate.splrep(thetaStraight, tDelta, k=5, per=True)
                tDeltaInt = interpolate.splev(
                    (t_ - thetaStraight[0]) % max(thetaStraight - thetaStraight[0]) + thetaStraight[0], tckp, ext=2
                )

                # now t0 represents the real thetas at which I get uniform straightline theta sampling
                # t0 is equal to the uniformly sampled straigthline theta plus the interpolated tDelta angle and some phase constants
                t0 = tDeltaInt + t_ + t[0] - thetaStraight[0]

                if t[0] < t[1]:
                    t0 = -t0
                    t = -t

                theta.append(t0)

        #                if k==self['levels'].size-1:
        #                    pyplot.figure()
        #                    pyplot.plot(
        #                        thetaStraight,tDelta,'o',
        #                        (t_-thetaStraight[0])%max(thetaStraight-thetaStraight[0])+thetaStraight[0],tDeltaInt,'.-r',
        #                    )
        #                    pyplot.figure()
        #                    pyplot.plot(
        #                        thetaStraight,t,'o',
        #                        t_,t0,'.-r',
        #                    )

        if self['levels'][0] == 0:
            if theta[1][0] < theta[1][-1]:
                theta[0] = np.linspace(min(theta[1]), min(theta[1]) + 2 * np.pi, np_)
            else:
                theta[0] = np.linspace(min(theta[1]) + 2 * np.pi, min(theta[1]), np_)

        return theta

    def surfAvg(self, function=None):
        """
        Flux surface averaged quantity for each flux surface

        :param function: function which returns the value of the quantity to be flux surface averaged at coordinates r,z

        :return: array of the quantity fluxs surface averaged for each flux surface

        :Example:

        >> def test_avg_function(r, z):
        >>     return RectBivariateSplineNaN(Z, R, PSI, k=1).ev(z,r)

        """
        t0 = datetime.datetime.now()
        if not self.calculateAvgGeo:
            return

        self._BrBzAndF()

        # define handy function for flux-surface averaging
        def flxAvg(k, input):
            return np.sum(self.fluxexpansion_dl[k] * input) / self.int_fluxexpansion_dl[k]

        # if user wants flux-surface averaging of a specific function, then calculate and return it
        if function is not None:
            if 'avg' not in self:
                self.surfAvg()
            if not self.quiet:
                printi('Flux surface averaging of user defined quantity ...')
            avg = np.zeros((self.nc))
            for k in range(self.nc):
                avg[k] = flxAvg(k, function(self['flux'][k]['R'], self['flux'][k]['Z']))
            return avg

        self['avg'] = SortedDict()
        self['geo'] = SortedDict()
        self['midplane'] = SortedDict()
        self['geo']['psi'] = self['levels'] * (self.flx - self.PSIaxis) + self.PSIaxis
        self['geo']['psin'] = self['levels']

        # calculate flux surface average of typical quantities
        if not self.quiet:
            printi('Flux surface averaging ...')
        for item in [
            'R',
            'a',
            'R**2',
            '1/R',
            '1/R**2',
            'Bp',
            'Bp**2',
            'Bp*R',
            'Bp**2*R**2',
            'Btot',
            'Btot**2',
            'Bt',
            'Bt**2',
            'ip',
            'vp',
            'q',
            'hf',
            'Jt',
            'Jt/R',
            'fc',
            'grad_term',
            'P',
            'F',
            'PPRIME',
            'FFPRIM',
        ]:
            self['avg'][item] = np.zeros((self.nc))
        for k in range(self.nc):
            Bp2 = self['flux'][k]['Br'] ** 2 + self['flux'][k]['Bz'] ** 2
            signBp = (
                self._cocos['sigma_rhotp']
                * self._cocos['sigma_RpZ']
                * np.sign(
                    (self['flux'][k]['Z'] - self['Z0']) * self['flux'][k]['Br']
                    - (self['flux'][k]['R'] - self['R0']) * self['flux'][k]['Bz']
                )
            )
            Bp = signBp * np.sqrt(Bp2)
            Bt = self['flux'][k]['F'] / self['flux'][k]['R']
            B2 = Bp2 + Bt**2
            B = np.sqrt(B2)
            bratio = B / np.max(B)
            self['flux'][k]['Bmax'] = np.max(B)

            # self['avg']['psi'][k]       = flxAvg(k, function(self['flux'][k]['R'],self['flux'][k]['Z']) )
            self['avg']['R'][k] = flxAvg(k, self['flux'][k]['R'])
            self['avg']['a'][k] = flxAvg(k, np.sqrt((self['flux'][k]['R'] - self['R0']) ** 2 + (self['flux'][k]['Z'] - self['Z0']) ** 2))
            self['avg']['R**2'][k] = flxAvg(k, self['flux'][k]['R'] ** 2)
            self['avg']['1/R'][k] = flxAvg(k, 1.0 / self['flux'][k]['R'])
            self['avg']['1/R**2'][k] = flxAvg(k, 1.0 / self['flux'][k]['R'] ** 2)
            self['avg']['Bp'][k] = flxAvg(k, Bp)
            self['avg']['Bp**2'][k] = flxAvg(k, Bp2)
            self['avg']['Bp*R'][k] = flxAvg(k, Bp * self['flux'][k]['R'])
            self['avg']['Bp**2*R**2'][k] = flxAvg(k, Bp2 * self['flux'][k]['R'] ** 2)
            self['avg']['Btot'][k] = flxAvg(k, B)
            self['avg']['Btot**2'][k] = flxAvg(k, B2)
            self['avg']['Bt'][k] = flxAvg(k, Bt)
            self['avg']['Bt**2'][k] = flxAvg(k, Bt**2)
            self['avg']['vp'][k] = (
                self._cocos['sigma_rhotp']
                * self._cocos['sigma_Bp']
                * np.sign(self['avg']['Bp'][k])
                * self.int_fluxexpansion_dl[k]
                * (2.0 * np.pi) ** (1.0 - self._cocos['exp_Bp'])
            )
            self['avg']['q'][k] = (
                self._cocos['sigma_rhotp']
                * self._cocos['sigma_Bp']
                * self['avg']['vp'][k]
                * self['flux'][k]['F']
                * self['avg']['1/R**2'][k]
                / ((2 * np.pi) ** (2.0 - self._cocos['exp_Bp']))
            )
            self['avg']['hf'][k] = flxAvg(k, (1.0 - np.sqrt(1.0 - bratio) * (1.0 + bratio / 2.0)) / bratio**2)

            # these quantites are calculated from Bx,By and hence are not that precise
            # if information is available about P, PPRIME, and F, FFPRIM then they will be substittuted
            self['avg']['ip'][k] = self._cocos['sigma_rhotp'] * np.sum(self.dl[k] * Bp) / (4e-7 * np.pi)
            self['avg']['Jt'][k] = flxAvg(k, self['flux'][k]['Jt'])
            self['avg']['Jt/R'][k] = flxAvg(k, self['flux'][k]['Jt'] / self['flux'][k]['R'])
            self['avg']['F'][k] = self['flux'][k]['F']
            if hasattr(self, 'P'):
                self['avg']['P'][k] = self['flux'][k]['P']
            elif 'P' in self['avg']:
                del self['avg']['P']
            if hasattr(self, 'PPRIME'):
                self['avg']['PPRIME'][k] = self['flux'][k]['PPRIME']
            elif 'PPRIME' in self['avg']:
                del self['avg']['PPRIME']
            if hasattr(self, 'FFPRIM'):
                self['avg']['FFPRIM'][k] = self['flux'][k]['FFPRIM']
            elif 'FFPRIM' in self['avg']:
                del self['avg']['FFPRIM']

            ## The circulating particle fraction calculation has been converted from IDL to python
            ## following the fraction_circ.pro which is widely used at DIII-D
            ## Formula 4.54 of S.P. Hirshman and D.J. Sigmar 1981 Nucl. Fusion 21 1079
            # x=np.array([0.0387724175, 0.1160840706, 0.1926975807, 0.268152185, 0.3419940908, 0.4137792043, 0.4830758016, 0.549467125, 0.6125538896, 0.6719566846, 0.7273182551, 0.7783056514, 0.8246122308, 0.8659595032, 0.9020988069, 0.9328128082, 0.9579168192, 0.9772599499, 0.9907262386, 0.9982377097])
            # w=np.array([0.0775059479, 0.0770398181, 0.0761103619, 0.074723169, 0.0728865823, 0.0706116473, 0.0679120458, 0.0648040134, 0.0613062424, 0.057439769, 0.0532278469, 0.0486958076, 0.0438709081, 0.0387821679, 0.0334601952, 0.0279370069, 0.0222458491, 0.0164210583, 0.0104982845, 0.004521277])
            # lmbd   = 1-x**2
            # weight = 2.*w*np.sqrt(1. - lmbd)
            # denom  = np.zeros((len(lmbd)))
            # for n in range(len(lmbd)):
            #    denom[n] = flxAvg(k, np.sqrt(1.-lmbd[n]*bratio) )
            # integral=np.sum(weight*lmbd/denom)
            # self['avg']['fc'][k]        =0.75*self['avg']['Btot**2'][k]/np.max(B)**2*integral
            #
            # The above calculation is exactly equivalent to the Lin-Lu form of trapped particle fraction
            # article: Y.R. Lin-Liu and R.L. Miller, Phys. of Plamsas 2 (1995) 1666
            h = self['avg']['Btot'][k] / self['flux'][k]['Bmax']
            h2 = self['avg']['Btot**2'][k] / self['flux'][k]['Bmax'] ** 2
            # Equation 4
            ftu = 1.0 - h2 / (h**2) * (1.0 - np.sqrt(1.0 - h) * (1.0 + 0.5 * h))
            # Equation 7
            ftl = 1.0 - h2 * self['avg']['hf'][k]
            # Equation 18,19
            self['avg']['fc'][k] = 1 - (0.75 * ftu + 0.25 * ftl)

            grad_parallel = np.diff(B) / self.fluxexpansion_dl[k][1:] / B[1:]
            self['avg']['grad_term'][k] = np.sum(self.fluxexpansion_dl[k][1:] * grad_parallel**2) / self.int_fluxexpansion_dl[k]

        # q on axis by extrapolation
        if self['levels'][0] == 0:
            x = self['levels'][1:]
            y = self['avg']['q'][1:]
            self['avg']['q'][0] = y[1] - ((y[1] - y[0]) / (x[1] - x[0])) * x[1]

        for k in range(self.nc):
            self['flux'][k]['q'] = self['avg']['q'][k]

        if 'P' in self['avg'] and 'PPRIME' not in self['avg']:
            self['avg']['PPRIME'] = deriv(self['geo']['psi'], self['avg']['P'])

        if 'F' in self['avg'] and 'FFPRIM' not in self['avg']:
            self['avg']['FFPRIM'] = self['avg']['F'] * deriv(self['geo']['psi'], self['avg']['F'])

        if 'PPRIME' in self['avg'] and 'FFPRIM' in self['avg']:
            self['avg']['Jt/R'] = (
                -self._cocos['sigma_Bp']
                * (self['avg']['PPRIME'] + self['avg']['FFPRIM'] * self['avg']['1/R**2'] / (4 * np.pi * 1e-7))
                * (2.0 * np.pi) ** self._cocos['exp_Bp']
            )

        # calculate currents based on Grad-Shafranov if pressure information is available
        # TEMPORARILY DISABLED: issue at first knot when looking at Jeff which is near zero on axis
        if False and 'PPRIME' in self['avg'] and 'F' in self['avg'] and 'FFPRIM' in self['avg']:
            self['avg']['dip/dpsi'] = (
                -self._cocos['sigma_Bp']
                * self['avg']['vp']
                * (self['avg']['PPRIME'] + self['avg']['FFPRIM'] * self['avg']['1/R**2'] / (4e-7 * np.pi))
                / ((2 * np.pi) ** (1.0 - self._cocos['exp_Bp']))
            )
            self['avg']['ip'] = integrate.cumtrapz(self['avg']['dip/dpsi'], self['geo']['psi'], initial=0)
        else:
            self['avg']['dip/dpsi'] = deriv(self['geo']['psi'], self['avg']['ip'])
        self['avg']['Jeff'] = (
            self._cocos['sigma_Bp']
            * self._cocos['sigma_rhotp']
            * self['avg']['dip/dpsi']
            * self['BCENTR']
            / (self['avg']['q'] * (2 * np.pi) ** (1.0 - self._cocos['exp_Bp']))
        )
        self['CURRENT'] = self['avg']['ip'][-1]

        # calculate geometric quantities
        if not self.quiet:
            printi('  > Took {:}'.format(datetime.datetime.now() - t0))
        if not self.quiet:
            printi('Geometric quantities ...')
        t0 = datetime.datetime.now()
        for k in range(self.nc):
            geo = fluxGeo(self['flux'][k]['R'], self['flux'][k]['Z'], lcfs=(k == (self.nc - 1)))
            for item in sorted(geo):
                if item not in self['geo']:
                    self['geo'][item] = np.zeros((self.nc))
                self['geo'][item][k] = geo[item]
        self['geo']['vol'] = np.abs(self.volume_integral(1))
        self['geo']['cxArea'] = np.abs(self.surface_integral(1))
        self['geo']['phi'] = (
            self._cocos['sigma_Bp']
            * self._cocos['sigma_rhotp']
            * integrate.cumtrapz(self['avg']['q'], self['geo']['psi'], initial=0)
            * (2.0 * np.pi) ** (1.0 - self._cocos['exp_Bp'])
        )
        # self['geo']['bunit']=(abs(self['avg']['q'])/self['geo']['a'])*( deriv(self['geo']['a'],self['geo']['psi']) )
        self['geo']['bunit'] = deriv(self['geo']['a'], self['geo']['phi']) / (2.0 * np.pi * self['geo']['a'])

        # fix geometric quantities on axis
        if self['levels'][0] == 0:
            self['geo']['delu'][0] = 0.0
            self['geo']['dell'][0] = 0.0
            self['geo']['delta'][0] = 0.0
            self['geo']['zeta'][0] = 0.0
            self['geo']['zetaou'][0] = 0.0
            self['geo']['zetaiu'][0] = 0.0
            self['geo']['zetail'][0] = 0.0
            self['geo']['zetaol'][0] = 0.0
            # linear extrapolation
            x = self['levels'][1:]
            for item in ['kapu', 'kapl', 'bunit']:
                y = self['geo'][item][1:]
                self['geo'][item][0] = y[1] - ((y[1] - y[0]) / (x[1] - x[0])) * x[1]
            self['geo']['kap'][0] = 0.5 * self['geo']['kapu'][0] + 0.5 * self['geo']['kapl'][0]
            #  calculate rho only if levels start from 0
            self['geo']['rho'] = np.sqrt(np.abs(self['geo']['phi'] / (np.pi * self['BCENTR'])))
        else:
            # the values of phi, rho have meaning only if I can integrate from the first flux surface on...
            if 'phi' in self['geo']:
                del self['geo']['phi']
            if 'rho' in self['geo']:
                del self['geo']['rho']

        # calculate betas
        if 'P' in self['avg']:
            Btvac = self['BCENTR'] * self['RCENTR'] / self['geo']['R'][-1]
            self['avg']['beta_t'] = abs(
                self.volume_integral(self['avg']['P']) / (Btvac**2 / 2.0 / 4.0 / np.pi / 1e-7) / self['geo']['vol'][-1]
            )
            i = self['CURRENT'] / 1e6
            a = self['geo']['a'][-1]
            self['avg']['beta_n'] = self['avg']['beta_t'] / abs(i / a / Btvac) * 100
            Bpave = self['CURRENT'] * (4 * np.pi * 1e-7) / self['geo']['per'][-1]
            self['avg']['beta_p'] = abs(
                self.volume_integral(self['avg']['P']) / (Bpave**2 / 2.0 / 4.0 / np.pi / 1e-7) / self['geo']['vol'][-1]
            )

        # the values of rhon has a meaning only if I have the value at the lcfs
        if 'rho' in self['geo'] and self['levels'][self.nc - 1] == 1.0:
            self['geo']['rhon'] = self['geo']['rho'] / max(self['geo']['rho'])

            # fcap, f(psilim)/f(psi)
            self['avg']['fcap'] = np.zeros((self.nc))
            for k in range(self.nc):
                self['avg']['fcap'][k] = self['flux'][self.nc - 1]['F'] / self['flux'][k]['F']

            # hcap, fcap / <R0**2/R**2>
            self['avg']['hcap'] = self['avg']['fcap'] / (self['RCENTR'] ** 2 * self['avg']['1/R**2'])

            # RHORZ (linear extrapolation for rho>1)
            def ext_arr_linear(x, y):
                dydx = (y[-1] - y[-2]) / (x[-1] - x[-2])
                extra_x = (x[-1] - x[-2]) * np.r_[1:1000] + x[-1]
                extra_y = (x[-1] - x[-2]) * np.r_[1:1000] * dydx + y[-1]
                x = np.hstack((x, extra_x))
                y = np.hstack((y, extra_y))
                return [x, y]

            [new_psi_mesh0, new_PHI] = ext_arr_linear(self['geo']['psi'], self['geo']['phi'])
            PHIRZ = interpolate.interp1d(new_psi_mesh0, new_PHI, kind='linear', bounds_error=False)(self.PSIin)
            RHORZ = np.sqrt(abs(PHIRZ / np.pi / self['BCENTR']))

            # gcap <(grad rho)**2*(R0/R)**2>
            dRHOdZ, dRHOdR = np.gradient(RHORZ, self.Zin[2] - self.Zin[1], self.Rin[2] - self.Rin[1])
            dPHI2 = dRHOdZ**2 + dRHOdR**2
            dp2fun = RectBivariateSplineNaN(self.Zin, self.Rin, dPHI2)
            self['avg']['gcap'] = np.zeros((self.nc))
            for k in range(self.nc):
                self['avg']['gcap'][k] = (
                    np.sum(self.fluxexpansion_dl[k] * dp2fun.ev(self['flux'][k]['Z'], self['flux'][k]['R']) / self['flux'][k]['R'] ** 2)
                    / self.int_fluxexpansion_dl[k]
                )
            self['avg']['gcap'] *= self['RCENTR'] ** 2  # * self['avg']['1/R**2']

            # linear extrapolation
            x = self['levels'][1:]
            for item in ['gcap', 'hcap', 'fcap']:
                y = self['avg'][item][1:]
                self['avg'][item][0] = y[1] - ((y[1] - y[0]) / (x[1] - x[0])) * x[1]

        else:
            if 'rhon' in self['geo']:
                del self['geo']['rhon']

        # midplane quantities
        self['midplane']['R'] = self['geo']['R'] + self['geo']['a']
        self['midplane']['Z'] = self['midplane']['R'] * 0 + self['Z0']

        Br, Bz = self._calcBrBz()
        self['midplane']['Br'] = RectBivariateSplineNaN(self.Zin, self.Rin, Br).ev(self['midplane']['Z'], self['midplane']['R'])
        self['midplane']['Bz'] = RectBivariateSplineNaN(self.Zin, self.Rin, Bz).ev(self['midplane']['Z'], self['midplane']['R'])

        signBp = -self._cocos['sigma_rhotp'] * self._cocos['sigma_RpZ'] * np.sign(self['midplane']['Bz'])
        self['midplane']['Bp'] = signBp * np.sqrt(self['midplane']['Br'] ** 2 + self['midplane']['Bz'] ** 2)

        self['midplane']['Bt'] = []
        for k in range(self.nc):
            self['midplane']['Bt'].append(self['flux'][k]['F'] / self['midplane']['R'][k])
        self['midplane']['Bt'] = np.array(self['midplane']['Bt'])

        # ============
        # extra infos
        # ============
        self['info'] = SortedDict()

        # Normlized plasma inductance
        # * calculated using {Inductive flux usage and its optimization in tokamak operation T.C.Luce et al.} EQ (A2,A3,A4)
        # * ITER IMAS li3
        ip = self['CURRENT']
        vol = self['geo']['vol'][-1]
        dpsi = np.abs(np.gradient(self['geo']['psi']))
        r_axis = self['R0']
        a = self['geo']['a'][-1]
        if self['RCENTR'] is None:
            printw('Using magnetic axis as RCENTR of vacuum field ( BCENTR = Fpol[-1] / RCENTR)')
            r_0 = self['R0']
        else:
            r_0 = self['RCENTR']
        kappa_x = self['geo']['kap'][-1]  # should be used if
        kappa_a = vol / (2.0 * np.pi * r_0 * np.pi * a * a)
        correction_factor = (1 + kappa_x**2) / (2.0 * kappa_a)
        Bp2_vol = 0
        for k in range(self.nc):  # loop over the flux surfaces
            Bp = np.sqrt(self['flux'][k]['Br'] ** 2 + self['flux'][k]['Bz'] ** 2)
            dl = np.sqrt(np.ediff1d(self['flux'][k]['R'], to_begin=0) ** 2 + np.ediff1d(self['flux'][k]['Z'], to_begin=0) ** 2)
            Bpl = np.sum(Bp * dl * 2 * np.pi)  # integral over flux surface
            Bp2_vol += Bpl * dpsi[k]  # integral over dpsi (making it <Bp**2> * V )
        circum = np.sum(dl)  # to calculate the length of the last closed flux surface
        li_from_definition = Bp2_vol / vol / constants.mu_0 / constants.mu_0 / ip / ip * circum * circum
        # li_3_TLUCE is the same as li_3_IMAS (by numbers)
        # ali_1_EFIT is the same as li_from_definition
        self['info']['internal_inductance'] = {
            "li_from_definition": li_from_definition,
            "li_(1)_TLUCE": li_from_definition / circum / circum * 2 * vol / r_0 * correction_factor,
            "li_(2)_TLUCE": li_from_definition / circum / circum * 2 * vol / r_axis,
            "li_(3)_TLUCE": li_from_definition / circum / circum * 2 * vol / r_0,
            "li_(1)_EFIT": circum * circum * Bp2_vol / (vol * constants.mu_0 * constants.mu_0 * ip * ip),
            "li_(3)_IMAS": 2 * Bp2_vol / r_0 / ip / ip / constants.mu_0 / constants.mu_0,
        }

        # EFIT current normalization
        self['info']['J_efit_norm'] = (
            (self['RCENTR'] * self['avg']['1/R']) * self['avg']['Jt'] / (self['CURRENT'] / self['geo']['cxArea'][-1])
        )

        # open separatrix
        if self.open_sep is not None:
            try:
                self['info']['open_separatrix'] = self.sol(levels=[1], open_flx={1: self.open_sep})[0][0]
            except Exception as _excp:
                printw('Error tracing open field-line separatrix: ' + repr(_excp))
                self['info']['open_separatrix'] = _excp
            else:
                ros = self['info']['open_separatrix']['R']
                istrk = np.array([0, -1] if ros[-1] > ros[0] else [-1, 0])  # Sort it so it goes inner, then outer strk pt
                self['info']['rvsin'], self['info']['rvsout'] = ros[istrk]
                self['info']['zvsin'], self['info']['zvsout'] = self['info']['open_separatrix']['Z'][istrk]

        # primary xpoint
        i = np.argmin(np.sqrt(self['flux'][self.nc - 1]['Br'] ** 2 + self['flux'][self.nc - 1]['Bz'] ** 2))
        self['info']['xpoint'] = np.array([self['flux'][self.nc - 1]['R'][i], self['flux'][self.nc - 1]['Z'][i]])

        # identify sol regions (works for single x-point >> do not do this for double-X-point or limited cases)
        if (
            'rvsin' in self['info']
            and 'zvsin' in self['info']
            and np.sign(self.open_sep[0, 1]) == np.sign(self.open_sep[-1, 1])
            and self.open_sep[0, 1] != self.open_sep[-1, 1]
            and self.open_sep[0, 0] != self.open_sep[-1, 0]
        ):
            rx, zx = self['info']['xpoint']

            # find minimum distance between legs of open separatrix used to estimate circle radius `a`
            k = int(len(self.open_sep) // 2)
            r0 = self.open_sep[:k, 0]
            z0 = self.open_sep[:k, 1]
            r1 = self.open_sep[k:, 0]
            z1 = self.open_sep[k:, 1]
            d0 = np.sqrt((r0 - rx) ** 2 + (z0 - zx) ** 2)
            i0 = np.argmin(d0)
            d1 = np.sqrt((r1 - rx) ** 2 + (z1 - zx) ** 2)
            i1 = np.argmin(d1) + k
            a = np.sqrt((self.open_sep[i0, 0] - self.open_sep[i1, 0]) ** 2 + (self.open_sep[i0, 1] - self.open_sep[i1, 1]) ** 2)
            a *= 3

            # circle
            t = np.linspace(0, 2 * np.pi, 101)[:-1]
            r = a * np.cos(t) + rx
            z = a * np.sin(t) + zx

            # intersect open separatrix with small circle around xpoint
            circle = line_intersect(np.array([self.open_sep[:, 0], self.open_sep[:, 1]]).T, np.array([r, z]).T)

            if len(circle) == 4:

                # always sort points so that they are in [inner_strike, outer_strike, outer_midplane, inner_midplane] order
                circle0 = circle - np.array([rx, zx])[np.newaxis, :]
                # clockwise for upper Xpoint
                if zx > 0 and np.sign(circle0[0, 0] * circle0[1, 1] - circle0[1, 0] * circle0[0, 1]) > 0:
                    circle = circle[::-1]
                # counter clockwise for lower Xpoint
                elif zx < 0 and np.sign(circle0[0, 0] * circle0[1, 1] - circle0[1, 0] * circle0[0, 1]) < 0:
                    circle = circle[::-1]
                # start numbering from inner strike wall
                index = np.argmin(np.sqrt((circle[:, 0] - self['info']['rvsin']) ** 2 + (circle[:, 1] - self['info']['zvsin']) ** 2))
                circle = np.vstack((circle, circle))[index : index + 4, :]
                for k, item in enumerate(['xpoint_inner_strike', 'xpoint_outer_strike', 'xpoint_outer_midplane', 'xpoint_inner_midplane']):
                    try:
                        self['info'][item] = circle[k]
                    except IndexError:
                        printe('Error parsing %s' % item)

                # regions are defined at midway points between the open separatrix points
                regions = circle + np.diff(np.vstack((circle, circle[0])), axis=0) / 2.0
                for k, item in enumerate(['xpoint_private_region', 'xpoint_outer_region', 'xpoint_core_region', 'xpoint_inner_region']):
                    try:
                        self['info'][item] = regions[k]
                    except IndexError:
                        printe('Error parsing %s' % item)

            # logic for secondary xpoint evaluation starts here
            # find where Bz=0 on the opposite side of the primary X-point: this is xpoint2_start
            Bz_sep = self['flux'][self.nc - 1]['Bz'].copy()
            mask = self['flux'][self.nc - 1]['Z'] * np.sign(self['info']['xpoint'][1]) > 0
            Bz_sep[mask] = np.nan
            index = np.nanargmin(abs(Bz_sep))
            xpoint2_start = [self['flux'][self.nc - 1]['R'][index], self['flux'][self.nc - 1]['Z'][index]]

            # trace Bz=0 contour and find the contour line that passes closest to xpoint2_start: this is the rz_divider line
            Bz0 = contourPaths(self.Rin, self.Zin, Bz, [0], remove_boundary_points=True, smooth_factor=1)[0]
            d = []
            for item in Bz0:
                d.append(np.min(np.sqrt((item.vertices[:, 0] - xpoint2_start[0]) ** 2 + (item.vertices[:, 1] - xpoint2_start[1]) ** 2)))
            rz_divider = Bz0[np.argmin(d)].vertices

            # evaluate Br along rz_divider line and consider only side opposite side of the primary X-point
            Br_divider = RectBivariateSplineNaN(self.Zin, self.Rin, Br).ev(rz_divider[:, 1], rz_divider[:, 0])
            mask = (rz_divider[:, 1] * np.sign(self['info']['xpoint'][1])) < -abs(self['info']['xpoint'][1]) / 10.0
            Br_divider = Br_divider[mask]
            rz_divider = rz_divider[mask, :]
            if abs(rz_divider[0, 1]) > abs(rz_divider[-1, 1]):
                rz_divider = rz_divider[::-1, :]
                Br_divider = Br_divider[::-1]

            # secondary xpoint where Br flips sign
            tmp = np.where(np.sign(Br_divider) != np.sign(Br_divider)[0])[0]
            if len(tmp):
                ix = tmp[0]
                self['info']['xpoint2'] = (rz_divider[ix - 1, :] + rz_divider[ix, :]) * 0.5
            else:
                self['info']['xpoint2'] = None

        # limiter
        if (
            hasattr(self, 'rlim')
            and self.rlim is not None
            and len(self.rlim) > 3
            and hasattr(self, 'zlim')
            and self.zlim is not None
            and len(self.zlim) > 3
        ):
            self['info']['rlim'] = self.rlim
            self['info']['zlim'] = self.zlim

        if not self.quiet:
            printi('  > Took {:}'.format(datetime.datetime.now() - t0))

    def surface_integral(self, what):
        """
        Cross section integral of a quantity

        :param what: quantity to be integrated specified as array at flux surface

        :return: array of the integration from core to edge
        """
        return integrate.cumtrapz(self['avg']['vp'] * self['avg']['1/R'] * what, self['geo']['psi'], initial=0) / (2.0 * np.pi)

    def volume_integral(self, what):
        """
        Volume integral of a quantity

        :param what: quantity to be integrated specified as array at flux surface

        :return: array of the integration from core to edge
        """
        return integrate.cumtrapz(self['avg']['vp'] * what, self['geo']['psi'], initial=0)

    def _surfMean(self, what):
        tmp = np.zeros((self.nc))
        whatfun = RectBivariateSplineNaN(self.Zin, self.Rin, what)
        for k in range(self.nc):
            pt = int(np.ceil(self['flux'][k]['R'].size / 8.0))
            whatSamples = whatfun.ev(self['flux'][k]['Z'][::pt], self['flux'][k]['R'][::pt])
            tmp[k] = np.mean(whatSamples)
        return tmp

    @dynaLoad
    def plotFigure(self, *args, **kw):
        from matplotlib import pyplot

        pyplot.figure(*args)
        self.plot(**kw)

    @dynaLoad
    def plot(self, only2D=False, info=False, label=None, **kw):
        from matplotlib import pyplot

        if not self.quiet:
            printi('Plotting ...')

        if not only2D:
            tmp = []
            if 'geo' in self:
                tmp = ['geo_' + item for item in self['geo'].keys()]
            if 'avg' in self:
                tmp.extend(['avg_' + item for item in self['avg'].keys()])
            nplot = len(tmp)
            cplot = int(np.floor(np.sqrt(nplot) / 2.0) * 2)
            rplot = int(np.ceil(nplot * 1.0 / cplot))
            pyplot.subplots_adjust(wspace=0.35, hspace=0.0)

            for k, item in enumerate(tmp):
                r = int(np.floor(k * 1.0 / cplot))
                c = k - r * cplot

                if k == 0:
                    ax1 = pyplot.subplot(rplot, cplot + 2, r * (cplot + 2) + c + 1 + 2)
                    ax = ax1
                else:
                    ax = pyplot.subplot(rplot, cplot + 2, r * (cplot + 2) + c + 1 + 2, sharex=ax1)
                ax.ticklabel_format(style='sci', scilimits=(-3, 3))

                if item[:3] == 'avg' and item[4:] in self['avg']:
                    pyplot.plot(self['levels'], self['avg'][item[4:]])
                    pyplot.text(
                        0.5,
                        0.9,
                        '< ' + item[4:] + ' >',
                        horizontalalignment='center',
                        verticalalignment='top',
                        size='medium',
                        transform=ax.transAxes,
                    )

                elif item[:3] == 'geo' and item[4:] in self['geo']:
                    pyplot.plot(self['levels'], self['geo'][item[4:]])
                    pyplot.text(
                        0.5, 0.8, item[4:], horizontalalignment='center', verticalalignment='top', size='medium', transform=ax.transAxes
                    )

                if label is not None:
                    ax.lines[-1].set_label(label)

                if k < len(tmp) - cplot:
                    pyplot.setp(ax.get_xticklabels(), visible=False)
                else:
                    ax.set_xlabel('$\\psi$')

            pyplot.subplot(1, int((cplot + 2) // 2), 1)

        r = []
        z = []
        # Br=[]
        # Bz=[]
        for k in range(self.nc)[:: max([1, (self.nc - 1) // (33 - 1)])]:
            r = np.hstack((r, self['flux'][k]['R'], np.nan))
            z = np.hstack((z, self['flux'][k]['Z'], np.nan))
            # Br=np.hstack((Br,self['flux'][k]['Br'],np.nan))
            # Bz=np.hstack((Bz,self['flux'][k]['Bz'],np.nan))
        pyplot.plot(r, z, markeredgewidth=0)
        pyplot.plot(self['R0'], self['Z0'], '+', color=pyplot.gca().lines[-1].get_color())
        if 'info' in self:
            if 'rlim' in self['info'] and 'zlim' in self['info']:
                pyplot.plot(self['info']['rlim'], self['info']['zlim'], 'k')
            if (
                'open_separatrix' in self['info']
                and isinstance(self['info']['open_separatrix'], SortedDict)
                and 'R' in self['info']['open_separatrix']
                and 'Z' in self['info']['open_separatrix']
            ):
                pyplot.plot(
                    self['info']['open_separatrix']['R'], self['info']['open_separatrix']['Z'], color=pyplot.gca().lines[-1].get_color()
                )
            if info:
                for item in self['info']:
                    if item.startswith('xpoint'):
                        line = pyplot.plot(self['info'][item][0], self['info'][item][1], marker='.')
                        pyplot.text(self['info'][item][0], self['info'][item][1], item, color=line[0].get_color())
        # pyplot.quiver(r,z,Br,Bz,pivot='middle',units='xy')

        r = []
        z = []
        if 'sol' in self:
            for k in range(len(self['sol'])):
                r = np.hstack((r, self['sol'][k]['R'], np.nan))
                z = np.hstack((z, self['sol'][k]['Z'], np.nan))
        pyplot.plot(r, z, markeredgewidth=0, alpha=0.5, color=pyplot.gca().lines[-1].get_color())
        pyplot.axis('tight')
        pyplot.gca().set_aspect('equal')
        pyplot.gca().set_frame_on(False)

    def rz_miller_geometry(self, poloidal_resolution=101):
        '''
        return R,Z coordinates for all flux surfaces from miller geometry coefficients in geo
        # based on gacode/gapy/src/gapy_geo.f90

        :param poloidal_resolution: integer with number of equispaced points in toroidal angle, or array of toroidal angles

        :return: 2D arrays with (R, Z) flux surface coordinates
        '''
        if isinstance(poloidal_resolution, int):
            t0 = np.linspace(0, 2 * np.pi, poloidal_resolution)
        else:
            t0 = poloidal_resolution

        x = np.arcsin(self['geo']['delta'])

        # R
        a = t0[:, np.newaxis] + x[np.newaxis, :] * np.sin(t0[:, np.newaxis])
        r0 = self['geo']['R'][np.newaxis, :] + self['geo']['a'][np.newaxis, :] * np.cos(a)

        # Z
        a = t0[:, np.newaxis] + self['geo']['zeta'][np.newaxis, :] * np.sin(2 * t0[:, np.newaxis])
        z0 = self['geo']['Z'][np.newaxis, :] + self['geo']['kap'][np.newaxis, :] * self['geo']['a'][np.newaxis, :] * np.sin(a)

        return r0, z0

    def __deepcopy__(self, memo):
        return pickle.loads(pickle.dumps(self, pickle.HIGHEST_PROTOCOL))

    @dynaLoad
    def sol(self, levels=31, packing=3, resolution=0.01, rlim=None, zlim=None, open_flx=None):
        '''
        Trace open field lines flux surfaces in the SOL

        :param levels: where flux surfaces should be traced

           * integer number of flux surface

           * list of levels

        :param packing: if `levels` is integer, packing of flux surfaces close to the separatrix

        :param resolution: accuracy of the flux surface tracing

        :param rlim: list of R coordinates points where flux surfaces intersect limiter

        :param zlim: list of Z coordinates points where flux surfaces intersect limiter

        :param open_flx: dictionary with flux surface rhon value as keys of where to calculate SOL (passing this will not set the `sol` entry in the flux-surfaces class)

        :return: dictionary with SOL flux surface information
        '''

        # pack more surfaces near the separatrix
        if is_int(levels):
            R = np.array([self['R0'], max(self.Rin)])
            tmp = RectBivariateSplineNaN(self.Zin, self.Rin, self.PSIin).ev(R * 0 + self['Z0'], R)
            tmp = max((tmp - self.PSIaxis) / ((self.flx - self.PSIaxis) * self.maxPSI))
            levels = pack_points(levels + 2, -1, packing)
            levels = (levels - min(levels)) / (max(levels) - min(levels)) * (tmp - 1) + 1
            levels = levels[1:-1]

        if rlim is None and len(self.rlim) > 3:
            rlim = self.rlim
        if zlim is None and len(self.zlim) > 3:
            zlim = self.zlim
        if rlim is None or zlim is None:
            dd = 0.01
            rlim = [min(self.Rin) + dd, max(self.Rin) - dd, max(self.Rin) - dd, min(self.Rin) + dd, min(self.Rin) + dd]
            zlim = [min(self.Zin) + dd, min(self.Zin) + dd, max(self.Zin) - dd, max(self.Zin) - dd, min(self.Zin) + dd]

        rlim = copy.deepcopy(rlim)
        zlim = copy.deepcopy(zlim)

        store_SOL = False
        if open_flx is None:
            store_SOL = True

            # SOL requires higher resolution
            self.changeResolution(resolution)

            # find SOL fied lines
            open_flx = self._findSurfaces(levels=levels)

        # midplane
        Rm = np.array([self['R0'], max(self.Rin)])
        Zm = Rm * 0 + self['Z0']

        SOL = SortedDict()
        for k, level in enumerate(open_flx.keys()):
            SOL[k] = SortedDict()
            SOL[k]['psi'] = level * (self.flx - self.PSIaxis) * self.maxPSI + self.PSIaxis
            SOL[k]['rhon'] = level
            SOL[k]['R'] = open_flx[level][:, 0]
            SOL[k]['Z'] = open_flx[level][:, 1]

        max_mid = line_intersect(np.array([Rm, Zm]).T, np.array([rlim, zlim]).T)
        if len(max_mid):
            max_mid = max_mid[0]
        else:
            printw('fluxsurfaces: there is no intersection between the horizontal line at the magnetic axis Z and the limiter')
            max_mid = [max(Rm), np.argmax(Rm)]

        def line_split(x1, x2):
            inter_midp, index_midp = line_intersect(x1, x2, True)
            index_midp = index_midp[0][0]
            inter_midp = inter_midp[0]
            if inter_midp[0] > max_mid[0]:
                return None
            inter_wall, index_wall = line_intersect(x1, np.array([rlim, zlim]).T, True)

            i = np.array([k[0] for k in index_wall]).astype(int)
            i1 = int(np.where(i == i[np.where(i < index_midp)[0]][-1])[0][0])
            i2 = int(np.where(i == i[np.where(i >= index_midp)[0]][0])[0][0])

            legs = (
                np.vstack((x1[: index_wall[i1][0], :], inter_wall[i1])),
                np.vstack(([inter_wall[i1]], x1[index_wall[i1][0] + 1 : index_midp, :], [inter_midp])),
                np.vstack(([inter_midp], x1[index_midp + 1 : index_wall[i2][0], :], [inter_wall[i2]])),
                np.vstack(([inter_wall[i2]], x1[index_wall[i2][0] + 1 :, :])),
            )

            if legs[1][-2][1] < legs[2][1][1]:
                legs = [np.flipud(legs[3]), np.flipud(legs[2]), np.flipud(legs[1]), np.flipud(legs[0])]

            # pyplot.plot(legs[0][:,0],legs[0][:,1],'b.')
            # pyplot.plot(legs[1][:,0],legs[1][:,1],'rx')
            # pyplot.plot(legs[2][:,0],legs[2][:,1],'b.')
            # pyplot.plot(legs[3][:,0],legs[3][:,1],'rx')
            # print(map(len,legs))

            return legs

        Br, Bz = self._calcBrBz()

        Brfun = RectBivariateSplineNaN(self.Zin, self.Rin, Br)
        Bzfun = RectBivariateSplineNaN(self.Zin, self.Rin, Bz)

        valid_levels = []
        sol_levels = []
        for k in range(len(SOL)):
            try:
                legs = line_split(np.array([SOL[k]['R'], SOL[k]['Z']]).T, np.array([Rm, Zm]).T)
            except IndexError:
                for k1 in range(k, len(SOL)):
                    del SOL[k1]
                break
            if legs is not None:
                r = SOL[k]['R'] = np.hstack((legs[0][:-1, 0], legs[1][:-1, 0], legs[2][:-1, 0], legs[3][:, 0]))
                z = SOL[k]['Z'] = np.hstack((legs[0][:-1, 1], legs[1][:-1, 1], legs[2][:-1, 1], legs[3][:, 1]))
                w1 = len(legs[0][:-1, 0])
                ii = len(legs[0][:-1, 0]) + len(legs[1][:-1, 0])
                w2 = len(legs[0][:-1, 0]) + len(legs[1][:-1, 0]) + len(legs[2][:-1, 0])
                sol_levels.append(SOL[k]['rhon'])
            else:
                continue
            valid_levels.append(k)

            SOL[k]['Br'] = Brfun.ev(z, r)
            SOL[k]['Bz'] = Bzfun.ev(z, r)

            Bt = np.abs(self['BCENTR'] * self['RCENTR'] / r)
            Bp = np.sqrt(SOL[k]['Br'] ** 2 + SOL[k]['Bz'] ** 2)

            dp = np.sqrt(np.gradient(r) ** 2 + np.gradient(z) ** 2)
            pitch = np.sqrt(1 + (Bt / Bp) ** 2)
            s = np.cumsum(pitch * dp)
            s = np.abs(s - s[ii])
            SOL[k]['s'] = s

            for item in SOL[k]:
                if isinstance(SOL[k][item], np.ndarray):
                    SOL[k][item] = SOL[k][item][w1 : w2 + 1]
            SOL[k]['mid_index'] = ii - w1

            SOL[k]['rho'] = SOL[k]['rhon'] * self['geo']['rho'][-1]

        SOLV = SortedDict()
        for k, v in enumerate(valid_levels):
            SOLV[k] = SOL[v]

        if store_SOL:
            self['sol'] = SOLV
            self['sol_levels'] = np.array([s['psi'] for s in SOLV.values()])
        return SOL, sol_levels

    @dynaLoad
    def to_omas(self, ods=None, time_index=0):
        '''
        translate fluxSurfaces class to OMAS data structure

        :param ods: input ods to which data is added

        :param time_index: time index to which data is added

        :return: ODS
        '''
        if ods is None:
            ods = ODS()

        cocosio = self.cocosin

        # align psi grid
        psi = self['geo']['psi']
        if 'equilibrium.time_slice.%d.profiles_1d.psi' % time_index in ods:
            with omas_environment(ods, cocosio=cocosio):
                m0 = psi[0]
                M0 = psi[-1]
                m1 = ods['equilibrium.time_slice.%d.profiles_1d.psi' % time_index][0]
                M1 = ods['equilibrium.time_slice.%d.profiles_1d.psi' % time_index][-1]
                psi = (psi - m0) / (M0 - m0) * (M1 - m1) + m1
        coordsio = {'equilibrium.time_slice.%d.profiles_1d.psi' % time_index: psi}

        # add fluxSurfaces quantities
        with omas_environment(ods, cocosio=cocosio, coordsio=coordsio):
            eq1d = ods['equilibrium.time_slice'][time_index]['profiles_1d']
            glob = ods['equilibrium.time_slice'][time_index]['global_quantities']

            eq1d['rho_tor_norm'] = self['geo']['rhon']
            eq1d['rho_tor'] = self['geo']['rho']
            eq1d['r_inboard'] = self['geo']['R'] - self['geo']['a']
            eq1d['r_outboard'] = self['geo']['R'] + self['geo']['a']
            eq1d['volume'] = self['geo']['vol']
            eq1d['triangularity_upper'] = self['geo']['delu']
            eq1d['triangularity_lower'] = self['geo']['dell']
            # eq1d['average_outer_squareness'] = self['geo']['zeta'] # IMAS does not have a placeholder for this quantity yet
            eq1d['squareness_lower_inner'] = self['geo']['zeta']  # self['geo']['zetail']
            eq1d['squareness_upper_inner'] = self['geo']['zeta']  # self['geo']['zetaiu']
            eq1d['squareness_lower_outer'] = self['geo']['zeta']  # self['geo']['zetaol']
            eq1d['squareness_upper_outer'] = self['geo']['zeta']  # self['geo']['zetaou']
            eq1d['trapped_fraction'] = 1.0 - self['avg']['fc']
            eq1d['surface'] = self['geo']['surfArea']
            eq1d['q'] = self['avg']['q']
            eq1d['pressure'] = self['avg']['P']
            eq1d['phi'] = self['geo']['phi']
            eq1d['j_tor'] = self['avg']['Jt/R'] / self['avg']['1/R']
            eq1d['gm9'] = self['avg']['1/R']
            eq1d['gm8'] = self['avg']['R']
            eq1d['gm5'] = self['avg']['Btot**2']
            eq1d['gm2'] = self['avg']['gcap'] / self['RCENTR'] ** 2
            eq1d['gm1'] = self['avg']['1/R**2']
            eq1d['f'] = self['avg']['F']
            eq1d['elongation'] = 0.5 * (self['geo']['kapu'] + self['geo']['kapl'])
            eq1d['dvolume_drho_tor'] = deriv(self['geo']['rho'], self['geo']['vol'])
            eq1d['dvolume_dpsi'] = deriv(self['geo']['psi'], self['geo']['vol'])
            eq1d['dpsi_drho_tor'] = deriv(self['geo']['rho'], self['geo']['psi'])
            eq1d['darea_drho_tor'] = deriv(self['geo']['rho'], self['geo']['cxArea'])
            eq1d['darea_dpsi'] = deriv(self['geo']['psi'], self['geo']['cxArea'])
            tmp = [
                np.sqrt(self['flux'][k]['Br'] ** 2 + self['flux'][k]['Bz'] ** 2 + (self['flux'][k]['F'] / self['flux'][k]['R']) ** 2)
                for k in self['flux']
            ]
            eq1d['b_field_min'] = np.array([np.min(v) for v in tmp])
            eq1d['b_field_max'] = np.array([np.max(v) for v in tmp])
            eq1d['b_field_average'] = np.array([np.mean(v) for v in tmp])
            eq1d['area'] = self['geo']['cxArea']
            eq1d['geometric_axis.r'] = self['geo']['R']
            eq1d['geometric_axis.z'] = self['geo']['Z']
            eq1d['centroid.r'] = self['geo']['R_centroid']
            eq1d['centroid.z'] = self['geo']['Z_centroid']
            eq1d['centroid.r_max'] = self['geo']['Rmax_centroid']
            eq1d['centroid.r_min'] = self['geo']['Rmin_centroid']

            glob['beta_normal'] = self['avg']['beta_n'][-1]
            glob['beta_tor'] = self['avg']['beta_t'][-1]
            glob['beta_pol'] = self['avg']['beta_p'][-1]
            glob['magnetic_axis.r'] = eq1d['geometric_axis.r'][0]
            glob['magnetic_axis.z'] = eq1d['geometric_axis.z'][0]
            glob['li_3'] = self['info']['internal_inductance']['li_(3)_IMAS']
            glob['area'] = self['geo']['cxArea'][-1]

        return ods

    def from_omas(self, ods, time_index=0):
        '''
        populate fluxSurfaces class from OMAS data structure

        :param ods: input ods to which data is added

        :param time_index: time index to which data is added

        :return: ODS
        '''
        eq = ods['equilibrium.time_slice'][time_index]
        eq1d = ods['equilibrium.time_slice'][time_index]['profiles_1d']
        psin = eq1d['psi']
        psin = (psin - min(psin)) / (max(psin) - min(psin))

        if 'profiles_2d.0.b_field_tor' not in eq:
            ods.physics_equilibrium_consistent()

        rlim = 'wall.description_2d.%d.limiter.unit.0.outline.r' % time_index
        zlim = 'wall.description_2d.%d.limiter.unit.0.outline.z' % time_index
        if rlim in ods and zlim in ods:
            rlim = ods[rlim]
            zlim = ods[zlim]
        else:
            rlim = zlim = None

        self.__init__(
            Rin=eq['profiles_2d.0.r'][:, 0],
            Zin=eq['profiles_2d.0.z'][0, :],
            PSIin=eq['profiles_2d.0.psi'].T,
            Btin=eq['profiles_2d.0.b_field_tor'].T,
            Rcenter=ods['equilibrium.vacuum_toroidal_field.r0'],
            F=eq1d['f'],
            P=eq1d['pressure'],
            levels=psin,
            cocosin=ods.cocos,
            rlim=rlim,
            zlim=zlim,
            quiet=True,
        )


def boundaryShape(
    a,
    eps,
    kapu,
    kapl,
    delu,
    dell,
    zetaou,
    zetaiu,
    zetail,
    zetaol,
    zoffset,
    upnull=False,
    lonull=False,
    npts=90,
    doPlot=False,
    newsq=np.zeros(4),
    **kw,
):
    '''
    Function used to generate boundary shapes based on `T. C. Luce, PPCF, 55 9 (2013)`
    Direct Python translation of the IDL program /u/luce/idl/shapemaker3.pro

    :param a: minor radius

    :param eps: aspect ratio

    :param kapu: upper elongation

    :param lkap: lower elongation

    :param delu: upper triangularity

    :param dell: lower triangularity

    :param zetaou: upper outer squareness

    :param zetaiu: upper inner squareness

    :param zetail: lower inner squareness

    :param zetaol: lower outer squareness

    :param zoffset: z-offset

    :param upnull: toggle upper x-point

    :param lonull: toggle lower x-point

    :param npts: int
        number of points (per quadrant)

    :param doPlot: plot boundary shape construction

    :param newsq: A 4 element array, into which the new squareness values are stored

    :return: tuple with arrays of r,z,zref

    >> boundaryShape(a=0.608,eps=0.374,kapu=1.920,kapl=1.719,delu=0.769,dell=0.463,zetaou=-0.155,zetaiu=-0.255,zetail=-0.174,zetaol=-0.227,zoffset=0.000,upnull=False,lonull=False,doPlot=True)
    '''
    ukap = kapu
    lkap = kapl
    utri = delu
    ltri = dell
    uosq = zetaou
    uisq = zetaiu
    lisq = zetail
    losq = zetaol
    amin = a
    newsq[0:4] = [uosq, uisq, lisq, losq]
    if is_int(npts):
        ang = np.linspace(0, 2 * np.pi, (npts * 4 + 1))
    else:
        ang = npts

    i1 = np.where((ang >= 0 * np.pi / 2.0) & (ang < 1 * np.pi / 2.0))
    i2 = np.where((ang >= 1 * np.pi / 2.0) & (ang < 2 * np.pi / 2.0))
    i3 = np.where((ang >= 2 * np.pi / 2.0) & (ang < 3 * np.pi / 2.0))
    i4 = np.where((ang >= 3 * np.pi / 2.0) & (ang < 4 * np.pi / 2.0))

    ang1 = ang[i1]
    ang2 = ang[i2]
    ang3 = ang[i3]
    ang4 = ang[i4]

    rsr2 = 1.0 / np.sqrt(2.0)
    cc = 1.0 - rsr2

    if uosq < -1.0 * rsr2:
        uosq = -1.0 * rsr2

    if uisq < -1.0 * rsr2:
        uisq = -1.0 * rsr2

    if lisq < -1.0 * rsr2:
        lisq = -1.0 * rsr2

    if losq < -1.0 * rsr2:
        losq = -1.0 * rsr2

    # n1=-alog(2.)/alog(uosq*cc+rsr2)
    n1 = -np.log(2.0) / np.log(uosq * cc + rsr2)
    # r1=amin*(1./eps-utri)+amin*(1.+utri)*cos(ang1)
    r1 = amin * (1.0 / eps - utri) + amin * (1.0 + utri) * np.cos(ang1) ** (2.0 / n1)
    # z1=zoffset+amin*ukap*sin(ang1) ^(2./n1)
    z1 = zoffset + amin * ukap * np.sin(ang1) ** (2.0 / n1)
    # z1ref=zoffset+amin*ukap*sin(ang1)
    z1ref = zoffset + amin * ukap * np.sin(ang1)
    # n2=-alog(2.)/alog(uisq*cc+rsr2)
    n2 = -np.log(2.0) / np.log(uisq * cc + rsr2)
    # r2=amin*(1./eps-utri)-amin*(1.-utri)*abs(cos(ang2))
    r2 = amin * (1.0 / eps - utri) - amin * (1.0 - utri) * abs(np.cos(ang2)) ** (2.0 / n2)
    # z2=zoffset+amin*ukap*sin(ang2) ^(2./n2)
    z2 = zoffset + amin * ukap * np.sin(ang2) ** (2.0 / n2)
    # z2ref=zoffset+amin*ukap*sin(ang2)
    z2ref = zoffset + amin * ukap * np.sin(ang2)

    if upnull:

        ##        f=findgen(99)/100.+0.01
        f = np.linspace(0.01, 1, 100)
        ##        n=findgen(99)/25.+1.04
        n = np.linspace(1.04, 5, 100)
        ##
        ##        h1=1.-(1.-uosq)*cc
        h1 = 1.0 - (1.0 - uosq) * cc
        ##        h2=1.-(1.-uisq)*cc
        h2 = 1.0 - (1.0 - uisq) * cc
        ##
        ##        a1=amin*(1.+utri)
        a1 = amin * (1.0 + utri)
        ##        a2=amin*(1.-utri)
        a2 = amin * (1.0 - utri)
        ##        b=amin*ukap
        b = amin * ukap
        ##
        ##        if (utri ge 0.) then c1=utri-1. else c1=-1./(1.+utri)
        c1 = utri - 1 if utri >= 0 else -1.0 / (1.0 + utri)
        ##        if (utri ge 0.) then c2=-1./(1.-utri) else c2=-1.*(1.+utri)
        c2 = -1.0 / (1.0 - utri) if utri >= 0.0 else -1.0 * (1.0 + utri)
        ##
        ##        y1q1=fltarr(99,99)
        y1q1 = np.zeros((100, 100))
        ##        y2q1=fltarr(99,99)
        y2q1 = np.zeros((100, 100))
        ##        y1q2=fltarr(99,99)
        y1q2 = np.zeros((100, 100))
        ##        y2q2=fltarr(99,99)
        y2q2 = np.zeros((100, 100))
        ##
        ##        for i=0,98 do begin
        for i in range(y1q1.shape[0]):
            ##        for j=0,98 do begin
            for j in range(y1q1.shape[1]):
                ##                y1q1(i,j)=(f(i)+h1*(1.-f(i))) ^n(j)+(1.-f(i) ^n(j))*h1 ^n(j)-1.
                y1q1[j, i] = (f[i] + h1 * (1.0 - f[i])) ** n[j] + (1.0 - f[i] ** n[j]) * h1 ** n[j] - 1.0
                ##                y2q1(i,j)=f(i) ^(n(j)-1.)*(f(i)*(c1+b/a1)-b/a1)-c1
                y2q1[j, i] = f[i] ** (n[j] - 1.0) * (f[i] * (c1 + b / a1) - b / a1) - c1
                ##                y1q2(i,j)=(f(i)+h2*(1.-f(i))) ^n(j)+(1.-f(i) ^n(j))*h2^n(j)-1.
                y1q2[j, i] = (f[i] + h2 * (1.0 - f[i])) ** n[j] + (1.0 - f[i] ** n[j]) * h2 ** n[j] - 1.0
                ##                y2q2(i,j)=f(i) ^(n(j)-1.)*(f(i)*(c2+b/a2)-b/a2)-c2
                y2q2[j, i] = f[i] ** (n[j] - 1.0) * (f[i] * (c2 + b / a2) - b / a2) - c2
        ##        endfor
        ##        endfor
        ##        contour,y1q1,f,n,/overplot,level=[0.0],path_xy=xy1q1,path_info=info,closed=0,/path_data_coords
        xy1q1 = contourPaths(f, n, y1q1, [0], remove_boundary_points=True)[0][0].vertices
        ##        contour,y2q1,f,n,/overplot,level=[0.0],path_xy=xy2q1,path_info=info,closed=0,/path_data_coords
        xy2q1 = contourPaths(f, n, y2q1, [0], remove_boundary_points=True)[0][0].vertices
        ##        contour,y1q2,f,n,/overplot,level=[0.0],path_xy=xy1q2,path_info=info,closed=0,/path_data_coords
        xy1q2 = contourPaths(f, n, y1q2, [0], remove_boundary_points=True)[0][0].vertices
        ##        contour,y2q2,f,n,/overplot,level=[0.0],path_xy=xy2q2,path_info=info,closed=0,/path_data_coords
        xy2q2 = contourPaths(f, n, y2q2, [0], remove_boundary_points=True)[0][0].vertices
        ##
        ##        y1q1sol=interpol(xy1q1(1,*),xy1q1(0,*),f)
        y1q1sol = np.interp(f, xy1q1[:, 0], xy1q1[:, 1])
        ##        y2q1sol=interpol(xy2q1(1,*),xy2q1(0,*),f)
        y2q1sol = np.interp(f, xy2q1[:, 0], xy2q1[:, 1])
        ##        y1q2sol=interpol(xy1q2(1,*),xy1q2(0,*),f)
        y1q2sol = np.interp(f, xy1q2[:, 0], xy1q2[:, 1])
        ##        y2q2sol=interpol(xy2q2(1,*),xy2q2(0,*),f)
        y2q2sol = np.interp(f, xy2q2[:, 0], xy2q2[:, 1])

        ##        maxdiffq1=max(y1q1sol-y2q1sol)
        maxdiffq1 = max(y1q1sol - y2q1sol)
        ##        mindiffq1=min(y1q1sol-y2q1sol)
        mindiffq1 = min(y1q1sol - y2q1sol)
        ##        maxdiffq2=max(y1q2sol-y2q2sol)
        maxdiffq2 = max(y1q2sol - y2q2sol)
        ##        mindiffq2=min(y1q2sol-y2q2sol)
        mindiffq2 = min(y1q2sol - y2q2sol)
        ##
        ##        if (maxdiffq1/mindiffq1 lt 0.) then begin
        if maxdiffq1 / mindiffq1 < 0.0:
            ##        y12q1sol=min(abs(y1q1sol-y2q1sol),imin)
            y12q1sol = min(abs(y1q1sol - y2q1sol))
            imin = np.argmin(abs(y1q1sol - y2q1sol))
            ##        fsolq1=f(imin)
            fsolq1 = f[imin]
            ##        nsolq1=y1q1sol(imin)
            nsolq1 = y1q1sol[imin]
            ##        gsolq1=(1.-fsolq1^nsolq1)^(1./nsolq1)
            gsolq1 = (1.0 - fsolq1**nsolq1) ** (1.0 / nsolq1)
        ##        endif else begin
        else:
            ##        if (maxdiffq1 gt 0.) then begin
            if maxdiffq1 > 0:
                ##          y1new=(f[94]+h1*(1.-f[94]))^y2q1sol[94]+(1.-f[94]^y2q1sol[94])*h1^y2q1sol[94]-1.
                y1new = (f[94] + h1 * (1.0 - f[94])) ** y2q1sol[94] + (1.0 - f[94] ** y2q1sol[94]) * h1 ** y2q1sol[94] - 1.0
                ##          y2new=f[94]^(y2q1sol[94]-1.)*(f[94]*(c1+b/a1)-b/a1)-c1
                y2new = f[94] ** (y2q1sol[94] - 1.0) * (f[94] * (c1 + b / a1) - b / a1) - c1
                ##          while (y1new gt y2new) do begin
                while y1new > y2new:
                    ##            h1=h1-0.01
                    h1 = h1 - 0.01
                    ##            y1new=(f[94]+h1*(1.-f[94]))^y2q1sol[94]+(1.-f[94]^y2q1sol[94])*h1^y2q1sol[94]-1.
                    y1new = (f[94] + h1 * (1.0 - f[94])) ** y2q1sol[94] + (1.0 - f[94] ** y2q1sol[94]) * h1 ** y2q1sol[94] - 1.0
                ##          endwhile
                ##          fsolq1=f[94]
                fsolq1 = f[94]
                ##          nsolq1=y2q1sol[94]
                nsolq1 = y2q1sol[94]
                ##          gsolq1=(1.-fsolq1^nsolq1)^(1./nsolq1)
                gsolq1 = (1.0 - fsolq1**nsolq1) ** (1.0 / nsolq1)
            ##        endif else begin
            else:
                ##          y1new=(f[4]+h1*(1.-f[4]))^y2q1sol[4]+(1.-f[4]^y2q1sol[4])*h1^y2q1sol[4]-1.
                y1new = (f[4] + h1 * (1.0 - f[4])) ** y2q1sol[4] + (1.0 - f[4] ** y2q1sol[4]) * h1 ** y2q1sol[4] - 1.0
                ##          y2new=f[4]^(y2q1sol[4]-1.)*(f[4]*(c1+b/a1)-b/a1)-c1
                y2new = f[4] ** (y2q1sol[4] - 1.0) * (f[4] * (c1 + b / a1) - b / a1) - c1
                ##          while (y1new lt y2new) do begin
                while y1new < y2new:
                    ##            h1=h1+0.01
                    h1 = h1 + 0.01
                    ##            y1new=(f[4]+h1*(1.-f[4]))^y2q1sol[4]+(1.-f[4]^y2q1sol[4])*h1^y2q1sol[4]-1.
                    y1new = (f[4] + h1 * (1.0 - f[4])) ** y2q1sol[4] + (1.0 - f[4] ** y2q1sol[4]) * h1 ** y2q1sol[4] - 1.0
                ##          endwhile
                ##          fsolq1=f[4]
                fsolq1 = f[4]
                ##          nsolq1=y2q1sol[4]
                nsolq1 = y2q1sol[4]
                ##          gsolq1=(1.-fsolq1^nsolq1)^(1./nsolq1)
                gsolq1 = (1.0 - fsolq1**nsolq1) ** (1.0 / nsolq1)
            ##        endelse
            ##        sqnew1=1.-(1.-h1)/cc
            sqnew1 = 1.0 - (1.0 - h1) / cc
            newsq[0] = sqnew1
        ##        endelse
        ##
        ##        alpha1=a1/(1.-fsolq1)
        alpha1 = a1 / (1.0 - fsolq1)
        ##        beta1=b/gsolq1
        beta1 = b / gsolq1
        ##
        ##        y1=beta1*(1.-((r1-amin*(1./eps+1.))/alpha1+1.)^nsolq1)^(1./nsolq1)
        y1 = beta1 * (1.0 - ((r1 - amin * (1.0 / eps + 1.0)) / alpha1 + 1.0) ** nsolq1) ** (1.0 / nsolq1)
        ##        z1=y1+zoffset
        z1 = y1 + zoffset
        ##
        ##        if (maxdiffq2/mindiffq2 lt 0.) then begin
        if maxdiffq2 / mindiffq2 < 0.0:
            ##        y12q2sol=min(abs(y1q2sol-y2q2sol),imin)
            y12q2sol = min(abs(y1q2sol - y2q2sol))
            imin = np.argmin(abs(y1q2sol - y2q2sol))
            ##        fsolq2=f(imin)
            fsolq2 = f[imin]
            ##        nsolq2=y1q2sol(imin)
            nsolq2 = y1q2sol[imin]
            ##        gsolq2=(1.-fsolq2^nsolq2)^(1./nsolq2)
            gsolq2 = (1.0 - fsolq2**nsolq2) ** (1.0 / nsolq2)
        ##        endif else begin
        else:
            ##        if (maxdiffq2 gt 0.) then begin
            if maxdiffq2 > 0.0:
                ##          y1new=(f[94]+h2*(1.-f[94]))^y2q2sol[94]+(1.-f[94]^y2q2sol[94])*h2^y2q2sol[94]-1.
                y1new = (f[94] + h2 * (1.0 - f[94])) ** y2q2sol[94] + (1.0 - f[94] ** y2q2sol[94]) * h2 ** y2q2sol[94] - 1.0
                ##          y2new=f[94]^(y2q2sol[94]-1.)*(f[94]*(c2+b/a2)-b/a2)-c2
                y2new = f[94] ** (y2q2sol[94] - 1.0) * (f[94] * (c2 + b / a2) - b / a2) - c2
                ##          while (y1new gt y2new) do begin
                while y1new > y2new:
                    ##            h2=h2-0.01
                    h2 = h2 - 0.01
                    ##            y1new=(f[94]+h2*(1.-f[94]))^y2q2sol[94]+(1.-f[94]^y2q2sol[94])*h2^y2q2sol[94]-1.
                    y1new = (f[94] + h2 * (1.0 - f[94])) ** y2q2sol[94] + (1.0 - f[94] ** y2q2sol[94]) * h2 ** y2q2sol[94] - 1.0
                ##          endwhile
                ##          fsolq2=f[94]
                fsolq2 = f[94]
                ##          nsolq2=y2q2sol[94]
                nsolq2 = y2q2sol[94]
                ##          gsolq2=(1.-fsolq2^nsolq2)^(1./nsolq2)
                gsolq2 = (1.0 - fsolq2**nsolq2) ** (1.0 / nsolq2)
            ##          endif else begin
            else:
                ##          y1new=(f[4]+h2*(1.-f[4]))^y2q2sol[4]+(1.-f[4]^y2q2sol[4])*h2^y2q2sol[4]-1.
                y1new = (f[4] + h2 * (1.0 - f[4])) ** y2q2sol[4] + (1.0 - f[4] ** y2q2sol[4]) * h2 ** y2q2sol[4] - 1.0
                ##          y2new=f[4]^(y2q2sol[4]-1.)*(f[4]*(c2+b/a2)-b/a2)-c2
                y2new = f[4] ** (y2q2sol[4] - 1.0) * (f[4] * (c2 + b / a2) - b / a2) - c2
                ##          while (y1new lt y2new) do begin
                while y1new < y2new:
                    ##            h2=h2+0.01
                    h2 = h2 + 0.01
                    ##            y1new=(f[4]+h2*(1.-f[4]))^y2q2sol[4]+(1.-f[4]^y2q2sol[4])*h2^y2q2sol[4]-1.
                    y1new = (f[4] + h2 * (1.0 - f[4])) ** y2q2sol[4] + (1.0 - f[4] ** y2q2sol[4]) * h2 ** y2q2sol[4] - 1.0
                ##          endwhile
                ##          fsolq2=f[4]
                fsolq2 = f[4]
                ##          nsolq2=y2q2sol[4]
                nsolq2 = y2q2sol[4]
                ##          gsolq2=(1.-fsolq2^nsolq2)^(1./nsolq2)
                gsolq2 = (1.0 - fsolq2**nsolq2) ** (1.0 / nsolq2)
            ##          endelse
            ##        sqnew2=1.-(1.-h2)/cc
            sqnew2 = 1.0 - (1.0 - h2) / cc
            newsq[1] = sqnew2
        ##        endelse

        ##
        ##        alpha2=a2/(1.-fsolq2)
        alpha2 = a2 / (1.0 - fsolq2)
        ##        beta2=b/gsolq2
        beta2 = b / gsolq2
        ##
        ##        y2=beta2*(1.-(1.+(amin*(1./eps-1.)-r2)/alpha2)^nsolq2)^(1./nsolq2)
        y2 = beta2 * (1.0 - (1.0 + (amin * (1.0 / eps - 1.0) - r2) / alpha2) ** nsolq2) ** (1.0 / nsolq2)
        ##        z2=y2+zoffset
        z2 = y2 + zoffset
    ##    endif

    # n3=-alog(2.)/alog(lisq*cc+rsr2)
    n3 = -np.log(2.0) / np.log(lisq * cc + rsr2)
    # r3=amin*(1./eps-ltri)-amin*(1.-ltri)*abs(cos(ang(180:269)))
    r3 = amin * (1.0 / eps - ltri) - amin * (1.0 - ltri) * abs(np.cos(ang3)) ** (2.0 / n3)
    # z3=zoffset-amin*lkap*abs(sin(ang(180:269))) ^(2./n3)
    z3 = zoffset - amin * lkap * abs(np.sin(ang3)) ** (2.0 / n3)
    # z3ref=zoffset+amin*lkap*sin(ang(180:269))
    z3ref = zoffset + amin * lkap * np.sin(ang3)
    # n4=-alog(2.)/alog(losq*cc+rsr2)
    n4 = -np.log(2.0) / np.log(losq * cc + rsr2)
    # r4=amin*(1./eps-ltri)+amin*(1.+ltri)*cos(ang(270:359))
    r4 = amin * (1.0 / eps - ltri) + amin * (1.0 + ltri) * abs(np.cos(ang4)) ** (2.0 / n4)
    # z4=zoffset-amin*lkap*abs(sin(ang(270:359))) ^(2./n4)
    z4 = zoffset - amin * lkap * abs(np.sin(ang4)) ** (2.0 / n4)
    # z4ref=zoffset+amin*lkap*sin(ang(270:359))
    z4ref = zoffset + amin * lkap * np.sin(ang4)

    if lonull:
        # f=findgen(99)/100.+0.01
        f = np.arange(99) / 100.0 + 0.01
        # n=findgen(99)/25.+1.04
        n = np.arange(99) / 25.0 + 1.04

        # h4=1.-(1.-losq)*cc
        h4 = 1.0 - (1.0 - losq) * cc
        # h3=1.-(1.-lisq)*cc
        h3 = 1.0 - (1.0 - lisq) * cc

        # a4=amin*(1.+ltri)
        a4 = amin * (1.0 + ltri)
        # a3=amin*(1.-ltri)
        a3 = amin * (1.0 - ltri)
        # b=amin*lkap
        b = amin * lkap

        # if (ltri ge 0.) then c4=ltri-1. else c4=-1./(1.+ltri)
        c4 = ltri - 1.0 if (ltri >= 0.0) else -1.0 / (1.0 + ltri)
        # if (ltri ge 0.) then c3=-1./(1.-ltri) else c3=-1.*(1.+ltri)
        c3 = -1.0 / (1.0 - ltri) if (ltri >= 0.0) else -1.0 * (1.0 + ltri)

        # y1q4=fltarr(99,99)
        y1q4 = np.zeros((99, 99))
        # y2q4=fltarr(99,99)
        y2q4 = np.zeros((99, 99))
        # y1q3=fltarr(99,99)
        y1q3 = np.zeros((99, 99))
        # y2q3=fltarr(99,99)
        y2q3 = np.zeros((99, 99))

        # for i=0,98 do begin
        for i in range(len(f)):
            # for j=0,98 do begin
            for j in range(len(n)):
                # y1q4(i,j)=(f(i)+h4*(1.-f(i))) ^n(j)+(1.-f(i) ^n(j))*h4 ^n(j)-1.
                y1q4[j, i] = (f[i] + h4 * (1.0 - f[i])) ** n[j] + (1.0 - f[i] ** n[j]) * h4 ** n[j] - 1.0
                # y2q4(i,j)=f(i) ^(n(j)-1.)*(f(i)*(c4+b/a4)-b/a4)-c4
                y2q4[j, i] = f[i] ** (n[j] - 1.0) * (f[i] * (c4 + b / a4) - b / a4) - c4
                # y1q3(i,j)=(f(i)+h3*(1.-f(i))) ^n(j)+(1.-f(i) ^n(j))*h3 ^n(j)-1.
                y1q3[j, i] = (f[i] + h3 * (1.0 - f[i])) ** n[j] + (1.0 - f[i] ** n[j]) * h3 ** n[j] - 1.0
                # y2q3(i,j)=f(i) ^(n(j)-1.)*(f(i)*(c3+b/a3)-b/a3)-c3
                y2q3[j, i] = f[i] ** (n[j] - 1.0) * (f[i] * (c3 + b / a3) - b / a3) - c3
        # endfor
        # endfor

        # contour,y1q4,f,n,/overplot,level=[0.0],path_xy=xy1q4,path_info=info,closed=0,/path_data_coords
        xy1q4 = contourPaths(f, n, y1q4, [0], remove_boundary_points=True)[0][0].vertices
        # contour,y2q4,f,n,/overplot,level=[0.0],path_xy=xy2q4,path_info=info,closed=0,/path_data_coords
        xy2q4 = contourPaths(f, n, y2q4, [0], remove_boundary_points=True)[0][0].vertices
        # contour,y1q3,f,n,/overplot,level=[0.0],path_xy=xy1q3,path_info=info,closed=0,/path_data_coords
        xy1q3 = contourPaths(f, n, y1q3, [0], remove_boundary_points=True)[0][0].vertices
        # contour,y2q3,f,n,/overplot,level=[0.0],path_xy=xy2q3,path_info=info,closed=0,/path_data_coords
        xy2q3 = contourPaths(f, n, y2q3, [0], remove_boundary_points=True)[0][0].vertices

        # y1q4sol=interpol(xy1q4(1,*),xy1q4(0,*),f)
        y1q4sol = np.interp(f, xy1q4[:, 0], xy1q4[:, 1])
        # y2q4sol=interpol(xy2q4(1,*),xy2q4(0,*),f)
        y2q4sol = np.interp(f, xy2q4[:, 0], xy2q4[:, 1])
        # y1q3sol=interpol(xy1q3(1,*),xy1q3(0,*),f)
        y1q3sol = np.interp(f, xy1q3[:, 0], xy1q3[:, 1])
        # y2q3sol=interpol(xy2q3(1,*),xy2q3(0,*),f)
        y2q3sol = np.interp(f, xy2q3[:, 0], xy2q3[:, 1])

        # maxdiffq4=max(y1q4sol-y2q4sol)
        maxdiffq4 = max(y1q4sol - y2q4sol)
        # mindiffq4=min(y1q4sol-y2q4sol)
        mindiffq4 = min(y1q4sol - y2q4sol)
        # maxdiffq3=max(y1q3sol-y2q3sol)
        maxdiffq3 = max(y1q3sol - y2q3sol)
        # mindiffq3=min(y1q3sol-y2q3sol)
        mindiffq3 = min(y1q3sol - y2q3sol)

        # if (maxdiffq4/mindiffq4 lt 0.) then begin
        if maxdiffq4 / mindiffq4 < 0.0:
            # y12q4sol=min(abs(y1q4sol-y2q4sol),imin)
            y12q4sol = min(abs(y1q4sol - y2q4sol))
            imin = np.argmin(abs(y1q4sol - y2q4sol))
            # fsolq4=f(imin)
            fsolq4 = f[imin]
            # nsolq4=y1q4sol(imin)
            nsolq4 = y1q4sol[imin]
            # gsolq4=(1.-fsolq4 ^nsolq4) ^(1./nsolq4)
            gsolq4 = (1.0 - fsolq4**nsolq4) ** (1.0 / nsolq4)
        # endif else begin
        else:
            # if (maxdiffq4 gt 0.) then begin
            if maxdiffq4 > 0.0:
                # y1new=(f(94)+h4*(1.-f(94))) ^y2q4sol(94)+(1.-f(94) ^y2q4sol(94))*h4 ^y2q4sol(94)-1.
                y1new = (f[94] + h4 * (1.0 - f[94])) ** y2q4sol[94] + (1.0 - f[94] ** y2q4sol[94]) * h4 ** y2q4sol[94] - 1.0
                # y2new=f(94) ^(y2q4sol(94)-1.)*(f(94)*(c4+b/a4)-b/a4)-c4
                y2new = f[94] ** (y2q4sol[94] - 1.0) * (f[94] * (c4 + b / a4) - b / a4) - c4
                # while (y1new gt y2new) do begin
                while y1new > y2new:
                    # h4=h4-0.01
                    h4 = h4 - 0.01
                    # y1new=(f(94)+h4*(1.-f(94))) ^y2q4sol(94)+(1.-f(94) ^y2q4sol(94))*h4 ^y2q4sol(94)-1.
                    y1new = (f[94] + h4 * (1.0 - f[94])) ** y2q4sol[94] + (1.0 - f[94] ** y2q4sol[94]) * h4 ** y2q4sol[94] - 1.0
                # endwhile

                # fsolq4=f(94)
                fsolq4 = f[94]
                # nsolq4=y2q4sol(94)
                nsolq4 = y2q4sol[94]
                # gsolq4=(1.-fsolq4 ^nsolq4) ^(1./nsolq4)
                gsolq4 = (1.0 - fsolq4**nsolq4) ** (1.0 / nsolq4)
            # endif else begin
            else:
                # y1new=(f(4)+h4*(1.-f(4))) ^y2q4sol(4)+(1.-f(4) ^y2q4sol(4))*h4 ^y2q4sol(4)-1.
                y1new = (f[4] + h4 * (1.0 - f[4])) ** y2q4sol[4] + (1.0 - f[4] ** y2q4sol[4]) * h4 ** y2q4sol[4] - 1.0
                # y2new=f(4) ^(y2q4sol(4)-1.)*(f(4)*(c4+b/a4)-b/a4)-c4
                y2new = f[4] ** (y2q4sol[4] - 1.0) * (f[4] * (c4 + b / a4) - b / a4) - c4
                # while (y1new lt y2new) do begin
                while y1new < y2new:
                    # h4=h4+0.01
                    h4 = h4 + 0.01
                    # y1new=(f(4)+h4*(1.-f(4))) ^y2q4sol(4)+(1.-f(4) ^y2q4sol(4))*h4 ^y2q4sol(4)-1.
                    y1new = (f[4] + h4 * (1.0 - f[4])) ** y2q4sol[4] + (1.0 - f[4] ** y2q4sol[4]) * h4 ** y2q4sol[4] - 1.0
                # endwhile

                # fsolq4=f(4)
                fsolq4 = f[4]
                # nsolq4=y2q4sol(4)
                nsolq4 = y2q4sol[4]
                # gsolq4=(1.-fsolq4 ^nsolq4) ^(1./nsolq4)
                gsolq4 = (1.0 - fsolq4**nsolq4) ** (1.0 / nsolq4)
            # endelse

            # sqnew4=1.-(1.-h4)/cc
            sqnew4 = 1.0 - (1.0 - h4) / cc
            newsq[3] = sqnew4
        # endelse

        # alpha4=a4/(1.-fsolq4)
        alpha4 = a4 / (1.0 - fsolq4)
        # beta4=b/gsolq4
        beta4 = b / gsolq4

        # y4=-1.*(beta4*(1.-((r4-amin*(1./eps+1.))/alpha4+1.) ^nsolq4) ^(1./nsolq4))
        y4 = -1.0 * (beta4 * (1.0 - ((r4 - amin * (1.0 / eps + 1.0)) / alpha4 + 1.0) ** nsolq4) ** (1.0 / nsolq4))
        # z4=y4+zoffset
        z4 = y4 + zoffset

        # if (maxdiffq3/mindiffq3 lt 0.) then begin
        if maxdiffq3 / mindiffq3 < 0.0:
            # y12q3sol=min(abs(y1q3sol-y2q3sol),imin)
            y12q3sol = min(abs(y1q3sol - y2q3sol))
            imin = np.argmin(abs(y1q3sol - y2q3sol))
            # fsolq3=f(imin)
            fsolq3 = f[imin]
            # nsolq3=y1q3sol(imin)
            nsolq3 = y1q3sol[imin]
            # gsolq3=(1.-fsolq3 ^nsolq3) ^(1./nsolq3)
            gsolq3 = (1.0 - fsolq3**nsolq3) ** (1.0 / nsolq3)
        # endif else begin
        else:
            # if (maxdiffq3 gt 0.) then begin
            if maxdiffq3 > 0.0:
                # y1new=(f(94)+h3*(1.-f(94))) ^y2q3sol(94)+(1.-f(94) ^y2q3sol(94))*h3 ^y2q3sol(94)-1.
                y1new = (f[94] + h3 * (1.0 - f[94])) ** y2q3sol[94] + (1.0 - f[94] ** y2q3sol[94]) * h3 ** y2q3sol[94] - 1.0
                # y2new=f(94) ^(y2q3sol(94)-1.)*(f(94)*(c3+b/a3)-b/a3)-c3
                y2new = f[94] ** (y2q3sol[94] - 1.0) * (f[94] * (c3 + b / a3) - b / a3) - c3
                # while (y1new gt y2new) do begin
                while y1new > y2new:
                    # h3=h3-0.01
                    h3 = h3 - 0.01
                    # y1new=(f(94)+h3*(1.-f(94))) ^y2q3sol(94)+(1.-f(94) ^y2q3sol(94))*h3 ^y2q3sol(94)-1.
                    y1new = (f[94] + h3 * (1.0 - f[94])) ** y2q3sol[94] + (1.0 - f[94] ** y2q3sol[94]) * h3 ** y2q3sol[94] - 1.0
                # endwhile

                # fsolq3=f(94)
                fsolq3 = f[94]
                # nsolq3=y2q3sol(94)
                nsolq3 = y2q3sol[94]
                # gsolq3=(1.-fsolq3 ^nsolq3) ^(1./nsolq3)
                gsolq3 = (1.0 - fsolq3**nsolq3) ** (1.0 / nsolq3)
            # endif else begin
            else:
                # y1new=(f(4)+h3*(1.-f(4))) ^y2q3sol(4)+(1.-f(4) ^y2q3sol(4))*h3 ^y2q3sol(4)-1.
                y1new = (f[4] + h3 * (1.0 - f[4])) ** y2q3sol[4] + (1.0 - f[4] ** y2q3sol[4]) * h3 ** y2q3sol[4] - 1.0
                # y2new=f(4) ^(y2q3sol(4)-1.)*(f(4)*(c3+b/a3)-b/a3)-c3
                y2new = f[4] ** (y2q3sol[4] - 1.0) * (f[4] * (c3 + b / a3) - b / a3) - c3
                # while (y1new lt y2new) do begin
                while y1new < y2new:
                    # h3=h3+0.01
                    h3 = h3 + 0.01
                    # y1new=(f(4)+h3*(1.-f(4))) ^y2q3sol(4)+(1.-f(4) ^y2q3sol(4))*h3 ^y2q3sol(4)-1.
                    y1new = (f[4] + h3 * (1.0 - f[4])) ** y2q3sol[4] + (1.0 - f[4] ** y2q3sol[4]) * h3 ** y2q3sol[4] - 1.0
                # endwhile

                # fsolq3=f(4)
                fsolq3 = f[4]
                # nsolq3=y2q3sol(4)
                nsolq3 = y2q3sol[4]
                # gsolq3=(1.-fsolq3 ^nsolq3) ^(1./nsolq3)
                gsolq3 = (1.0 - fsolq3**nsolq3) ** (1.0 / nsolq3)
            # endelse

            # sqnew3=1.-(1.-h3)/cc
            sqnew3 = 1.0 - (1.0 - h3) / cc
            newsq[2] = sqnew3
        # endelse

        # alpha3=a3/(1.-fsolq3)
        alpha3 = a3 / (1.0 - fsolq3)
        # beta3=b/gsolq3
        beta3 = b / gsolq3

        # y3=-1.*(beta3*(1.-(1.+(amin*(1./eps-1.)-r3)/alpha3) ^nsolq3) ^(1./nsolq3))
        y3 = -1.0 * (beta3 * (1.0 - (1.0 + (amin * (1.0 / eps - 1.0) - r3) / alpha3) ** nsolq3) ** (1.0 / nsolq3))
        # z3=y3+zoffset
        z3 = y3 + zoffset

    # r=             [r1,r2,r3,r4]
    r = np.hstack((r1, r2, r3, r4))
    # z=             [z1,z2,z3,z4]
    z = np.hstack((z1, z2, z3, z4))
    # zref=             [z1ref,z2ref,z3ref,z4ref]
    zref = np.hstack((z1ref, z2ref, z3ref, z4ref))

    if doPlot:
        pyplot.plot(r, z, '-')
        pyplot.gca().set_aspect('equal')

    return r, z, zref


class BoundaryShape(SortedDict):
    """
    Class used to generate boundary shapes based on `T. C. Luce, PPCF, 55 9 (2013)`
    """

    def __init__(
        self,
        a=None,
        eps=None,
        kapu=None,
        kapl=None,
        delu=None,
        dell=None,
        zetaou=None,
        zetaiu=None,
        zetail=None,
        zetaol=None,
        zoffset=None,
        upnull=None,
        lonull=None,
        rbbbs=None,
        zbbbs=None,
        rlim=None,
        zlim=None,
        gEQDSK=None,
        npts=90,
    ):
        """
        :param a: minor radius

        :param eps: inverse aspect ratio (a/R)

        :param kapu: upper elongation

        :param kapl: lower elongation

        :param delu: upper triangularity

        :param dell: lower triangularity

        :param zetaou: upper outer squareness

        :param zetaiu: upper inner squareness

        :param zetail: lower inner squareness

        :param zetaol: lower outer squareness

        :param zoffset: z-offset

        :param upnull: toggle upper x-point

        :param lonull: toggle lower x-point

        :param rbbbs: R boundary points

        :param zbbbs: Z boundary points

        :param rlim: R limiter points

        :param zlim: Z limiter points

        :param npts: int
            number of points (per quadrant)

        :param doPlot: plot boundary shape

        :return: tuple with arrays of r,z,zref

        >> BoundaryShape(a=0.608,eps=0.374,kapu=1.920,kapl=1.719,delu=0.769,dell=0.463,zetaou=-0.155,zetaiu=-0.255,zetail=-0.174,zetaol=-0.227,zoffset=0.000,doPlot=True)
        """

        if gEQDSK is not None:
            rbbbs = gEQDSK['RBBBS']
            zbbbs = gEQDSK['ZBBBS']
            rlim = gEQDSK['RLIM']
            zlim = gEQDSK['ZLIM']

        self['RBBBS'] = rbbbs
        self['ZBBBS'] = zbbbs

        self['RLIM'] = rlim
        self['ZLIM'] = zlim

        if rbbbs is None or zbbbs is None:
            if upnull is None:
                upnull = False
            if lonull is None:
                lonull = False
        self['upnull'] = upnull
        self['lonull'] = lonull

        if rbbbs is not None and zbbbs is not None:
            self.sameBoundaryShapeAs(rbbbs=rbbbs, zbbbs=zbbbs, upnull=upnull, lonull=lonull)

        else:
            self['a'] = a
            self['eps'] = eps
            self['kapu'] = kapu
            self['kapl'] = kapl
            self['delu'] = delu
            self['dell'] = dell
            self['zetaou'] = zetaou
            self['zetaiu'] = zetaiu
            self['zetail'] = zetail
            self['zetaol'] = zetaol
            self['zoffset'] = zoffset

        self._calc(npts)

    def _calc(self, npts=90):
        kw = {}
        argspec = inspect.getfullargspec(boundaryShape)
        for k in argspec.args:
            if k in self:
                kw[k] = self[k]
        newsq = np.zeros(4)
        r, z, zref = boundaryShape(npts=npts, newsq=newsq, **kw)
        for si, sq in enumerate(['zetaou', 'zetaiu', 'zetail', 'zetaol']):
            self[sq] = newsq[si]
        if 'r' not in self or len(r) != len(self['r']):
            self['r'] = r
        else:
            self['r'][:] = r
        if 'z' not in self or len(z) != len(self['z']):
            self['z'] = z
        else:
            self['z'][:] = z
        if 'zref' not in self or len(zref) != len(self['zref']):
            self['zref'] = zref
        else:
            self['zref'][:] = zref

        # bounds
        rsr2 = 1.0 / np.sqrt(2.0)
        self['bounds'] = SortedDict()
        self['bounds']['a'] = [self['a'] / 2.0, self['a'] * 2]
        self['bounds']['eps'] = [self['eps'] / 2.0, self['eps'] * 2]
        self['bounds']['kapu'] = [0.5, 2.5]
        self['bounds']['kapl'] = [0.5, 2.5]
        self['bounds']['delu'] = [-1 + 1e-9, 1 - 1e-9]
        self['bounds']['dell'] = [-1 + 1e-9, 1 - 1e-9]
        self['bounds']['zetaou'] = [-rsr2, rsr2]
        self['bounds']['zetaiu'] = [-rsr2, rsr2]
        self['bounds']['zetail'] = [-rsr2, rsr2]
        self['bounds']['zetaol'] = [-rsr2, rsr2]
        self['bounds']['zoffset'] = [-self['a'], self['a']]

        return r, z, zref

    def plot(self, fig=None):
        from matplotlib import pyplot
        from matplotlib.widgets import CheckButtons, Button, TextBox, Slider

        orig = SortedDict()
        for k in self['bounds'].keys() + ['upnull', 'lonull']:
            orig[k] = self[k]

        if fig is None:
            fig = pyplot.gcf()

        ax = pyplot.subplot(1, 2, 1)
        line_rz, line_bbbs, line_lim = self._plot(ax)

        def update(varName=None):
            pv = {}
            if translator.get(varName, varName) in self:
                pv[translator.get(varName, varName)] = not self[translator.get(varName, varName)]
            else:
                for varName in self['bounds'].keys():
                    pv[varName] = ctrl[varName].val
            self.update(pv)
            try:
                self._calc()
            except IndexError:
                pass
            else:
                line_rz[0].set_data(self['r'], self['z'])
                fig.canvas.draw()
                if 'rootGUI' in OMFITaux and OMFITaux['rootGUI']:
                    OMFITaux['rootGUI'].event_generate("<<update_treeGUI>>")

        translator = {
            'eps': 'a/R',
            'zoffset': 'z_{\\rm offset}',
            'kapu': '\\kappa_{\\rm U}',
            'kapl': '\\kappa_{\\rm L}',
            'delu': '\\delta_{\\rm U}',
            'dell': '\\delta_{\\rm L}',
            'zetaou': '\\zeta_{\\rm OU}',
            'zetaiu': '\\zeta_{\\rm IU}',
            'zetaol': '\\zeta_{\\rm OL}',
            'zetail': '\\zeta_{\\rm IL}',
            'Upper X point': 'upnull',
            'Lower X point': 'lonull',
        }

        ax_ctrl = {}
        ctrl = {}
        for k, varName in enumerate(
            reversed(
                ['a', 'eps', 'zoffset'] + ['kapu', 'delu', 'zetaou', 'zetaiu'] + ['kapl', 'dell', 'zetail', 'zetaol'] + ['upnull', 'lonull']
            )
        ):
            if varName not in ['upnull', 'lonull']:
                ax_ctrl[varName] = pyplot.axes(
                    [0.6, 0.1 + k / float(len(self['bounds']) + 2) * 0.85, 0.3, 0.9 / float(4 + len(self['bounds'].keys()))]
                )
                limits = self['bounds'][varName]
                ctrl[varName] = Slider(
                    ax_ctrl[varName], '$%s$' % translator.get(varName, varName), limits[0], limits[1], valinit=self[varName]
                )
                ctrl[varName].on_changed(update)

        check_ax = pyplot.axes([0.58, 0.02, 0.3, 3 / float(4 + len(self['bounds'].keys()))])
        check_ax.set_frame_on(False)
        ctrl['null'] = CheckButtons(check_ax, ['Upper X point', 'Lower X point'], [self['upnull'], self['lonull']])
        ctrl['null'].on_clicked(update)

        if 'RBBBS' in self and 'ZBBBS' in self and self['RBBBS'] is not None and self['ZBBBS'] is not None:
            bt_ax = pyplot.axes([0.75, 0.13, 0.15, 0.6 / float(4 + len(self['bounds'].keys()))])
            ctrl['measure'] = Button(bt_ax, 'measure')

            def measure(dummy=None):
                print('measuring boundary...')
                self.sameBoundaryShapeAs()
                for varName in self['bounds'].keys():
                    ctrl[varName].val = self[varName]
                update()

            ctrl['measure'].on_clicked(measure)

            bt_ax = pyplot.axes([0.75, 0.07, 0.15, 0.6 / float(4 + len(self['bounds'].keys()))])
            ctrl['fit'] = Button(bt_ax, 'fit')

            def fit(dummy=None):
                print('fitting boundary...')
                self.fitBoundaryShape()
                for varName in self['bounds'].keys():
                    ctrl[varName].val = self[varName]
                update()

            ctrl['fit'].on_clicked(fit)

        update()

        return ax

    def _plot(self, ax=None):
        from matplotlib import pyplot

        if ax is None:
            ax = pyplot.gca()
        self._calc()
        line_bbbs = None
        if 'RBBBS' in self and 'ZBBBS' in self and self['RBBBS'] is not None and self['ZBBBS'] is not None:
            line_bbbs = ax.plot(self['RBBBS'], self['ZBBBS'], '--')
        line_lim = None
        if 'RLIM' in self and 'ZLIM' in self and self['RLIM'] is not None and self['ZLIM'] is not None:
            line_lim = ax.plot(self['RLIM'], self['ZLIM'], '-', color='k', linewidth=1.5)
        line_rz = ax.plot(self['r'], self['z'], '-')
        ax.set_aspect('equal')
        return line_rz, line_bbbs, line_lim

    def sameBoundaryShapeAs(self, rbbbs=None, zbbbs=None, upnull=None, lonull=None, gEQDSK=None, verbose=None, npts=90):
        """
        Measure boundary shape from input

        :param rbbbs: array of R points to match

        :param zbbbs: array of Z points to match

        :param upnull: upper x-point

        :param lonull: lower x-point

        :param gEQDSK: input gEQDSK to match (wins over rbbbs and zbbbs)

        :param verbose: print debug statements

        :param npts: int
            Number of points

        :return: dictionary with parameters to feed to the boundaryShape function
            [a, eps, kapu, kapl, delu, dell, zetaou, zetaiu, zetail, zetaol, zoffset, upnull, lonull]
        """
        if gEQDSK is not None:
            rbbbs = gEQDSK['RBBBS']
            zbbbs = gEQDSK['ZBBBS']
        if rbbbs is None and 'RBBBS' in self:
            rbbbs = self['RBBBS']
        if zbbbs is None and 'ZBBBS' in self:
            zbbbs = self['ZBBBS']
        self['RBBBS'] = rbbbs
        self['ZBBBS'] = zbbbs

        # measure
        geo = fluxGeo(rbbbs, zbbbs, True)
        kapu = (max(zbbbs) - geo['Z']) / geo['a']
        kapl = (geo['Z'] - min(zbbbs)) / geo['a']
        p = np.array(
            [
                geo['a'],  # 0 a
                geo['a'] / geo['R'],  # 1 eps
                geo['kapu'],  # 2 ku
                geo['kapl'],  # 3 kl
                geo['delu'],  # 4 du
                geo['dell'],  # 5 dl
                geo['zetaou'],  # 6 dou
                geo['zetaiu'],  # 7 diu
                geo['zetail'],  # 8 dil
                geo['zetaol'],  # 9 dol
                geo['zoffset'],
            ]
        )  # 10 z

        argspec = inspect.getfullargspec(boundaryShape)

        # update keys
        kw = OrderedDict()
        for k in range(len(p)):
            kw[argspec.args[k]] = p[k]
            if verbose:
                print(argspec.args[k].ljust(8) + ': % 3.3f  <-- % 3.3f --> % 3.3f' % (b[k][0], p[k], b[k][1]))
        self.update(kw)

        self['upnull'] = geo['upnull']
        self['lonull'] = geo['lonull']

        self._calc(npts=npts)

        return kw

    def fitBoundaryShape(self, rbbbs=None, zbbbs=None, upnull=None, lonull=None, gEQDSK=None, verbose=None, precision=1e-3, npts=90):
        """
        Fit boundary shape from input

        :param rbbbs: array of R points to match

        :param zbbbs: array of Z points to match

        :param upnull: upper x-point

        :param lonull: lower x-point

        :param gEQDSK: input gEQDSK to match (wins over rbbbs and zbbbs)

        :param verbose: print debug statements

        :param doPlot: visualize match

        :param precision: optimization tolerance

        :param npts: int
            Number of points

        :return: dictionary with parameters to feed to the boundaryShape function
            [a, eps, kapu, kapl, delu, dell, zetaou, zetaiu, zetail, zetaol, zoffset, upnull, lonull]
        """

        if gEQDSK is not None:
            rbbbs = gEQDSK['RBBBS']
            zbbbs = gEQDSK['ZBBBS']
        if rbbbs is None and 'RBBBS' in self:
            rbbbs = self['RBBBS']
        if zbbbs is None and 'ZBBBS' in self:
            zbbbs = self['ZBBBS']
        self['RBBBS'] = rbbbs
        self['ZBBBS'] = zbbbs

        if upnull is None and self['upnull'] is not None:
            upnull = self['upnull']
        if lonull is None and self['lonull'] is not None:
            lonull = self['lonull']
        self['upnull'] = upnull
        self['lonull'] = lonull

        rbbbs = rbbbs[:: int(len(rbbbs) // 200) + 1]
        zbbbs = zbbbs[:: int(len(zbbbs) // 200) + 1]
        rnrm = np.nanmax(rbbbs) - np.nanmin(rbbbs)
        znrm = np.nanmax(zbbbs) - np.nanmin(zbbbs)

        def _sameShape(p):
            r, z, zref = boundaryShape(*p, upnull=upnull, lonull=lonull, doPlot=False, npts=npts0)
            ax = 1

            dR = ((r[np.newaxis, :] - rbbbs[:, np.newaxis]) / rnrm) ** 2
            dZ = ((z[np.newaxis, :] - zbbbs[:, np.newaxis]) / znrm) ** 2
            cost = nansum(np.min(np.sqrt(dR + dZ), ax)) ** 2

            if verbose:
                print('%3.3e' % cost, map(lambda x: '%+3.3f' % x, p))
            return cost

        # initial guess
        self.sameBoundaryShapeAs(rbbbs, zbbbs)
        p = []
        for k in ['a', 'eps', 'kapu', 'kapl', 'delu', 'dell', 'zetaou', 'zetaiu', 'zetail', 'zetaol', 'zoffset']:
            p.append(self[k])
        p = np.array(p)

        # fit
        npts0 = 100
        tmp = scipy.optimize.minimize(_sameShape, p, method='Nelder-Mead', options={'xtol': precision, 'ftol': 1e-10})
        p = tmp['x']

        argspec = inspect.getfullargspec(boundaryShape)

        kw = OrderedDict()
        for k in range(len(p)):
            kw[argspec.args[k]] = p[k]
            if verbose:
                print(argspec.args[k].ljust(8) + ': % 3.3f  <-- % 3.3f --> % 3.3f' % (b[k][0], p[k], b[k][1]))

        self.update(kw)

        self._calc(npts=npts)

        return kw


def rz_miller(a, R, kappa=1.0, delta=0.0, zeta=0.0, zmag=0.0, poloidal_resolution=101):
    """
    return R,Z coordinates for all flux surfaces from miller geometry coefficients in input.profiles file
    based on gacode/gapy/src/gapy_geo.f90

    :param a: minor radius

    :param R: major radius

    :param kappa: elongation

    :param delta: triandularity

    :param zeta: squareness

    :param zmag: z offset

    :param poloidal_resolution: integer with number of equispaced points in toroidal angle, or array of toroidal angles

    :return: 1D arrays with (R, Z) flux surface coordinates
    """
    if isinstance(poloidal_resolution, int):
        t0 = np.linspace(0, 2 * np.pi, poloidal_resolution)
    else:
        t0 = poloidal_resolution

    # R
    tmp = t0 + np.arcsin(delta) * np.sin(t0)
    r0 = R + a * np.cos(tmp)

    # Z
    tmp = t0 + zeta * np.sin(2 * t0)
    z0 = zmag + kappa * a * np.sin(tmp)

    return r0, z0


def miller_derived(rmin, rmaj, kappa, delta, zeta, zmag, q):
    '''
    Originally adapted by A. Tema from FORTRAN of gacode/shared/GEO/GEO_do.f90

    :param rmin: minor radius

    :param rmaj: major radius

    :param kappa: elongation

    :param delta: triangularity

    :param zeta: squareness

    :param zmag: z magnetic axis

    :param q: safety factor

    :return: dictionary with volume, grad_r0, bp0, bt0
    '''

    # -------------------------------------------------------------------------
    # CALCULATION OF GEOMETRY COEFFCIENTS FOR LOCAL EQUILIBRIUM MODEL.
    # here the Generalize Miller-type parametrization is implemented.
    # ---------------------------------------------------------------------------

    # This part should be turned in  a function, and consitute the basis of the EXPRO mapping.
    GEO_ntheta_in = 1001  # number of theta interpolation nodes use less if slow. use odd number.!
    n_theta = GEO_ntheta_in
    pi_2 = 8.0 * np.arctan(1.0)
    i = range(0, n_theta - 1)
    d_theta = pi_2 / (n_theta - 1)
    # ------------------------------------------------------------------
    # Compute fundamental geometric quantities (basic derivatives like dl/dt) and metric elements (g_tt).
    thetav = [-0.5 * pi_2 + (i - 1) * d_theta for i in range(n_theta - 1)]
    theta0 = thetav[-1]
    # -----------------------------------------------------------------
    #    Generalized Miller-type parameterization
    # -----------------------------------------------------------------
    # nexp is the grid size
    nexp = len(rmin)
    GEO_delta_in = delta
    GEO_rmin_in = rmin.copy()
    GEO_rmaj_in = rmaj
    GEO_drmaj_in = deriv(GEO_rmin_in, GEO_rmaj_in)  # dR_0/dr
    GEO_s_delta_in = GEO_rmin_in * deriv(GEO_rmin_in, GEO_delta_in)  # r*d(delta)/dr
    GEO_zeta_in = zeta
    GEO_zmag_in = zmag  # geo_zoffset
    GEO_kappa_in = kappa  # geo_kap
    GEO_dzmag_in = deriv(GEO_rmin_in, zmag)  # dz_0/dr
    GEO_s_kappa_in = (GEO_rmin_in / GEO_kappa_in) * deriv(GEO_rmin_in, GEO_kappa_in)  # (r/kappa) d(kappa)/dr
    GEO_s_zeta_in = GEO_rmin_in * deriv(GEO_rmin_in, GEO_zeta_in)  # r d(zeta)/dr
    GEO_q_in = q

    GEO_grad_r0 = []
    GEO_volume = []
    GEO_bp0 = []
    GEO_bt0 = []
    for kk in range(nexp):
        # ! A
        # ! dA/dtheta
        # ! d^2A/dtheta^2
        x = np.arcsin(GEO_delta_in[kk])
        a = [theta + x * np.sin(theta) for theta in thetav]
        a_t = [1.0 + x * np.cos(theta) for theta in thetav]
        a_tt = [-x * np.sin(theta) for theta in thetav]
        a_t2 = [element**2 for element in a_t]
        # ! R(theta)
        # ! dR/dr
        # ! dR/dtheta
        # ! d^2R/dtheta^2
        bigr = GEO_rmaj_in[kk] + GEO_rmin_in[kk] * np.cos(a)
        bigr_r = GEO_drmaj_in[kk] + np.cos(a) - GEO_s_delta_in[kk] / np.cos(x) * np.sin(theta0) * np.sin(a)
        bigr_t = -GEO_rmin_in[kk] * np.array(a_t * np.sin(a))
        # bigr_t2=[element**2 for element in bigr_t]
        bigr_tt = -GEO_rmin_in[kk] * np.array(a_t2 * np.cos(a)) - GEO_rmin_in[kk] * np.array(a_tt * np.sin(a))
        # -----------------------------------------------------------
        # ! A
        # ! dA/dtheta
        # ! d^2A/dtheta^2
        a = [theta + GEO_zeta_in[kk] * np.sin(2 * theta) for theta in thetav]
        a_t = [1.0 + 2 * GEO_zeta_in[kk] * np.cos(2 * theta) for theta in thetav]
        a_tt = [-4 * GEO_zeta_in[kk] * np.sin(2 * theta) for theta in thetav]
        a_t2 = [element**2 for element in a_t]
        # ! Z(theta)
        #    ! dZ/dr
        #    ! dZ/dtheta
        #    ! d^2Z/dtheta^2
        bigz = GEO_zmag_in[kk] + GEO_kappa_in[kk] * GEO_rmin_in[kk] * np.sin(a)
        bigz_r = (
            GEO_dzmag_in[kk]
            + GEO_kappa_in[kk] * (1.0 + GEO_s_kappa_in[kk]) * np.sin(a)
            + GEO_kappa_in[kk] * GEO_s_zeta_in[kk] * np.cos(a) * np.sin(2 * theta0)
        )
        bigz_t = GEO_kappa_in[kk] * GEO_rmin_in[kk] * np.cos(a) * a_t
        # bigz_t2=[element**2 for element in bigz_t]
        bigz_tt = -GEO_kappa_in[kk] * GEO_rmin_in[kk] * np.sin(a) * a_t2 + GEO_kappa_in[kk] * GEO_rmin_in[kk] * np.cos(a) * a_tt
        #
        g_tt = bigr_t**2 + bigz_t**2
        jac_r = bigr * (bigr_r * bigz_t - bigr_t * bigz_r)
        if rmin[kk] == 0:
            grad_r = bigr * 0 + 1
        else:
            grad_r = bigr * np.sqrt(g_tt) / jac_r

        l_t = np.sqrt(g_tt)

        # 1/(du/dl)
        # r_c=l_t**3/(bigr_t*bigz_tt-bigz_t*bigr_tt)

        # ! cos(u)
        # bigz_l= bigz_t/l_t
        # nsin=(bigr_r*bigr_t+bigz_r*bigz_t)/l_t

        # ---------------------------------------------
        # Loop integral (1 to n_theta-1) to compute f
        # This Step above is the only justification why we carried all the poloidal points.
        # Since we just need bp0,bt0,grad_r0 at theta =0, we needed the others angles for the INTEGRAL HERE.!

        ccc = np.sum(l_t[:-1] / (bigr[:-1] * grad_r[:-1]))
        if rmin[kk] == 0:
            f = 0
        else:
            f = GEO_rmin_in[kk] / (ccc * d_theta / pi_2)

        # ---------------------------------------------
        # ! bt (toroidal field, Bt)
        # ! bp (poloidal field, Bp)
        # ! b  (total field, B)

        bt = f / bigr
        bp = (GEO_rmin_in[kk] / GEO_q_in[kk]) * grad_r / bigr
        # b(i)  = GEO_signb_in*np.sqrt(bt(i)**2+bp(i)**2)

        GEO_grad_r0.append(grad_r[int(n_theta // 2) + 1])

        # !
        # ! pre-factor of 0.5 comes from triangular element in phi-direction:
        # ! dV = (0.5*R*dphi)*(R*dZ)
        # !
        GEO_volume.append(0.5 * pi_2 * np.sum(bigz_t * bigr**2) * d_theta)

        if rmin[kk] == 0:
            GEO_bp0.append(0.0)
        else:
            GEO_bp0.append(bp[int(n_theta // 2) + 1])
        GEO_bt0.append(bt[int(n_theta // 2) + 1])
    GEO_bt0[0] = interp1e(rmin[1:], GEO_bt0[1:])(0)

    # --------------------------------------------------------------------------------------------------------
    #  END OF  Generalized Miller-type parameterization: Rememeber to transform it into a function so that
    #  it will be the staring point of the EXPRO mapping.
    # --------------------------------------------------------------------------------------------------------
    quantities = {}
    for item in ['GEO_volume', 'GEO_grad_r0', 'GEO_bt0', 'GEO_bp0']:
        quantities[re.sub('^GEO_', '', item)] = np.array(eval(item))
    return quantities


# OMAS extra_structures
_extra_structures = {
    'equilibrium': {
        'equilibrium.time_slice[:].profiles_1d.centroid': {
            "data_type": "structure",
            "documentation": "geometric information of the flux surfaces centroid",
            "full_path": "equilibrium/time_slice(itime)/profiles_1d/centroid",
        },
        'equilibrium.time_slice[:].profiles_1d.centroid.r_max': {
            "full_path": "equilibrium/time_slice(itime)/profiles_1d/centroid.r_max(:)",
            "coordinates": ['equilibrium.time_slice[:].profiles_1d.psi'],
            "data_type": "FLT_1D",
            "description": "centroid r max",
            "units": 'm',
        },
        'equilibrium.time_slice[:].profiles_1d.centroid.r_min': {
            "full_path": "equilibrium/time_slice(itime)/profiles_1d/centroid.r_min(:)",
            "coordinates": ['equilibrium.time_slice[:].profiles_1d.psi'],
            "data_type": "FLT_1D",
            "description": "centroid r min",
            "units": 'm',
        },
        'equilibrium.time_slice[:].profiles_1d.centroid.r': {
            "full_path": "equilibrium/time_slice(itime)/profiles_1d/centroid.r(:)",
            "coordinates": ['equilibrium.time_slice[:].profiles_1d.psi'],
            "data_type": "FLT_1D",
            "description": "centroid r",
            "units": 'm',
        },
        'equilibrium.time_slice[:].profiles_1d.centroid.z': {
            "full_path": "equilibrium/time_slice(itime)/profiles_1d/centroid.z(:)",
            "coordinates": ['equilibrium.time_slice[:].profiles_1d.psi'],
            "data_type": "FLT_1D",
            "description": "centroid z",
            "units": 'm',
        },
    }
}
omas.omas_utils._structures = {}
omas.omas_utils._structures_dict = {}
if not hasattr(omas.omas_utils, '_extra_structures'):
    omas.omas_utils._extra_structures = {}
for _ids in _extra_structures:
    for _item in _extra_structures[_ids]:
        _extra_structures[_ids][_item]['lifecycle_status'] = 'omfit'
    omas.omas_utils._extra_structures.setdefault(_ids, {}).update(_extra_structures[_ids])

############################################
if '__main__' == __name__:
    test_classes_main_header()

    from matplotlib import pyplot

    from omfit_classes.omfit_eqdsk import OMFITgeqdsk

    OMFIT['test'] = OMFITgeqdsk(OMFITsrc + '/../samples/g146102.00465')
    OMFIT['test']['fluxSurfaces'].load()
    OMFIT['test']['fluxSurfaces'].plot()
    pyplot.show()
