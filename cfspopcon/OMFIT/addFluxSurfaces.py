from . import fluxSurfaces  
import numpy as np
import scipy
from scipy import interpolate, integrate
import omas
import shutil
import pickle
import os
import warnings

def is_int(value):
    """
    Convenience function check if value is integer

    :param value: value to check

    :return: True/False
    """
    import numpy as np

    return isinstance(value, (int, np.integer))

def sizeof_fmt(filename, separator='', format=None, unit=None):
    """
    function returns a string with nicely formatted filesize

    :param filename: string with path to the file or integer representing size in bytes

    :param separator: string between file size and units

    :param format: default None, format for the number

    :param unit: default None, unit of measure

    :return: string with file size
    """

    if unit in ['b', 'B']:
        unit = 'bytes'

    def _size(num):
        for u in ['bytes', 'kB', 'MB', 'GB']:
            if u == unit or (unit is None and num < 1024.0):
                break
            num /= 1024.0
        else:
            u = 'TB'
        if isinstance(format, str):
            f = format
        elif isinstance(format, dict) and u in list(format.keys()):
            f = format[u]
        else:
            f = '%3.1f'
        return f % num + separator + u

    if is_int(filename) and filename >= 0:
        return _size(filename)

    if isinstance(filename, str) and os.path.exists(filename):
        return _size(os.path.getsize(filename))

    else:
        return 'N/A'

def now(format_out='%d %b %Y  %H:%M', timezone=None):
    """

    :param format_out: https://docs.python.org/3/library/datetime.html#strftime-and-strptime-behavior

    :param timezone: [string] look at /usr/share/zoneinfo for available options
                     CST6CDT      Europe/?     Hongkong     MST          Portugal     WET
                     Africa/?     Canada/?     Factory      Iceland      MST7MDT      ROC
                     America/?    Chile/?      GB           Indian/?     Mexico/?     ROK
                     Antarctica/? Cuba         GB-Eire      Iran         NZ           Singapore
                     Arctic/?     EET          GMT          Israel       NZ-CHAT      Turkey
                     Asia/?       EST          GMT+0        Jamaica      Navajo       UCT
                     Atlantic/?   EST5EDT      GMT-0        Japan        PRC          US/?
                     Australia/?  Egypt        GMT0         Kwajalein    PST8PDT      UTC
                     Brazil/?     Eire         Greenwich    Libya        Pacific/?    Universal
                     CET          Etc/?        HST          MET          Poland       W-SU
                     Zulu

    :return: formatted datetime string
             if format_out is None, return datetime object
    """
    if timezone is not None:
        from dateutil import tz

        resolved_timezone = tz.gettz(timezone)
        if resolved_timezone is None:
            raise ValueError('Timezone `%s` not recognized! see /usr/share/zoneinfo/' % timezone)
        timezone = resolved_timezone
    dt = datetime.datetime.now(timezone)
    if format_out is None:
        return dt
    return dt.strftime(format_out)

class OMFITobject(object):
    def __init__(self, filename, **kw):

        readOnly = kw.pop('readOnly', False)
        if readOnly:
            kw['readOnly'] = True
        file_type = kw.pop('file_type', 'file')
        if file_type == 'dir':
            kw['file_type'] = 'dir'
        kw.pop('noCopyToCWD', None)  # this is always the case when running the classes outside the framework
        self.OMFITproperties = {}
        self.OMFITproperties.update(kw)

        # NOTE: some functionalities are available only when the OMFIT framework is running
        for item in ['serverPicker', 'remote', 'server', 'tunnel', 's3bucket']:
            if item in kw:
                raise Exception('`%s` functionality is only available within the OMFIT framework' % item)

        self.modifyOriginal = True  # importing classes outside of OMFIT --always-- operates on the original files
        self.readOnly = readOnly  # readonly functionality is supported outside of framework
        self.originalFilename = filename
        self.filename = filename
        self.link = filename
        self.dynaLoad = False

        if filename is None:
            return

        # remove trailing backslash from filename
        filename = filename.rstrip(os.sep)

        # handle comma separated filenames
        if file_type == 'dir':
            fnames = filename.split(',')
            if len(fnames) > 1:
                filename = os.path.split(filename.split(',')[0])[0]

        # create file if it does not exist
        if not os.path.exists(filename) or not len(filename):
            # if a directory was NOT specified...
            if not len(os.path.split(filename)[0]) or not len(filename):
                # an empty string generates a temporary file
                if not len(filename):
                    import tempfile

                    filename = tempfile._get_default_tempdir() + os.sep + file_type + '_' + now("%Y-%m-%d__%H_%M_%S__%f")
                # create file in the current working directory
                if file_type == 'dir':
                    os.makedirs(filename)
                else:
                    open(filename, 'w').close()
            else:
                raise OMFITexception("No such file or directory: '" + filename + "'")

        # set filename attributes
        filename = os.path.realpath(os.path.expandvars(os.path.expanduser(filename)))
        self.originalFilename = filename
        self.filename = filename
        self.link = filename

        # keep track of what classes have been loaded
        from omfit_classes.utils_base import _loaded_classes

        _loaded_classes.add(self.__class__.__name__)

    def save(self):
        """
        The save method is supposed to be overridden by classes which use OMFITobject as a superclass.
        If left as it is this method can detect if .filename was changed and if so, makes a copy from the original .filename (saved in the .link attribute) to the new .filename
        """
        return self._save_by_copy()

    def _save_by_copy(self):
        # if not exists or different
        if not (os.path.exists(self.filename) and os.path.samefile(self.link, self.filename)):

            if not os.path.exists(self.link):
                raise IOError('Missing link file ' + str(self.link))

            # remove existing file/directory if overwriting
            if not os.path.isdir(self.link):
                if os.path.exists(self.filename):
                    import filecmp

                    if filecmp.cmp(self.link, self.filename, shallow=False):
                        self.link = self.filename
                        return
                    else:
                        os.remove(self.filename)
                shutil.copy2(self.link, self.filename)
            else:
                tmp = os.getcwd()
                try:
                    # change working directory to handle possible overwriting
                    # of the current working directory or its parents
                    os.chdir('/')
                    if os.path.exists(self.filename):
                        shutil.rmtree(self.filename)
                    shutil.copytree(self.link, self.filename)
                finally:
                    os.chdir(tmp)

            self.link = self.filename

    def saveas(self, filename, remove_original_file=False, **kw):
        """
        This function changes the ``.filename`` attribute to filename and calls the ``.save()`` method and optionally remove original file.
        NOTE: use .deploy() to save an object to a given directory without changing the .filename that OMFIT uses

        :param filename: new absolute path of the filename.
                         If relative path is specified, then directory of current filename is used as root.
                         An empty filename will skip the save.

        :param remove_original_file: remove original file (forced to `False` if object is readOnly)

        :return: return value of save()
        """
        # empty filename skips the save
        if not filename:
            return

        # if not absolute path then use directory of current filename as root
        if filename.strip()[0] != os.sep:
            return self.saveas(
                os.path.split(os.path.abspath(self.filename))[0] + os.sep + filename, remove_original_file=remove_original_file, **kw
            )

        # save as
        old_filename = self.filename
        try:
            self.filename = filename
            tmp = self.save(**kw)
        except Exception:
            self.filename = old_filename
            raise

        # remove original file
        if remove_original_file and os.path.exists(old_filename) and not self.readOnly:
            if os.path.isdir(old_filename):
                shutil.rmtree(old_filename)
            else:
                os.remove(old_filename)
        return tmp

    def deploy(self, filename='', **kw):
        """
        The deploy method is used to save an OMFITobject to a location without affecting it's .filename nor .link attributes

        :param filename: filename or directory where to deploy file to
        """
        if filename == '' and self.filename:
            filename = os.path.split(self.filename)[1]

        tmpF = self.filename
        if hasattr(self, 'link'):
            tmpL = self.link

        try:
            if filename == '':
                self.filename = os.path.split(self.filename)[1]
            elif os.path.exists(filename) and os.path.isdir(filename):
                self.filename = filename + os.sep + os.path.split(tmpF)[1]
            else:
                self.filename = filename
            self.filename = os.path.abspath(self.filename)

            directory = os.path.split(self.filename)[0]
            if not os.path.exists(directory):
                os.makedirs(directory)

            self.save(**kw)

            return self.filename

        finally:
            self.filename = tmpF
            if hasattr(self, 'link'):
                self.link = tmpL

    def __deepcopy__(self, memo={}):
        """
        This method attempts to copy by pikling.
        If this fails, it will resort to calling the .duplicate() method

        :param memo: dictionary for preventing loops in deepcopy

        :return: copied object
        """
        try:
            return pickle.loads(pickle.dumps(self, pickle.HIGHEST_PROTOCOL))
        except Exception as _excp:
            printt('Failed to pickle: ' + repr(_excp) + '\nFallback on duplicating by file ' + self.filename)
            return self.duplicate(self.filename)

    def close(self):
        """
        This method:

            1. calls the .save() method, if it exists

            2. calls the .clear() method

            3. calls the .__init__() method with the arguments that were originally fed

        The purpose of this method is to unload from memory the typical OMFIT objects
        (which are a combination of SorteDict+OMFITobject classes), and has been added
        based on the considerations outlined in:
        http://deeplearning.net/software/theano/tutorial/python-memory-management.html#internal-memory-management
        http://www.evanjones.ca/memoryallocator/
        """
        if hasattr(self, 'dynaLoad') and self.dynaLoad:
            return
        if hasattr(self, 'save'):
            self.save()
        if hasattr(self, 'clear'):
            self.clear()
        self.__init__(self.filename, noCopyToCWD=True, **self.OMFITproperties)

    def __tree_repr__(self):
        if isinstance(self.filename, str):
            if os.path.isdir(self.filename):
                values = 'DIR: ' + os.path.split(self.filename)[1]
            else:
                values = 'FILE: ' + os.path.split(self.filename)[1] + '    (' + sizeof_fmt(self.filename) + ")"
        else:
            values = 'DIR/FILE: ?'
        return values, []

class OMFITpath(OMFITobject):
    r"""
    OMFIT class used to interface with files

    :param filename: filename passed to OMFITobject class

    :param \**kw: keyword dictionary passed to OMFITobject class
    """

    def __init__(self, filename, **kw):
        kw['file_type'] = 'file'
        OMFITobject.__init__(self, filename, **kw)

class OMFITascii(OMFITpath):
    r"""
    OMFIT class used to interface with ASCII files

    :param filename: filename passed to OMFITobject class

    :param fromString: string that is written to file

    :param \**kw: keyword dictionary passed to OMFITobject class
    """

    def __init__(self, filename, **kw):
        fromString = kw.pop('fromString', None)
        OMFITpath.__init__(self, filename, **kw)
        if fromString is not None:
            with open(self.filename, 'wb') as f:
                if isinstance(fromString, bytes):
                    f.write(fromString)
                else:
                    f.write(fromString.encode('utf-8'))

    def read(self):
        '''
        Read ASCII file and return content

        :return:  string with content of file
        '''
        with open(self.filename, 'r') as f:
            return f.read()

    def write(self, value):
        '''
        Write string value to ASCII file

        :param value: string to be written to file

        :return: string with content of file
        '''
        with open(self.filename, 'w') as f:
            f.write(value)
        return value

    def append(self, value):
        '''
        Append string value to ASCII file

        :param value: string to be written to file

        :return: string with content of file
        '''
        with open(self.filename, 'a') as f:
            f.write(value)
        return self.read()

def add_rhovn(self):
     """
     Calculate RHOVN from PSI and `q` profile
     """
     # add RHOVN if QPSI is non-zero (ie. vacuum gEQDSK)
     if np.sum(np.abs(self['QPSI'])):
         phi = integrate.cumtrapz(self['QPSI'], np.linspace(self['SIMAG'], self['SIBRY'], len(self['QPSI'])), initial=0)
         # only needed if the dimensions of phi are wanted
         # self['RHOVN'] = np.sqrt(np.abs(2 * np.pi * phi / (np.pi * self['BCENTR'])))
         self['RHOVN'] = np.sqrt(np.abs(phi))
         if np.nanmax(self['RHOVN']) > 0:
             self['RHOVN'] = self['RHOVN'] / np.nanmax(self['RHOVN'])
     else:
         # if no QPSI information, then set RHOVN to zeros
         self['RHOVN'] = self['QPSI'] * 0.0

class OMFITerror(object):
    def __init__(self, error='Error', traceback=None):
        self.error = error
        self.traceback = traceback

    def __repr__(self):
        return self.error
    
from .fluxSurfaces import SortedDict,dynaLoad
class OMFITgeqdsk(SortedDict, OMFITascii):
    r"""
    class used to interface G files generated by EFIT

    :param filename: filename passed to OMFITascii class

    :param \**kw: keyword dictionary passed to OMFITascii class
    """

    transform_signals = {
        'SIMAG': 'PSI',
        'SIBRY': 'PSI',
        'BCENTR': 'BT',
        'CURRENT': 'IP',
        'FPOL': 'BT',
        'FFPRIM': 'dPSI',
        'PPRIME': 'dPSI',
        'PSIRZ': 'PSI',
        'QPSI': 'Q',
    }

    def __init__(self, filename, **kw):
        OMFITascii.__init__(self, filename, **kw)
        SortedDict.__init__(self, caseInsensitive=True)
        self._cocos = 1
        self._AuxNamelistString = None
        self.dynaLoad = True

    def __getattr__(self, attr):
        try:
            return SortedDict.__getattr__(self, attr)
        except Exception:
            raise AttributeError('bad attribute `%s`' % attr)
        
    @property
    @dynaLoad
    def cocos(self):
        """
        Return COCOS of current gEQDSK as represented in memory
        """
        if self._cocos is None:
            return self.native_cocos()
        return self._cocos

    @cocos.setter
    def cocos(self, value):
        raise OMFITexception("gEQDSK COCOS should not be defined via .cocos property: use .cocosify() method")

    @dynaLoad
    def load(self, raw=False, add_aux=True):
        """
        Method used to read g-files
        :param raw: bool
            load gEQDSK exactly as it's on file, regardless of COCOS
        :param add_aux: bool
            Add AuxQuantities and fluxSurfaces when using `raw` mode. When not raw, these will be loaded regardless.
        """

        if self.filename is None or not os.stat(self.filename).st_size:
            return

        # todo should be rewritten using FortranRecordReader
        # based on w3.pppl.gov/ntcc/TORAY/G_EQDSK.pdf
        def splitter(inv, step=16):
            value = []
            for k in range(len(inv) // step):
                value.append(inv[step * k : step * (k + 1)])
            return value

        def merge(inv):
            if not len(inv):
                return ''
            if len(inv[0]) > 80:
                # SOLPS gEQDSK files add spaces between numbers
                # and positive numbers are preceeded by a +
                return (''.join(inv)).replace(' ', '')
            else:
                return ''.join(inv)

        self.clear()

        # clean lines from the carriage returns
        with open(self.filename, 'r') as f:
            EQDSK = f.read().splitlines()

        # first line is description and sizes
        self['CASE'] = np.array(splitter(EQDSK[0][0:48], 8))
        try:
            tmp = list([_f for _f in EQDSK[0][48:].split(' ') if _f])
            [IDUM, self['NW'], self['NH']] = list(map(int, tmp[:3]))
        except ValueError:  # Can happen if no space between numbers, such as 10231023
            IDUM = int(EQDSK[0][48:52])
            self['NW'] = int(EQDSK[0][52:56])
            self['NH'] = int(EQDSK[0][56:60])
            tmp = []
            printd('IDUM, NW, NH', IDUM, self['NW'], self['NH'], topic='OMFITgeqdsk.load')
        if len(tmp) > 3:
            self['EXTRA_HEADER'] = EQDSK[0][49 + len(re.findall('%d +%d +%d ' % (IDUM, self['NW'], self['NH']), EQDSK[0][49:])[0]) + 2 :]
        offset = 1

        # now, the next 20 numbers (5 per row)

        # fmt: off
        [self['RDIM'], self['ZDIM'], self['RCENTR'], self['RLEFT'], self['ZMID'],
         self['RMAXIS'], self['ZMAXIS'], self['SIMAG'], self['SIBRY'], self['BCENTR'],
         self['CURRENT'], self['SIMAG'], XDUM, self['RMAXIS'], XDUM,
         self['ZMAXIS'], XDUM, self['SIBRY'], XDUM, XDUM] = list(map(eval, splitter(merge(EQDSK[offset:offset + 4]))))
        # fmt: on
        offset = offset + 4

        # now I have to read NW elements
        nlNW = int(np.ceil(self['NW'] / 5.0))
        self['FPOL'] = np.array(list(map(float, splitter(merge(EQDSK[offset : offset + nlNW])))))
        offset = offset + nlNW
        self['PRES'] = np.array(list(map(float, splitter(merge(EQDSK[offset : offset + nlNW])))))
        offset = offset + nlNW
        self['FFPRIM'] = np.array(list(map(float, splitter(merge(EQDSK[offset : offset + nlNW])))))
        offset = offset + nlNW
        self['PPRIME'] = np.array(list(map(float, splitter(merge(EQDSK[offset : offset + nlNW])))))
        offset = offset + nlNW
        try:
            # official gEQDSK file format saves PSIRZ as a single flat array of size rowsXcols
            nlNWNH = int(np.ceil(self['NW'] * self['NH'] / 5.0))
            self['PSIRZ'] = np.reshape(
                np.fromiter(splitter(merge(EQDSK[offset : offset + nlNWNH])), dtype=np.float64)[: self['NH'] * self['NW']],
                (self['NH'], self['NW']),
            )
            offset = offset + nlNWNH
        except ValueError:
            # sometimes gEQDSK files save row by row of the PSIRZ grid (eg. FIESTA code)
            nlNWNH = self['NH'] * nlNW
            self['PSIRZ'] = np.reshape(
                np.fromiter(splitter(merge(EQDSK[offset : offset + nlNWNH])), dtype=np.float64)[: self['NH'] * self['NW']],
                (self['NH'], self['NW']),
            )
            offset = offset + nlNWNH
        self['QPSI'] = np.array(list(map(float, splitter(merge(EQDSK[offset : offset + nlNW])))))
        offset = offset + nlNW

        # now vacuum vessel and limiters
        if len(EQDSK) > (offset + 1):
            self['NBBBS'], self['LIMITR'] = list(map(int, [_f for _f in EQDSK[offset : offset + 1][0].split(' ') if _f][:2]))
            offset += 1

            nlNBBBS = int(np.ceil(self['NBBBS'] * 2 / 5.0))
            self['RBBBS'] = np.array(list(map(float, splitter(merge(EQDSK[offset : offset + nlNBBBS]))))[0::2])[: self['NBBBS']]
            self['ZBBBS'] = np.array(list(map(float, splitter(merge(EQDSK[offset : offset + nlNBBBS]))))[1::2])[: self['NBBBS']]
            offset = offset + max(nlNBBBS, 1)

            try:
                # this try/except is to handle some gEQDSK files written by older versions of ONETWO
                nlLIMITR = int(np.ceil(self['LIMITR'] * 2 / 5.0))
                self['RLIM'] = np.array(list(map(float, splitter(merge(EQDSK[offset : offset + nlLIMITR]))))[0::2])[: self['LIMITR']]
                self['ZLIM'] = np.array(list(map(float, splitter(merge(EQDSK[offset : offset + nlLIMITR]))))[1::2])[: self['LIMITR']]
                offset = offset + nlLIMITR
            except ValueError:
                # if it fails make the limiter as a rectangle around the plasma boundary that does not exceed the computational domain
                self['LIMITR'] = 5
                dd = self['RDIM'] / 10.0
                R = np.linspace(0, self['RDIM'], 2) + self['RLEFT']
                Z = np.linspace(0, self['ZDIM'], 2) - self['ZDIM'] / 2.0 + self['ZMID']
                self['RLIM'] = np.array(
                    [
                        max([R[0], np.min(self['RBBBS']) - dd]),
                        min([R[1], np.max(self['RBBBS']) + dd]),
                        min([R[1], np.max(self['RBBBS']) + dd]),
                        max([R[0], np.min(self['RBBBS']) - dd]),
                        max([R[0], np.min(self['RBBBS']) - dd]),
                    ]
                )
                self['ZLIM'] = np.array(
                    [
                        max([Z[0], np.min(self['ZBBBS']) - dd]),
                        max([Z[0], np.min(self['ZBBBS']) - dd]),
                        min([Z[1], np.max(self['ZBBBS']) + dd]),
                        min([Z[1], np.max(self['ZBBBS']) + dd]),
                        max([Z[0], np.min(self['ZBBBS']) - dd]),
                    ]
                )
        else:
            self['NBBBS'] = 0
            self['LIMITR'] = 0
            self['RBBBS'] = []
            self['ZBBBS'] = []
            self['RLIM'] = []
            self['ZLIM'] = []

        try:
            [self['KVTOR'], self['RVTOR'], self['NMASS']] = list(map(float, [_f for _f in EQDSK[offset : offset + 1][0].split(' ') if _f]))
            offset = offset + 1

            if self['KVTOR'] > 0:
                self['PRESSW'] = np.array(list(map(float, splitter(merge(EQDSK[offset : offset + nlNW])))))
                offset = offset + nlNW
                self['PWPRIM'] = np.array(list(map(float, splitter(merge(EQDSK[offset : offset + nlNW])))))
                offset = offset + nlNW

            if self['NMASS'] > 0:
                self['DMION'] = np.array(list(map(float, splitter(merge(EQDSK[offset : offset + nlNW])))))
                offset = offset + nlNW

            self['RHOVN'] = np.array(list(map(float, splitter(merge(EQDSK[offset : offset + nlNW])))))
            offset = offset + nlNW

            self['KEECUR'] = int(EQDSK[offset])
            offset = offset + 1

            if self['KEECUR'] > 0:
                self['EPOTEN'] = np.array(splitter(merge(EQDSK[offset : offset + nlNW])), dtype=float)
                offset = offset + nlNW

            # This will only work when IPLCOUT==2, which is not available in older versions of EFIT
            self['PCURRT'] = np.reshape(
                np.fromiter(splitter(merge(EQDSK[offset : offset + nlNWNH])), dtype=np.float64)[: self['NH'] * self['NW']],
                (self['NH'], self['NW']),
            )
            offset = offset + nlNWNH
            self['CJOR'] = np.array(splitter(merge(EQDSK[offset : offset + nlNW])), dtype=float)
            offset = offset + nlNW
            self['R1SURF'] = np.array(splitter(merge(EQDSK[offset : offset + nlNW])), dtype=float)
            offset = offset + nlNW
            self['R2SURF'] = np.array(splitter(merge(EQDSK[offset : offset + nlNW])), dtype=float)
            offset = offset + nlNW
            self['VOLP'] = np.array(splitter(merge(EQDSK[offset : offset + nlNW])), dtype=float)
            offset = offset + nlNW
            self['BPOLSS'] = np.array(splitter(merge(EQDSK[offset : offset + nlNW])), dtype=float)
            offset = offset + nlNW
        except Exception:
            pass

        # # add RHOVN if missing
        # if 'RHOVN' not in self or not len(self['RHOVN']) or not np.sum(self['RHOVN']):
        #     self.add_rhovn()

        # fix some gEQDSK files that do not fill PRES info (eg. EAST)
        if not np.sum(self['PRES']):
            pres = integrate.cumtrapz(self['PPRIME'], np.linspace(self['SIMAG'], self['SIBRY'], len(self['PPRIME'])), initial=0)
            self['PRES'] = pres - pres[-1]

        # parse auxiliary namelist
        #self.addAuxNamelist()

        if raw and add_aux:
            # add AuxQuantities and fluxSurfaces
        #    self.addAuxQuantities()
            self.addFluxSurfaces(**self.OMFITproperties)
        elif not raw:
            # Convert tree representation to COCOS 1
            self._cocos = self.native_cocos()
            self.cocosify(1, calcAuxQuantities=True, calcFluxSurfaces=True)

        #self.add_geqdsk_documentation()



    def addFluxSurfaces(self, **kw):
        r"""
        Adds ['fluxSurface'] to the current object

        :param \**kw: keyword dictionary passed to fluxSurfaces class

        :return: fluxSurfaces object based on the current gEQDSK file
        """
        if self['CURRENT'] == 0.0:
            printw('Skipped tracing of fluxSurfaces for vacuum equilibrium')
            return
        
        options = {}
        options.update(kw)
        options['quiet'] = kw.pop('quiet', self['NW'] <= 129)
        options['levels'] = kw.pop('levels', True)
        options['resolution'] = kw.pop('resolution', 0)
        options['calculateAvgGeo'] = kw.pop('calculateAvgGeo', True)

        # N.B., the middle option accounts for the new version of CHEASE
        #       where self['CASE'][1] = 'OM CHEAS'
        if (
            self['CASE'] is not None
            and self['CASE'][0] is not None
            and self['CASE'][1] is not None
            and ('CHEASE' in self['CASE'][0] or 'CHEAS' in self['CASE'][1] or 'TRXPL' in self['CASE'][0])
        ):
            options['forceFindSeparatrix'] = kw.pop('forceFindSeparatrix', False)
        else:
            options['forceFindSeparatrix'] = kw.pop('forceFindSeparatrix', True)

        try:
            self['fluxSurfaces'] = fluxSurfaces(gEQDSK=self, **options)
        except Exception as _excp:
            warnings.warn('Error tracing flux surfaces: ' + repr(_excp))
            self['fluxSurfaces'] = OMFITerror('Error tracing flux surfaces: ' + repr(_excp))

        return self['fluxSurfaces']