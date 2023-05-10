'''
stemdiff.gvars
--------------
Global variables/objects for package stemdiff.

The global variables are used throughout the whole program.
In order to be easily accessible, they are defined as two objects:
    
* SourceData = an object defining the input datafiles.
* DiffImages = an object desribing basic/common features of all NBD patterns.

The usage of stemdiff.gvars module is quite simple.
In 99% cases, we just add the following code at beginning of STEMDIFF script
(the arguments have to be adjusted to given experiment, of course):
    
>>> # Define source data
>>> # (datafiles from TimePix detector
>>> # (saved in data directory: D:/DATA.SH/STEMDIFF/SAMPLE_8
>>> # (datafiles are in subdirs: 01,02,... and named as: *.dat
>>> SDATA = stemdiff.gvars.SourceData(
>>>     detector  = stemdiff.detectors.TimePix(),
>>>     data_dir  = r'D:/DATA.SH/STEMDIFF/SAMPLE_8',
>>>     filenames = r'??/*.dat')
>>>
>>> # Set parameters of diffractin images
>>> # (we consider only central region with imgsize=100
>>> # (size of the region for PSF estimate will be: psfsize=30
>>> # (values of other/all args => documentation of stemdiff.gvars.DiffImages
>>> DIFFIMAGES = stemdiff.gvars.DiffImages(
>>>    imgsize=100, psfsize=30,
>>>    ctype=2, csquare=20, cintensity=0.8,
>>>    peak_height=100, peak_dist=9)
'''

from pathlib import Path

class SourceData:
    '''
    Define the input datafiles.
    
    Parameters
    ----------
    data_dir : str (which will be passed to Path method)
        This parameter is passed to Path method from pathlib library.
        It is strongly recommeded to use r-strings.
        Example: `data_dir = r'd:/data.sh/stemdiff/tio2'`.
    files : str (which will be passed to data_dir.glob method)
        This parameter is passed to Path.glob method from pathlib library.
        It is strongly recommeded to use r-strings.
        Example1: `datafiles = r'*.dat'` = all `*.dat` files in parent_dir;
        Example2: `datafiles = r'??/*.dat'` = all `*.dat` in subdirs `01,02...`
    
    Returns
    -------
    DataFiles object.
    '''

    def __init__(self, detector, data_dir, filenames):
        '''
        Initialize DataFiles object.
        The parameters are described above in class definition.
        '''
        self.detector = detector
        self.data_dir = Path(data_dir)
        self.filenames = self.data_dir.glob(filenames)
        
class DiffImages:
    '''
    Set parameters/characteristics of experimental diffraction images.
    
    Parameters
    ----------
    imgsize : integer, smaller than detector_size
        Size of array read from the detector is reduced to imgsize.
        If given, we sum only the central square with size = imgsize.
        Smaller area = higher speed;
        outer area = just weak diffractions.   
    psfize : integer, smaller than detector_size
        Size/edge of central square, from which 2D-PSF is determined.
    ctype : integer, values = 0,1,2
        0 = intensity center not determined, geometrical center is used;
        1 = center determined from the first image and kept constant;
        2 = center is determined for each individual image.
    csquare : integer, interval = 10--DET_SIZE
        Size of the central square (in pixels),
        within which the center of intensity is searched for.
    cintensity : float, interval = 0--1
        Intensity fraction, which is used for center determination.
        Example: cintensity=0.9 => consider only pixels > 0.9 * max.intensity.
    peak_height : float, optional, default is 100
        Search for peaks whose intensity > peak_height.
    peak_dist : integer, optional, default is 3
        Minimal distance between possible neighboring peaks.

    Returns
    -------
    DiffImages object.
    '''
    
    def __init__(self, 
                 imgsize=100, psfsize=30,
                 ctype=2, csquare=20, cintensity=0.8,
                 peak_height=100, peak_dist=3):
        '''
        Initialize parameters for center determination.
        The parameters are described above in class definition.
        '''
        self.imgsize = imgsize
        self.psfsize = psfsize
        self.ctype = ctype
        self.csquare = csquare
        self.cintensity = cintensity
        self.peak_height = peak_height
        self.peak_dist = peak_dist
