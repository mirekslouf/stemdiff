# Python package initialization file.

'''
STEMDIFF package
----------------
Conversion of 4D-STEM dataset to single 2D-powder diffration pattern.

* Input = 4D-STEM dataset
    - The 4D-STEM dataset = 2D-array of 2D-NBD (nanobeam diffraction) patterns.
    - 2D-array = usually one (or more) two-dimensional scanning array(s).
    - 2D-NBD's = monocrystalline-like nanobeam diffraction patterns.
* Output = 2D-powder diffraction pattern
    - A 2D-powder diffraction pattern is calculated from a 4D-STEM dataset.
    - The calculation is basically a specific summation of 2D-NBD patterns.
    - In other words, a 4D-STEM dataset is reduced to a 2D-diffraction pattern.
    - The whole method (and final pattern) is called 4D-STEM/PNBD (powder NBD).
'''
__version__ = "5.0.4"
