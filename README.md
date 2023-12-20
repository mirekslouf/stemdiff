STEMDIFF :: Simple processing of 4D-STEM data
---------------------------------------------

* The **STEMDIFF package** converts... <br>
  ... a 4D-STEM dataset from a SEM microscope (huge and complex) <br>
  ... to a 2D-powder diffraction pattern (simple and easy to work with).
* The STEMDIFF package is a key part of our **4D-STEM/PNBD** method, <br>
  which was described (together with the package) in open-access publications:
	1. *Nanomaterials* 11 (2021) 962.
	   [https://doi.org/10.3390/nano11040962](https://doi.org/10.3390/nano11040962)
	2. *Materials* 14 (2021) 7550.
       [https://doi.org/10.3390/ma14247550](https://doi.org/10.3390/ma14247550)
* If you use STEMDIFF package, **please cite** the 2nd publication (or both :-).

Principle
---------

<img src="https://mirekslouf.github.io/stemdiff/docs/assets/principle.pptx.png" alt="STEMDIFF principle" width="500"/>

Installation
------------
* Requirement: Python with sci-modules: numpy, matplotlib, scipy, pandas
* `pip install scikit-image` = 3rd party package for advanced image processing 
* `pip install idiff` = to improve diffractograms (remove noise, background ...)
* `pip install stemdiff` = STEMDIFF package itself (uses all packages above)

Quick start
-----------

* See how it works:
	- Look at [worked example](https://www.dropbox.com/scl/fi/moic552kc35dl95wr6qji/01_sdiff_au.nb.html.pdf?rlkey=olaso0gwcousxezhvaaqm6w0q&dl=0)
      in Jupyter.
* Try it yourself:
	- Download [complete examples with data](https://www.dropbox.com/scl/fo/ccb6hs28er9dc1xufshh4/h?rlkey=omk5bqoe17jmedhj407ng9xr0&dl=0).
	- After downloading, unzip and follow the instructions in *readme* file.

Documentation, help and examples
--------------------------------

* [PyPI](https://pypi.org/project/stemdiff) repository.
* [GitHub](https://github.com/mirekslouf/stemdiff) repository.
* [GitHub Pages](https://mirekslouf.github.io/stemdiff)
  with [documentation](https://mirekslouf.github.io/stemdiff/docs).

## Versions of STEMDIFF

* Version 1.0 = Matlab: just a simple summation of 4D-dataset
* Version 2.0 = like v.1.0 + post-processing in Jupyter
* Version 3.0 = Python scripts: summation + S-filtering
* Version 4.0 = Python package: summation + S-filtering + deconvolution
	* summation = summation of all 2D-diffractograms
	* S-filtering = sum only diffractograms with strong diffractions = high S
	* deconvolution = reduce the primary beam spread effect
	  &rArr; better resolution 
* Version 4.2 = like v.4.0 + a few important improvements, such as:
	* sum just the central region with the strongest diffractions
	  &rArr; higher speed
	* 3 centering types: (0) geometry, (1) center of 1st, (2) individual centers 
	* better definition of summation and centering parameters
	* better documentation strings + demo data + improved *master script*
* Version 5.0 = complete rewrite of v.4.2
	* all key features of v.4.2 (summation, filtering, deconvolution)
	* conversion *2D-diffractogram &rarr; 1D-profile* moved to package EDIFF
	* several generalizations and improvements, namely:
		- possibility to define and use more detectors/datafile formats
		- better filtering (including estimated number of diffractions)
		- more types of deconvolution (experimental; to be finished in v.6.0)
* Version 5.1 = (beta) support for parallel processing
* Version 5.2 = (beta) improvement of diff.patterns in sister package idiff
