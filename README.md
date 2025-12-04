STEMDIFF :: 4D-STEM dataset to 2D-diffractogram
-----------------------------------------------

* The **STEMDIFF package** converts... <br>
  ... a 4D-STEM dataset from a SEM microscope (huge and complex) <br>
  ... to a 2D-powder diffraction pattern (simple and easy to work with).
* If you use EDIFF in your research, **please cite** our recent paper:
	- *Microscopy and Microanalysis* 31, 2025, ozaf045. <br>
	  [https://doi.org/10.1093/mam/ozaf045](https://doi.org/10.1093/mam/ozaf045)

Principle
---------

<img src="https://mirekslouf.github.io/stemdiff/docs/assets/principle.pptx.png" alt="STEMDIFF principle" width="500"/>


Installation
------------
* Requirement: Python with sci-modules = numpy, matplotlib, scipy, pandas
* `pip install scikit-image` = 3rd party package for advanced image processing 
* `pip install tqdm` = to show progress meter during long summations
* `pip install idiff` = to improve diffractograms (remove noise, background ...)
* `pip install stemdiff` = STEMDIFF package itself (uses all packages above)


Quick start
-----------

* [Worked example](https://drive.google.com/file/d/1K3jWqiO1CpX-d4avcoYK0S8JfbAkZU78/view?usp=sharing)
  shows the STEMDIFF package in action.
* [Help on GitHub](https://mirekslouf.github.io/stemdiff/docs/)
  with complete
  [package documentation](https://mirekslouf.github.io/stemdiff/docs/pdoc.html/stemdiff.html)
  and
  [additional examples](https://drive.google.com/drive/folders/1X4TOdnMHVGSskzQJZ12edLt6MVl3HLvm?usp=sharing).

Documentation, help and examples
--------------------------------

* [PyPI](https://pypi.org/project/stemdiff) repository -
  the stable version to install.
* [GitHub](https://github.com/mirekslouf/stemdiff) repository - 
  the current version under development.
* [GitHub Pages](https://mirekslouf.github.io/stemdiff/) -
  the more user-friendly version of GitHub website.


Versions of STEMDIFF
--------------------

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
	* improved definition of summation and better documentation
* Version 5.0 = complete rewrite of v.4.2
	* conversion *2D-diffractogram &rarr; 1D-profile* moved to package EDIFF
	* better filtering (including estimated number of diffractions)
	* more detectors + more types of deconvolution (beta; to finish in v.6.0)
* Version 5.1 = (beta) support for parallel processing
* Version 5.2 = (beta) improvement of diff.patterns in sister package idiff


Acknowledgement
---------------

The development was co-funded by TACR, program NCK,
project [TN02000020](https://www.isibrno.cz/en/centre-advanced-electron-and-photonic-optics).