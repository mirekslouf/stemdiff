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

Quick start
-----------

* See how it works:
	- Look at [worked example](https://mirekslouf.github.io/stemdiff/docs/examples/ex1_stemdiff.nb.html)
      in Jupyter.
* Try it yourself:
	- Download and unzip the [complete example with data](https://www.dropbox.com/scl/fo/321rnw7ywyiym0gsv68r1/h?dl=0&rlkey=ucr4kmaxqmgewsx20soo4rjsm).
	- Look at `00readme.txt` and run the example in Jupyter.

Documentation, help and examples
--------------------------------

* [PyPI](https://pypi.org/project/stemdiff) repository.
* [GitHub](https://github.com/mirekslouf/stemdiff) repository.
* [GitHub Pages](https://mirekslouf.github.io/stemdiff)
  with [documentation](https://mirekslouf.github.io/stemdiff/docs).

## Versions of STEMDIFF

* Version 1.0 = Matlab: just simple summation of 4D-dataset
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
	* 3 centering types: (0) geometry, (1) weight of 1st, (2) individual weights 
	* better definition of summation and centering parameters
	* better documentation strings + demo data + improved *master script*
* Version 5.0 = complete rewrite of v.4.2
	* all key features of v.4.2 (summation, filtering, deconvolution)
	* conversion *2D-diffractogram &rarr; 1D-profile* moved to package EDIFF
	* several generalizations and improvements, namely:
		- possibility to define and use more detectors/datafile formats
		- better filtering (including estimated number of diffractions)
		- more types of deconvolution algorithms
	