Pygamma-agreement
=============

[![Package Version](https://img.shields.io/pypi/v/pygamma-agreement)](https://pypi.org/project/pygamma-agreement/)
[![Supported Python Version](https://img.shields.io/pypi/pyversions/pygamma-agreement)](https://pypi.org/project/pygamma-agreement/)
[![Build Status](https://travis-ci.com/bootphon/pygamma-agreement.svg?branch=master?token=RBFAQCRfvbxdpaEByTFc&branch=master)](https://travis-ci.com/bootphon/pygamma-agreement/)
[![Documentation Status](https://readthedocs.org/projects/pygamma-agreement/badge/?version=latest)](https://pygamma-agreement.readthedocs.io/en/latest/?badge=latest)
[![status](https://joss.theoj.org/papers/d54271e471b25775e95ebcfc9bcf2493/status.svg)](https://joss.theoj.org/papers/d54271e471b25775e95ebcfc9bcf2493)

**pygamma-agreement** is an open-source package to measure Inter/Intra-annotator [1]
agreement for sequences of annotations with the γ measure [2]. It is written in 
Python 3 and based mostly on NumPy, Numba and pyannote.core. For a full list of
 available functions, please refer to the [package documentation](https://pygamma-agreement.readthedocs.io/en/latest/).

![Alignment Example](docs/source/images/best_alignment.png)


## Dependencies

The main dependencies of pygamma-agreement are :

* [NumPy](https://numpy.org/) (>= 1.10)
* [sortedcontainers](http://www.grantjenks.com/docs/sortedcontainers/) (>=2.0.4)
* [pyannote.core](http://pyannote.github.io/pyannote-core/) (>= 4.1)
* [CVXPY](https://www.cvxpy.org/) (>= 1.0.25)
* [CVXOPT](http://cvxopt.org/) (>= 1.2.6)
* [Numba](https://numba.pydata.org/) (>= 0.48.0)
* [TextGrid](https://github.com/kylebgorman/textgrid) (>= 1.5)
* [Pympi-ling](https://github.com/dopefishh/pympi) (>= 1.69)

Optionally, to allow `pygamma-agreement` to display visual representations of
our API's objects in Jupyter Notebooks, [Matplotlib](https://matplotlib.org/>) 
is needed.

pygamma-agreement is a Python 3 package and is currently tested for Python 3.7. 
pygamma-agreement does not work with Python 2.7.


## Installation

pygamma-agreement can be easily installed using pip

  $ pip install pygamma-agreement


Pygamma-agreement uses the [GNU Linear Programming Kit](https://www.gnu.org/software/glpk/) as its default Mixed Integer
Programming solver (critical step of the gamma-agreement algorithm). Since it is quite slow, you can install the 
[CBC](https://projects.coin-or.org/Cbc) solver and its official [python API](https://mpy.github.io/CyLPdoc/). 
To use those in `pygamma-agreement`, simply install them:

- Ubuntu/Debian :  ```$ sudo apt install coinor-libcbc-dev```
- Fedora : ```$ sudo yum install coin-or-Cbc-devel```
- Arch Linux : ```$ sudo pacman -S coin-or-cbc```
- Mac OS X :
    - ```$ brew tap coin-or-tools/coinor```
    - ```$ brew install coin-or-tools/coinor/cbc pkg-config```

then:

    $ pip install cylp


If you have trouble during the two last steps, pygamma-agreement should work anyway,
although significantly slower for larger input.


| ⚠️   |  Warning : A bug in GLPK causes the standart ouput to be polluted by non-deactivable messages. |
|-----|----------------------------------------------------------------------------------|

## Tests

The package comes with a unit-tests suit. To run it, first install *pytest* on your Python environment:

    $ pip install pytest
    $ pytest test/

## Submitting and issue or contributing

Please read `CONTRIBUTING.md` before submitting and issue or writing some contribution 
to this package.

## Citing Pygamma

If you're using pygamma in your work, please cite our package using the following bibtex entry:

```
@article{Titeux2021,
  doi = {10.21105/joss.02989},
  url = {https://doi.org/10.21105/joss.02989},
  year = {2021},
  publisher = {The Open Journal},
  volume = {6},
  number = {62},
  pages = {2989},
  author = {Hadrien Titeux and Rachid Riad},
  title = {pygamma-agreement: Gamma $\gamma$ measure for inter/intra-annotator agreement in Python},
  journal = {Journal of Open Source Software}
}

```

## References

[1]: [Titeux H., Riad R.
     pygamma-agreement: Gamma γ measure for 
     inter/intra-annotator agreement in Python.](https://doi.org/10.21105/joss.02989)
           

[2]: [Mathet Y., Widlöcher A., Métivier, J.P.
     The unified and holistic method gamma γ for
     inter-annotator agreement measure and alignment. 
     Computational Linguistics](https://www.aclweb.org/anthology/J15-3003.pdf)
           
