Pygamma-agreement
=============

[![Build Status](https://travis-ci.com/bootphon/pygamma-agreement.svg?branch=master?token=RBFAQCRfvbxdpaEByTFc&branch=master)](https://travis-ci.com/bootphon/pygamma-agreement/)
[![Documentation Status](https://readthedocs.org/projects/pygamma-agreement/badge/?version=latest)](https://pygamma-agreement.readthedocs.io/en/latest/?badge=latest)


**pygamma-agreement** is an open-source package to measure Inter/Intra-annotator 
agreement for sequences of annotations with the γ measure [2]. It is written in 
Python 3 and based mostly on NumPy, Numba and pyannote.core. For a full list of
 available functions, please refer to the [package documentation](https://pygamma-agreement.readthedocs.io/en/latest/).

Installation
============

Dependencies
------------

The main dependencies of pygamma-agreement are :

* `NumPy <https://numpy.org/>`_ (>= 1.10)
* `SciPy <https://www.scipy.org/>`_ (>= 1.1.0)
* `sortedcontainers <http://www.grantjenks.com/docs/sortedcontainers/>`_ (>= 2.0)
* `pyannote.core <http://pyannote.github.io/pyannote-core/>`_ (>= 0.1.2)
* `Matplotlib <https://matplotlib.org/>`_ (>= 2.0)
* `CVXPY <https://www.cvxpy.org/>`_ (>= 1.0.25)
* `Numba <https://numba.pydata.org/>`_ (== 0.48.0)
* `TextGrid <https://github.com/kylebgorman/textgrid>`_ (== 1.5)


pygamma-agreement is a Python 3 package and is currently tested for Python 3.7. 
PyGammaAgreement does not work with Python 2.7.

User installation
-----------------

pygamma-agreement can be easily installed using pip

```shell script
  pip install pygamma-agreement
```


Quick start
============


### Tests

The package comes with a unit-tests suit. To run it, first install *pytest* on your Python environment:

    pip install pytest
    pytest test/


#### References
    .. [1] Titeux H., Riad R.
           *pygamma-agreement: Gamma γ measure for inter-annotator agreement and alignment in Python.*
           

    .. [2] Mathet Y., Widlöcher A., Métivier, J.P.
           *The unified and holistic method gamma γ for inter-annotator agreement measure and alignment.*
           Computational Linguistics
           
