PyGammaAgreement
=============

| Branch  | Build status                                                                                                                     |
|---------|----------------------------------------------------------------------------------------------------------------------------------|
| master  | [![Build Status](https://travis-ci.com/Rachine/PyGammaAgreement.svg?token=RBFAQCRfvbxdpaEByTFc&branch=master)](https://travis-ci.com/Rachine/PyGammaAgreement/)  |


**PyGammaAgreement** is an open-source package to measure Inter/Intra-annotator agreement for sequences of annotations with the γ measure [2]. It is written in Python 3 and based mostly on NumPy, Numba and Pyannote.core. For a full list of available functions, please refer to the `API documentation <https://github.com/Rachine/PyGammaAgreement>`_.

Installation
============

Dependencies
------------

The main dependencies of PyGammaAgreement are :

* `NumPy <https://numpy.org/>`_ (>= 1.10)
* `SciPy <https://www.scipy.org/>`_ (>= 1.1.0)
* `sortedcontainers <http://www.grantjenks.com/docs/sortedcontainers/>`_ (>= 2.0)
* `pyannote.core <http://pyannote.github.io/pyannote-core/>`_ (>= 0.1.2)
* `pyannote.database <http://pyannote.github.io/pyannote-database/>`_ (>= 0.1.2)
* `Matplotlib <https://matplotlib.org/>`_ (>= 2.0)
* `CVXPY <https://www.cvxpy.org/>`_ (>= 1.0.25)
* `Numba <https://numba.pydata.org/>`_ (== 0.48.0)
* `TextGrid <https://github.com/kylebgorman/textgrid>`_ (== 1.5)
* `typing_extensions <https://github.com/python/typing>`_ (>= 3.7.4.3)


PyGammaAgreement is a Python 3 package and is currently tested for Python 3.7. PyGammaAgreement does not work with Python 2.7.

User installation
-----------------

PyGammaAgreement can be easily installed using pip

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
           *PyGammaAgreement: Gamma γ measure for inter-annotator agreement and alignment in Python.*
           

    .. [2] Mathet Y., Widlöcher A., Métivier, J.P.
           *The unified and holistic method gamma γ for inter-annotator agreement measure and alignment.*
           Computational Linguistics
           
