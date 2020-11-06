Pygamma-agreement
=============

[![Build Status](https://travis-ci.com/bootphon/pygamma-agreement.svg?branch=master?token=RBFAQCRfvbxdpaEByTFc&branch=master)](https://travis-ci.com/bootphon/pygamma-agreement/)
[![Documentation Status](https://readthedocs.org/projects/pygamma-agreement/badge/?version=latest)](https://pygamma-agreement.readthedocs.io/en/latest/?badge=latest)


**pygamma-agreement** is an open-source package to measure Inter/Intra-annotator 
agreement for sequences of annotations with the γ measure [^2]. It is written in 
Python 3 and based mostly on NumPy, Numba and pyannote.core. For a full list of
 available functions, please refer to the [package documentation](https://pygamma-agreement.readthedocs.io/en/latest/).

![Alignment Example](docs/source/images/best_alignment.png)


## Dependencies

The main dependencies of pygamma-agreement are :

* [NumPy](https://numpy.org/>) (>= 1.10)
* [sortedcontainers](http://www.grantjenks.com/docs/sortedcontainers/>) (>=2.0.4)
* [pyannote.core](http://pyannote.github.io/pyannote-core/>) (>= 4.1)
* [Matplotlib](https://matplotlib.org/>) (>= 2.0)
* [CVXPY](https://www.cvxpy.org/>) (== 1.0.25)
* [Numba](https://numba.pydata.org/) (>= 0.48.0)
* [TextGrid](https://github.com/kylebgorman/textgrid) (>= 1.5)


pygamma-agreement is a Python 3 package and is currently tested for Python 3.7. 
pygamma-agreement does not work with Python 2.7.

## Installation

pygamma-agreement can be easily installed using pip

```shell script
  pip install pygamma-agreement
```


## Tests

The package comes with a unit-tests suit. To run it, first install *pytest* on your Python environment:

    pip install pytest
    pytest test/


## References

[1] Titeux H., Riad R.
   *pygamma-agreement: Gamma γ measure for inter/intra-annotator agreement in Python.*
           

[2] Mathet Y., Widlöcher A., Métivier, J.P.
   *The unified and holistic method gamma γ for inter-annotator agreement measure and alignment.*
   Computational Linguistics
           
