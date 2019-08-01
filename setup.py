#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2014-2019 CNRS

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# AUTHORS
# Rachid RIAD

# import versioneer

from setuptools import setup, find_packages

setup(

    # package
    namespace_packages=['pygamma'],
    packages=find_packages(),
    install_requires=[
        'sortedcontainers >= 2.0.4', 'numpy >= 1.10.4', 'scipy >= 1.1',
        'pandas >= 0.17.1', 'simplejson >= 3.8.1', 'matplotlib >= 2.0.0',
        'pyannote-core >=3.0', 'cvxpy >= 1.0.0', 'strsim == 0.0.3'
    ],
    extras_require={'test': ['pytest']},

    # # versioneer
    # version=versioneer.get_version(),
    # cmdclass=versioneer.get_cmdclass(),
    version='0.0.1',

    # PyPI
    name='pygamma',
    description=('Inter-annotator agreement measure and alginement'
                 'written in python'),
    author='Rachid RIAD',
    author_email='rachid.riad@ens.fr',
    # url='http://pyannote.github.io/',
    classifiers=[
        "Development Status :: 0.1 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English", "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Topic :: Scientific/Engineering"
    ],
)
