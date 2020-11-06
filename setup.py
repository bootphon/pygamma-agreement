#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2020 CoML

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

from setuptools import setup, find_packages

with open("requirements.txt") as req_file:
    requirements = req_file.read().split("\n")

with open("README.md") as readme_file:
    long_description = readme_file.read()

setup(

    # package
    packages=find_packages(),
    install_requires=requirements,
    extras_require={'test': ['pytest']},
    version='0.1.3',

    # PyPI
    name='pygamma-agreement',
    description=('Inter-annotator agreement measure and alignment'
                 'written in python'),
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Rachid RIAD',
    author_email='rachid.riad@ens.fr',
    url='http://pygamme.readthedocs.io/',
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering"
    ],
    entry_points={
        'console_scripts': [
            'pygamma-agreement = pygamma_agreement.cli_apps:pygamma_cmd',
        ]
    }
)
