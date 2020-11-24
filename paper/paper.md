---
title: 'pygamma-agreement : a Python implementation of the Gamma inter-annotator agreement'
tags:
  - Python
  - linguistics
  - annotation
  - statistics
authors:
  - name: Rachid Riad
    # orcid: 0000-0003-0872-7098 # todo, maybe
    affiliation: "1, 2"
  - name: Hadrien Titeux
    orcid: 0000-0002-8511-1644
    affiliation: 1
affiliations:
 - name: LSCP/ENS/CNRS/EHESS/INRIA/PSL Research University, Paris, France 
   index: 1
 - name: NPI/ENS/INSERM/UPEC/PSL Research University, Créteil, France
   index: 2
   
date: 10 November 2020
bibliography: paper.bib
---

# Introduction

A great part of the current efforts in Linguistic Studies and automated speech processing algorithms goes into recording audio corpora that are ever bigger in size and ever more diverse in origins. However, an audio corpus is close to useless for many applications if it hasn't been painstakingly annotated by human annotators to produce a reference annotation, that reliably indicates events contained in the audio track (be it speech [@ami-corpus], baby noises [@child-babble-corpus], animal vocalizations [@birds-sounds-corpus], or even just plain noises [@musan]). Moreover, indicating _when_ something happens in the audio is half of the work: in many cases, it's also important for the human annotator to indicate _what_ is the nature of the event. Indeed, many annotations are either categorical, or - in the case of speech - precise transcriptions [@chat-childes-book] of the recorded speech. However, human annotators are suceptible to biases and errors, which raises the obvious question of the consistency and the reproducibility of their annotations. For these reasons, small parts of a corpus are usually annoted several times by different annotators, to assess the _agreement_ between annotators, and thus establish a numerical measure of the difficulty of annotating this corpus. 

Consequently, the Gamma ($\gamma$) Inter-Annotator Agreement Measure was proposed by [@gamma-paper]. This statistical measure combines both of the common agreement paradigms : unitizing (_where_ are the annotations) and categorization (_what_ are the annotations).

The authors of [@gamma-paper] [provided a Java freeware](https://gamma.greyc.fr/) (and thus closed-source) GUI implementation. However, a lot of the work in either automated speech processing or linguistics today is done using Python or shell scripts. For this reason, we thought it would greatly benefit both communities if we could provide them with a fully open-source Python implementation of the original algorithm.


# The pygamma-agreement Package


The `pygamma-agreement` package provides users with two ways to compute (in Python) the $\gamma$-agreement for a corpus. The first one is to use the simple Python API. 

```python
import pygamma_agreement as pa
continuum = pa.Continuum.from_csv("data/PaulAlexSuzann.csv")
dissimilarity = pa.Dissimilarity(categories=list(continuum.categories))
gamma_results = continuum.compute_gamma(dissimilarity, confidence_level=0.02)
print(f"Gamma is {gamma_results.gamma}")
```

The most important primitives from our API (the `Continuum` \autoref{fig:continuum} and `Alignment` \autoref{fig:alignment} classes) can be displayed using the `matplotlib.pyplot` backend if the user is working in a Jupyter notebook. 

![Displaying a Continuum in a jupyter notebook. \label{fig:continuum}](continuum.png)

![Displaying an Alignment in a jupyter notebook. \label{fig:alignment}](best_alignment.png)

The second one is a command-line application that can be invoked directly from the shell, for those who prefer to use shell scripts for corpus processing:

```bash
pygamma-agreement corpus/*.csv --confidence_level 0.02 --output_csv results.csv
```

We support an array of commonly used annotation formats: RTTM, TextGrid, CSV and `pyannote.core.Annotation` objects.

Computing the gamma-agreement requires both array manipulation and some convex optimization. We thus used Numpy for array operations. Since some parts of the algorithm are fairly demanding, we made sure that these parts were heavily optimized using `numba` [@numba-paper]. The convex optimization is done using `cvxpy` [@cvxpy-paper]'s MIP-solving framework. For time-based annotations, we rely on primitives from `pyannote.core` [@pyannote-paper]. We made sure that it is robustly tested using the widely-adopted `pytest` testing framework. We also back-tested it against the original Java implementation.

We provide a [documentation](https://pygamma-agreement.readthedocs.io/en/latest/) as well as an example Jupyter notebook in our package's repository. Additionally, we've used and tested `pygamma-agreement` in conjunction with the development of our own custom-built annotation platform, Seshat [@seshat].

We've uploaded our package to the [Pypi repository](https://pypi.org/project/pygamma-agreement/), thus, `pygamma-agreement` can be installed using pip.


# Future Work

We've identified a small number of improvements that our package could benefit from:

* A low hanging fruit is to add the support for the "$\gamma$-cat" metric, a complement measure [@gamma-cat-paper] for the $\gamma$-agreement.
* The $\gamma$-agreement's theoretical framework allows for the inclusion of a sequence-based dissimilarity, based on the Levenshtein distance.
* While our implementation is already close to the fastest pure python can be, we've identified some parts of it that could benefit from `numba`'s automatic parallelization features.


# Acknowledgements

We  are   thankful  to  Yann Mathet's help on understanding his work.  This work is funded in part by the Agence Nationale pour la Recherche (ANR-17-EURE-0017Frontcog, ANR-10-IDEX-0001-02 PSL*, ANR-19-P3IA-0001PRAIRIE 3IA Institute) and Grants from Neuratris, from Facebook AI Research (Research Gift), Google (Faculty Research Award),  Microsoft  Research  (Azure  Credits  and  Grant), and Amazon Web Service (AWS Research Credits).

# References

