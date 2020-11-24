---
title: 'pygamma-agreement : a Python implementation of the Gamma inter-annotator agreement'
tags:
  - Python
  - linguistics
  - annotation
  - statistics
authors:
  - name: Rachid Riad
    # orcid: 0000-0002-7753-1219
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

In the last decades, it became easier to collect large audio recordings in naturalistic conditions and large corpora of text from the Internet. This broadens the scope of questions that can be addressed regarding speech and language.


Scientist need to challenge their hypotheses and quantify the observed phenomenons on speech and language; that is why scientists add different layers of annotations. Some type of human intervention is used to reliably describe events contained in the corpus's content (ex: Wikipedia articles, conversations, child babbling, animal vocalizations, or even just environmental sounds). These events can either be tagged at a particular point in time, or over a stretch of time. It is also commonplace to provide a categorical annotation or - in the case of speech -  even precise transcriptions [@chat-childes-book] for these events. 
Depending on the difficulty of the annotation task and the eventual expertise of the annotators, the annotations they produce can include a certain degree of interpretation.
A common strategy when building annotated corpora is to have small parts of a corpus annotated by several annotators, to be able quantify their consensus on that reduced subset of the corpus. 
If that consensus is deemed robust (i.e., agreement is high), we infer that the annotation task is well defined, less prone to interpretation, and that annotations that cover the rest of the corpus are reliable [@inter-rater-handbook].
An objective measure of the agreement (and subsequent disagreement) between annotators is thus desirable.

# Statement of Need

The Gamma ($\gamma$) Inter-Annotator Agreement Measure was proposed by [@gamma-paper] as a way to solve shortcomings of other pre-existing measures that aimed at quantifying inter-rater agreement. 
This quantification will have to satisfy some constraints : segmentation, unitizing, categorization, weighted categorization and the support for any number of annotators. They should also provide a chance-corrected value.
Measures, such as the $\kappa$ [@kappa-paper] or Krippendorff's $\alpha$'s [@alpha-paper],  have existed for some time to deal with these constraints, but never could address all of them at once. A detailed comparison between metrics is available in [@gamma-paper]. Furthermore, the authors of [@gamma-paper] [provided a Java freeware](https://gamma.greyc.fr/) GUI implementation along with their paper. 

Linguist and automated speech researchers today use analysis pipeline that are either Python or shell scripts. 
To this day, no open-source implementation allows for the $\gamma$-agreement to be computed in a programmatical way, and researchers that are already proficient in Python and willing to automate their work might be hindered by the graphical nature of the original Java implementation.
Moreover, the original $\gamma$-agreement algorithm has several parameters that are determinant in its computation and cannot be configured as of now.
For this reason, we thought it would greatly benefit the speech and linguistic scientific community if we could provide them with a fully open-source Python implementation of the original algorithm.


# The pygamma-agreement Package


The `pygamma-agreement` package provides users with two ways to compute the $\gamma$-agreement for a corpus of annotations. The first one is to use the package's Python API. 

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

We support a variety of commonly used annotation formats among speech researchers and linguists: RTTM, TextGrid, CSV and `pyannote.core.Annotation` objects.

Computing the gamma-agreement requires both array manipulation and the solving of multiple optimization problem formulated as Mixed-Integer Programming (MIP) problems. We thus used the _de facto_ standard for all of our basic array operations, NumPy [@numpy-paper]. Since some parts of the algorithm are fairly demanding, we made sure that these parts were heavily optimized using `numba` [@numba-paper]. We used `cvxpy`'s [@cvxpy-paper] MIP-solving framework to solve the optimization problem. For time-based annotations, we rely on primitives from `pyannote.core` [@pyannote-paper]. We made sure that it is robustly tested using the widely-adopted `pytest` testing framework. We also back-tested `pygamma-agreement`'s outputs against the original Java implementation's outputs to make sure they matched. We set-up an automated Travis CI to use these tests to ensure our package's quality. Most of our package's code is type-hinted and has descriptive docstrings, both of which  can be leveraged by IDEs to ease the use of our API.

We provide a user [documentation](https://pygamma-agreement.readthedocs.io/en/latest/) as well as an example Jupyter notebook in our package's repository. Additionally, we've used and tested `pygamma-agreement` in conjunction with the development of our own custom-built annotation platform, Seshat [@seshat]. In **Table 1**, we present two use cases for our implementation of the $\gamma$-agreement measure on two corpora. 


| Corpus              | Annotation                  | # Classes | Mean of $\gamma$ |
|---------------------|-----------------------------|-----------|------------------|
| Clinical Interviews | Turn-Takings                | 3         | 0.64             |
| Clinical Interviews | Utterances                  | 1         | 0.61             |
| Child Recordings    | Speech Activity             | 1         | 0.46             |
| Child Recordings    | Child/Adult-directed speech | 2         | 0.27             |


<p style="text-align: center;"><small>**Table 1**: $\gamma$ Inter-rater agreement for clinical interviews (16 samples) and child-centered day-long recordings (20 samples).</small></p>


We've uploaded our package to the [Pypi repository](https://pypi.org/project/pygamma-agreement/), thus, `pygamma-agreement` can be installed using pip.


# Future Work

We've identified a small number of improvements that our package could benefit from:

* An obvious improvement is to add support for the "$\gamma$-cat" metric, a complement measure [@gamma-cat-paper] for the $\gamma$-agreement.
* The $\gamma$-agreement's theoretical framework allows for the inclusion of a sequence-based dissimilarity, based on the Levenshtein distance. This would however require a numba re-implementation of the latter.
* While our implementation is already close to the fastest pure python can be, we've identified some parts of it that could benefit from `numba`'s automatic parallelization features.


# Acknowledgements

We are thankful to Yann Mathet for his help on understanding his work on the $\gamma$-agreement. We also thank Anne-Catherine Bachoux-Lévy and Emmanuel Dupoux for their advice, as well as Julien Karadayi for helpful discussions and feedbacks.  This work is funded in part by the Agence Nationale pour la Recherche (ANR-17-EURE-0017Frontcog, ANR-10-IDEX-0001-02 PSL*, ANR-19-P3IA-0001PRAIRIE 3IA Institute) and Grants from Neuratris, from Facebook AI Research (Research Gift), Google (Faculty Research Award),  Microsoft  Research  (Azure  Credits  and  Grant), and Amazon Web Service (AWS Research Credits).

# References

