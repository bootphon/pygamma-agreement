---
title: 'pygamma-agreement: Gamma $\gamma$ measure for inter/intra-annotator agreement in Python'
tags:
  - Python
  - linguistics
  - annotation
  - statistics
authors:
  - name: Hadrien Titeux
    orcid: 0000-0002-8511-1644
    affiliation: 1
  - name: Rachid Riad
    orcid: 0000-0002-7753-1219
    affiliation: "1, 2"
affiliations:
 - name: LSCP/ENS/CNRS/EHESS/INRIA/PSL Research University, Paris, France 
   index: 1
 - name: NPI/ENS/INSERM/UPEC/PSL Research University, Créteil, France
   index: 2
   
date: 10 November 2020
bibliography: paper.bib
---

# Introduction

Over the last few decades, it has become easier to collect large audio recordings in naturalistic conditions and large corpora of text from the Internet. This broadens the scope of questions that can be addressed in speech and language research.

Scientists need to challenge their hypotheses and quantify the observed phenomenons of speech and language; this is why researchers add different layers of annotations on top of speech and text data. Some types of human intervention are used to reliably describe events contained in the corpus's content (e.g., Wikipedia articles, conversations, child babbling, animal vocalizations, or even just environmental sounds). These events can either be tagged at a particular point in time, or over a period of time. It is also commonplace to provide a categorical annotation or - in the case of speech - even precise transcriptions [@chat-childes-book] for these events. 
Depending on the difficulty of the annotation task and the eventual expertise of the annotators, the annotations they produce can include a certain degree of interpretation.
A common strategy when building annotated corpora is to have small parts of a corpus annotated by several annotators to be able quantify their consensus on that reduced subset of the corpus. 
If that consensus is deemed robust (i.e., agreement is high), we infer that the annotation task is well defined, less prone to interpretation, and that annotations that cover the rest of the corpus are reliable [@inter-rater-handbook].
An objective measure of the agreement (and subsequent disagreement) between annotators is thus desirable.

# Statement of Need

The Gamma ($\gamma$) Inter-Annotator Agreement Measure was proposed by @gamma-paper as a way to quantify inter-rater agreement for sequences of annotations. The $\gamma$-agreement measure solves a number of the shortcomings of other pre-existing measures. 
This quantification will have to satisfy some constraints: segmentation, unitizing, categorization, weighted categorization and the support for any number of annotators. They should also provide a chance-corrected value.
Other measures, such as the $\kappa$ [@kappa-paper] or Krippendorff's $\alpha$'s [@alpha-paper], have existed for some time to deal with these constraints, but cannot address all of them at once. A detailed comparison between metrics is available in @gamma-paper.

To solve all of these constraints at once, the $\gamma$-agreement works in three steps:  1) a _disorder_ (i.e., a cost) is computed for each potential alignments between the different annotators' units, using an annotation-dependent _dissimilarity_ (akin to a distance between units). This _disorder_ models the disagreement between annotators. 2) Using a convex optimization algorithm, a global alignment with the lowest total disorder is found. 3) By sampling random and synthetic annotations from the original annotation and computing their own disorders, the original annotation's disorder value is chance-corrected to provide the final $\gamma$-agreement measure.
The authors of @gamma-paper [provided a Java freeware](https://gamma.greyc.fr/) GUI implementation along with their paper and this implementation has already been used by some researchers [@gamma-usage] to compute an inter-rater agreement measure on their annotations.

However, linguist and automated speech researchers today use analysis pipelines that are either Python or shell scripts. 
To this day, no open-source implementation allows for the $\gamma$-agreement to be computed in a programmatical way, and researchers that are already proficient in Python and willing to automate their work might be hindered by the graphical nature of the original Java implementation.
Moreover, the original $\gamma$-agreement algorithm has several parameters that are determinant in its computation and cannot be configured as of now.
For this reason, it would greatly benefit the speech and linguistic scientific community if a fully open-source Python implementation of the original algorithm was available – this is what we are making availble here.
We have made sure that our implementation has several key features:

- It is comparatively as fast as the original implementation, taking about 10s to compute a high-confidence $\gamma$-agreement measure on a middle-range processor.
- The `pygamma-agreement` package is modular and users can easily extend one of the modules without the burden of rebuilding an optimized code base from scratch (e.g., Users can easily add a new dissimilarity measure).
- Our code allows for fine-grained constructions of an annotation Continuum and an advanced configurability of the $\gamma$-agreement's different parameters.
- We also made sure the interpretability of the input (i.e., the annotation continuum) and the output (i.e., the alignments) data structures are straightforward, as both can be visualized in a Jupyter Notebook (see \autoref{fig:continuum}).
- Finally, we support most of the commonly used data formats and analysis pipelines in the speech and linguistic fields.


# The pygamma-agreement Package

The `pygamma-agreement` package provides users with two ways to compute the $\gamma$-agreement for a corpus of annotations. The first is to use the package's Python API. 

```python
import pygamma_agreement as pa
continuum = pa.Continuum.from_csv("data/PaulAlexSuzann.csv")
dissimilarity = pa.CombinedCategoricalDissimilarity(categories=list(continuum.categories))
gamma_results = continuum.compute_gamma(dissimilarity, precision_level=0.02)
print(f"Gamma is {gamma_results.gamma}")
```

The most important primitives from our API (the `Continuum` \autoref{fig:continuum} and `Alignment` \autoref{fig:alignment} classes) can be displayed using the `matplotlib.pyplot` backend if the user is working in a Jupyter notebook. 

![Displaying a Continuum in a jupyter notebook. This is a temporally accurate representation of the annotated data. \label{fig:continuum}](continuum.png)

![Displaying an Alignment in a jupyter notebook. This is a visual and schematic representation of the alignment computed between annotations of the original continuum (the order of units is respected but the annotations are scaled for visual clarity). \label{fig:alignment}](best_alignment.png)

The second is a command-line application that can be invoked directly from the shell, for those who prefer to use shell scripts for corpus processing:

```bash
pygamma-agreement corpus/*.csv --confidence_level 0.02 --output_csv results.csv
```

We support a variety of commonly used annotation formats among speech researchers and linguists: RTTM, ELAN, TextGrid, CSV and `pyannote.core.Annotation` objects.

Computing the $\gamma$-agreement requires both array manipulation and the solving of multiple optimization problem formulated as Mixed-Integer Programming (MIP) problems. We thus used the _de facto_ standard for all of our basic array operations, NumPy [@numpy-paper].
Since some parts of the algorithm are fairly demanding, we made sure that these parts were heavily optimized using `numba` [@numba-paper]. We used `cvxpy`'s [@cvxpy-paper] MIP-solving framework to solve the optimization problem. For time-based annotations, we rely on primitives from `pyannote.core` [@pyannote-paper]. We made sure that it is robustly tested using the widely-adopted `pytest` testing framework. We also made sure that `pygamma-agreement`'s outputs matched both the theoretical values from the original paper and those of the Java freeware. Travis CI is used to run tests to ensure package quality is maintained and most of the code is type-hinted and has descriptive docstrings, both of which can be leveraged by IDEs to ease the use of the API.

We provide a user [documentation](https://pygamma-agreement.readthedocs.io/en/latest/) as well as an example Jupyter notebook in the package's repository. Additionally, we have used and tested `pygamma-agreement` in conjunction with the development of our own custom-built annotation platform, Seshat [@seshat]. In \autoref{tbl:statistics}, we present two use cases for our implementation of the $\gamma$-agreement measure on two corpora. These two corpora, ranging from medical interviews to child recording, allowed us to evaluate the performance of the $\gamma$-agreement on a wide panel of annotation types.


| Corpus              | Annotation                  | # Classes | Mean of $\gamma$ |
|---------------------|-----------------------------|-----------|------------------|
| Clinical Interviews | Turn-Takings                | 3         | 0.64             |
| Clinical Interviews | Utterances                  | 1         | 0.61             |
| Child Recordings    | Speech Activity             | 1         | 0.46             |
| Child Recordings    | Child/Adult-directed speech | 2         | 0.27             |


Table: $\gamma$ Inter-rater agreement for clinical interviews (16 samples) and child-centered day-long recordings (20 samples). \label{tbl:statistics}


The package is published in [PyPi](https://pypi.org/project/pygamma-agreement/), thus, `pygamma-agreement` can be installed using pip.


# Future Work

This implementation of the $\gamma$-agreement opens the path for a number of potential extensions:

* An obvious improvement is to add support for the "$\gamma$-cat" metric, a complement measure [@gamma-cat-paper] for the $\gamma$-agreement.
* The $\gamma$-agreement's theoretical framework allows for some useful improvements such as:
  - The implementation of new dissimilarities, such as a sequence-based dissimilarity (based on the Levenshtein distance), an ordinal dissimilarity (for ordered sets of categories) and a scalar dissimilarity.
  - For categorical annotations, the support for an undefined set of categories, with annotators using different sets of categories. This would be solved using an adapted implementation of the [Hungarian Algorithm](https://en.wikipedia.org/wiki/Hungarian_algorithm). This could be useful in unconstrained diarization annotation tasks.
  - More hypothetically, the possibility to have so-called "soft alignments", where a unit has weighted alignments with other units in its continuum.


# Acknowledgements

We are thankful to Yann Mathet for his help on understanding his work on the $\gamma$-agreement. We also thank Xuan-Nga Cao, Anne-Catherine Bachoux-Lévy and Emmanuel Dupoux for their advice, as well as Julien Karadayi for helpful discussions and feedback. This work is funded in part by the Agence Nationale pour la Recherche (ANR-17-EURE-0017Frontcog, ANR-10-IDEX-0001-02 PSL*, ANR-19-P3IA-0001PRAIRIE 3IA Institute) and Grants from Neuratris, from Facebook AI Research (Research Gift), Google (Faculty Research Award), Microsoft Research (Azure  Credits  and  Grant), and Amazon Web Service (AWS Research Credits).

# References

