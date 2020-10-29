========
Tutorial
========


Quickstart
==========

In this small tutorial, you'll learn how to use the essential features
of pygamma-agreement.

Loading the data
~~~~~~~~~~~~~~~~

Let's say we have a short recording of some human speech, in which several
people are chatting. Let's call these people _Robin_, _Maureen_ and _Marvin_.
We asked 3 annotators to annotate this audio file to indicate when each of
these people is talking. Each annotated segment can be represented as a
tuple of 3 information:

    * Who is speaking ("Marvin")
    * Segment start (at 3.5s)
    * Segment end (at 7.2s)

Obviously, our annotators sometimes disagree on who might be talking,
or when exactly each person's speech turn is starting or ending. Luckily, the Gamma
inter-annotator agreement enables us to measure that.

We'll first load the annotation into `pygamma-agreement`'s base data structure,
the `Continuum`, made to store this kind of annotated data.

.. code-block:: python

    from pygamma_agreement import Continuum
    from pyannote.core import Segment

    continuum = Continuum()
    continuum.add("Annotator1", Segment(2.5, 4.3), "Maureen")
    continuum.add("Annotator1", Segment(4.6, 7.4), "Marvin")
    continuum.add("Annotator1", Segment(8.2, 11.4), "Marvin")
    continuum.add("Annotator1", Segment(13.5, 16.0), "Robin")

    continuum.add("Annotator2", Segment(2.3, 4.5), "Maureen")
    continuum.add("Annotator2", Segment(4.3, 7.2), "Marvin")
    continuum.add("Annotator2", Segment(7.9, 11.2), "Robin")
    continuum.add("Annotator2", Segment(13.0, 16.1), "Maureen")

    continuum.add("Annotator3", Segment(2.5, 4.3), "Maureen")
    continuum.add("Annotator3", Segment(4.6, 11.5), "Marvin")
    continuum.add("Annotator3", Segment(13.1, 17.1), "Robin")


Setting up a dissimilarity
~~~~~~~~~~~~~~~~~~~~~~~~~~

To measure our inter-annotator agreement (or disagreement), we'll need
a dissimilarity. Dissimilarities can be understood to be like distances,
although they don't quite satisfy some of their theoretical requirements.

In our case, we want that dissimilarity to measure both the disagreement on
**segment boundaries** (when is someone talking?) and **segment annotations** (who's talking?).
Thus, we'll be using the `CombinedCategoricalDissimilarity`, which uses both
**temporal** and **categorical** data to measure the disagreement between annotations.

Since we think that eventual categorical mismatches are more important
than temporal mismatches, we'll assign a greater weight to the former
using the `alpha` (for temporal mismatches) and `beta` (for categorical mismatches)
coefficients.

.. code-block:: python

    from pygamma_agreement import CombinedCategoricalDissimilarity

    dissim = CombinedCategoricalDissimilarity(list(continuum.categories),
                                              alpha=1, beta=2)


Computing the Gamma
~~~~~~~~~~~~~~~~~~~

We're all set to compute the gamma agreement now!

.. code-block:: python

    gamma_results = continuum.compute_gamma(dissim)
    print(f"The gamma for that annotation is f{gamma_results.gamma}")

Principles
==========

Continua (or Continuums)
~~~~~~~~~~~~~~~~~~~~~~~~

TODO

Alignments
~~~~~~~~~~

TODO

Dissimilarities
~~~~~~~~~~~~~~~

TODO


The Gamma agreement
~~~~~~~~~~~~~~~~~~~

TODO


