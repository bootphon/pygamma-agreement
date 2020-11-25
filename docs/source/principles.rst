==========
Principles
==========

`pygamma-agreement` provides a set of classes that can be used to compute the γ-agreement measure on
a different set of annotation data and in different ways. What follows is a detailed
explanation of how these classes can be built and used together.

.. warning::

  A great part of this page is just a rehash of concepts that are explained
  in a much clearer and more detailed way in the original γ-agreement paper  [mathet2015]_.
  This documentation is mainly aimed to giving the reader a better understanding of our API's core
  principles.
  If you want a deeper understanding of the gamma agreement, we strongly advise that
  you take a look at this paper.

.. _units:

Units
~~~~~

A **unit** is an object that is defined in time (meaning, it has a starting point and an ending point),
and might bear an annotation. In our case, the only supported type of annotation is a category.
In `pygamma-agreement`, for all basic usages, ``Unit`` objects are built automatically when
you add an annotation to :ref:`a continuum <continua>` .
You might however have to create ``Unit`` objects if you want to create your own :ref:`alignments` .

.. note::

  `Unit` instances are sortable, meaning that they satisfy a total ordering.
  The ordering uses their temporal positions (starting/ending times) as a first
  parameter to order them, and if these are strictly equal, uses the alphabetical
  order to compare them. Ergo,

  .. code-block:: python

    >>> Unit(Segment(0,1), "C") < Unit(Segment(2,3), "A")
    True
    >>> Unit(Segment(0,1), "C") > Unit(Segment(0,1), "A")
    True
    >>> Unit(Segment(0,1)) < Unit(Segment(0,1), "A")
    True

.. _continua:

Continua (or Continuums)
~~~~~~~~~~~~~~~~~~~~~~~~

A **continuum** is an object that that stores the set of annotations produced by
several annotators, all referring to the same annotated file. It is equivalent to
the term `Annotated Set` used in Mathet et Al.  [mathet2015]_.
Annotations are stored as :ref:`units` , each annotator's units being stored in a sorted set.

`Continuums` are the "center piece" of our API, from which most of the work needed
to obtain gamma-agreement measures can be done. From a ``Continuum`` instance,
it's possible to compute and obtain:

- its :ref:`best alignment <best_alignments>` .
- its :ref:`gamma agreement measure <gamma_agreement>` .

If you're working in a Jupyter Notebook, outputting a `Continuum` instance will automatically
show you a graphical representation of that `Continuum`:

.. code-block:: ipython

 In  [mathet2015]: continuum = Continuum.from_csv("data/PaulAlexSuzan.csv")
     continuum

.. image:: images/continuum_APS.png
  :alt: showing a continuum in a jupyter notebook
  :align: center


.. _alignments:

Alignments, Unitary Alignments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**A unitary alignment** is a tuple of `units`, each belonging to a unique annotator.
For a given `continuum` containing annotations from ``n`` different annotators,
the tuple will *have* to be of length ``n``. It represents a "match" (or the hypothesis of an agreement)
between units of different annotators. A unitary alignment can contain `empty units`,
which are "fake" (and null) annotations inserted to represent the absence of corresponding annotation
for one or more annotators.


**An alignment** is a set of `unitary alignments` that constitute a *partition*
of a continuum. This means that

* each and every unit of each annotator from the partitioned continuum can be found in the alignment
* each unit can be found once and *only* once

.. figure:: images/alignments.png
  :align: center
  :alt: Representation of unitary alignments

  Visual representations of unitary alignments taken from  [mathet2015]_ , with
  varying disorders.

For both alignments and unitary alignments, it is possible to compute a corresponding
:ref:`disorder <disorders>`. This possible via the ``compute_disorder`` method,
which takes a :ref:`dissimilarity <dissimilarities>` as an argument.
This (roughly) corresponds to the disagreement between
annotators for a given alignment or unitary alignment.

In practice, you shouldn't have to build both unitary alignments and alignments
yourself, as they are automatically constructed by a `continuum` instance's
``get_best_alignment`` method.


.. _best_alignments:

Best Alignments
---------------

For a given `continuum`, the alignment with the lowest possible disorder.
This alignment is found using a combination of 3 steps:

* the computation of the disorder for all potential unitary alignments
* a simple heuristic explained in  [mathet2015]_'s section 5.1.1 to eliminate
  a big part of those potential unitary alignments based on their disorder
* the usage of multiple optimization problem formulated as Mixed-Integer Programming (MIP)

The function that implements these 3 steps is, by far, the most compute-intensive in the
whole ``pygamma-agreement`` library. The best alignment's disorder is used to compute
the :ref:`gamma_agreement` .

.. _disorders:

Disorders
~~~~~~~~~

The disorder for either an :ref:`alignment or a unitary alignment <alignments>` corresponds to the
disagreement between its constituting units.

* the disorder between two units is directly computed using a dissimilarity.
* the disorder between units of a unitary alignment is an average of the disorder of each of its unit couples
* the disorder of an alignment is the mean of its constituting unitary alignments's disorders

What should be remembered is that a :ref:`dissimilarity <dissimilarities>` is needed
to compute the disorder of either an alignment or a unitary alignment, and that
its "raw" value alone isn't of much use but it is needed to compute the γ-agreement.

.. _dissimilarities:

Dissimilarities
~~~~~~~~~~~~~~~

A dissimilarity is a function that "tells to what degree two units should be considered as different,
taking into account such features as their positions, their annotations, or a combination of the two."  [mathet2015]_.
A dissimilarity has the following mathematical properties:

* it is positive (:math:`dissim (u,v) > 0`))
* it is symmetric ( :math:`dissim(u,v) = dissim(v,u)` )
* :math:`dissim(x,x) = 0`

Although dissimilarities do look a lot like distances (in the mathematical sense),
they don't necessarily are distances. Hence, they don't necessarily honor the
triangular inequality property.

If one of the units in the dissimilarity is the *empty unit*, its value is
:math:`\Delta_{\emptyset}`. This value is constant, and can be set as a parameter
of a dissimilarity object before it is used to compute an alignment's disorder.

TODO : show usage of dissimarity instances with continuua, alignments

.. note::

    Although you will have to instantiate dissimilarities objects when using
    `pygamma-agreement`, you'll never have to use them in any way other than
    just by passing it as an argument as shown beforehand.

Positional Dissimilarity
------------------------
The positional dissimilarity is used to measure the *positional* or *temporal* disagreement
between two annotated units :math:`u` and :math:`v`. Its formula is :

.. math::
    d_{pos}(u,v) = \left(
                    \frac{\lvert start(u) - start(v) \rvert + \lvert end(u) - end(v) \rvert}
                    {(duration(u) + duration(v))}
                \right)^2 \cdot \Delta_{\emptyset}


TODO : show instantiation and parameters

Categorical Dissimilarity
-------------------------

The categorical dissimilarity is used to measure the *categorical* disagreement
between two annotated units :math:`u` and :math:`v`.


.. math::

    d_{cat}(u,v) = dist_{cat}(cat(u), cat(v)) \cdot \Delta_{\emptyset}


In our case, the function :math:`dist_{cat}`
is computed using a simple lookup in a *categorical distance matrix* ``D``.
Let's suppose we have :math:`K` categories, this matrix will be of shape ``(K,K)``.

Here is an example of a distance matrix for 3 categories:

.. code-block:: python

    >>> D
    array([[0. , 0.5, 1. ],
           [0.5, 0. , 1. ],
           [1. , 1. , 0. ]])


To comply with the properties of a dissimilarity, the matrix has to be symmetrical,
and has to have an empty diagonal. Moreover, its values have be between 0 and 1.
By default, for two units with differing categories, :math:`d_{cat}(u,v) = 1`,
and thus the corresponding matrix is:

.. code-block:: python

    >>> D_default
    array([[0., 1., 1. ],
           [1., 0., 1. ],
           [1., 1., 0. ]])

.. warning::

    This dissimilarity, as of now, cannot directly be used to compute the γ-agreement.
    In some near future, it will be usable to compute the γ-categorical agreement [mathet2018]_.
    However, since this dissimilarity is part of the following :ref:`combined-dissim`,
    we thought it was useful to explain its functioning.

.. _combined-dissim:

Combined Dissimilarity
----------------------

The combined dissimilarity uses a linear combination of the two previous
*categorical* and *positional* dissimilarities. The two coefficients used to
weight the importance of each dissimilarity are :math:`\alpha` and :math:`\beta` :

.. math::

    d_{combi}^{\alpha,\beta}(u,v) = \alpha . d_{pos} + \beta . d_{cat}

TODO : show instantiation and parameters

.. _gamma_agreement:

The Gamma (γ) agreement
~~~~~~~~~~~~~~~~~~~~~~~

The γ-agreement is a *chance-adjusted* measure of the agreement between annotators.
To be computed, it requires

* a :ref:`continuum <continua>` , containing the annotators's annotated units.
* a :ref:`dissimilarity <dissimilarities>`, to evaluate the disorder between the hypothesized
  alignments of the annotated units.

Using these two components, we can compute the :ref:`best alignment <best_alignments>` and its
disorder. Let's call this disorder :math:`\delta_{best}`.

Without getting into its details, our package implements a method of sampling random
annotations from a continuum. Using these :math:`N` sampled continuum, we can also compute a best
alignment and its subsequent disorder. Let's call these disorders :math:`\delta_{random}^i`, and the
mean of these values :math:`\delta_{random} =  \frac{\sum_i \delta_{random}^i}{N}`

The gamma agreement's formula is finally:

.. math::

    \gamma = 1 - \frac{\delta_{best}}{\delta_{random}}

Several points that should be made about that value:

* it is not bounded, but for most "regular" situations it should be contained in :math:`[0, 1]`
* the higher and the closer it is to 1, the better.


TODO : supplement with an explanation of the ``compute_gamma`` API.


..  [mathet2015] Yann Mathet et Al.
    The Unified and Holistic Method Gamma (γ) for Inter-Annotator Agreement
    Measure and Alignment (Yann Mathet, Antoine Widlöcher, Jean-Philippe Métivier)

..  [mathet2018] Yann Mathet
    The Agreement Measure Gamma-Cat : a Complement to Gamma Focused on Categorization of a Continuum
    (Yann Mathet 2018)
