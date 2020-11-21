==========
Principles
==========

`pygamma-agreement` provides a set of classes that can be used to compute the γ-agreement measure on
a different set of annotation data and in different ways. What follows is a detailed
explaination of how these classes can be built and used together.

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
the term `Annotated Set` used in [mathet]_.
Annotations are stored as :ref:`units` , each annotator's units being stored in a sorted set.

`Continuums` are the "center piece" of our API, from which most of the work needed
to obtain gamma-agreement measures can be done. From a ``Continuum`` instance,
it's possible to compute and obtain:

- its :ref:`best alignment <best_alignments>` .
- its :ref:`gamma agreement measure <gamma_agreement>` .

If you're working in a Jupyter Notebook, outputting a `Continuum` instance will automatically
show you a graphical representation of that `Continuum`:

.. code-block:: ipython

  In [1]: continuum = Continuum.from_csv("data/PaulAlexSuzan.csv")
          continuum

.. image:: images/continuum_APS.png
   :alt: showing a continuum in a jupyter notebook
   :align: center


.. _alignments:

Alignments
~~~~~~~~~~

TODO


.. _best_alignments:

Best Alignments
~~~~~~~~~~~~~~~

TODO


.. _disorders:

Disorders
~~~~~~~~~

TODO

.. _dissimilarities:

Dissimilarities
~~~~~~~~~~~~~~~


Positional Dissimilarity
------------------------

TODO


Categorical Dissimilarity
-------------------------

TODO


Combined Dissimilarity
----------------------


TODO


.. _gamma_agreement:

The Gamma agreement
~~~~~~~~~~~~~~~~~~~

TODO


.. [mathet] Yann Mathet et Al.
    The Unified and Holistic Method Gamma (γ) for Inter-Annotator Agreement
    Measure and Alignment (Yann Mathet, Antoine Widlöcher, Jean-Philippe Métivier)
