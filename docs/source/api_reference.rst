
API Reference
=============

Data structures
---------------

There are 2 types aliases that are important to understand :ref:`dissimilarities`:

* `Annot = Union[str, List[str]]` :
    this is either a simple string that represents a category
    (used by the categorical dissimilarity), or a list of strings that represent a sequence
    (used by the sequence dissimilarity).
* `Unit = Tuple[Segment, Annot]` :
    This corresponds to a concept introduced in the original Gamma Paper [1]_.
    In practice, it's a segment and its corresponding annotation. The dissimilarity between two
    `Unit` can be computed by being passed to :ref:`dissimilarities` instances.


.. [1] Yann Mathet et Al.
    The Unified and Holistic Method Gamma (γ) for Inter-Annotator Agreement
    Measure and Alignment (Yann Mathet, Antoine Widlöcher, Jean-Philippe Métivier)


.. autoclass:: pygamma.Continuum
    :members:
    :special-members:


.. _dissimilarities:

Dissimilarities
---------------

.. autoclass:: pygamma.PositionalDissimilarity
    :members:
    :special-members:

.. autoclass:: pygamma.CategoricalDissimilarity
    :members:
    :special-members:

.. autoclass:: pygamma.SequenceDissimilarity
    :members:
    :special-members:

.. autoclass:: pygamma.CombinedCategoricalDissimilarity
    :members:
    :special-members:

.. autoclass:: pygamma.CombinedSequenceDissimilarity
    :members:
    :special-members:


.. _alignments:

Alignments
----------

.. autoclass:: pygamma.UnitaryAlignment
    :members:
    :special-members:

.. autoclass:: pygamma.Alignment
    :members:
    :special-members: