=================
Command Line Tool
=================


The ``pygamma-agreement`` CLI enables you to compute the gamma inter-annotator
agreement without having to use our library's Python API. For now, and to keep its usage
as simple as possible, it only supports the Combined Categorical Dissimilarity.

To use the command line tool, make sure you've installed the ``pygamma-agreement``
package via pip. You can check that the CLI tool is properly installed in your
environment by running

.. code-block:: bash

    pygamma-agreement -h

Supported data formats
-----------------------

Pygamma-agreement's CLI takes CSV or RTTM files as input.
CSV files need to have the following structure:

.. code-block::

    annotator_id, annotation, segment_start, segment_end

Thus, for instance:

.. code-block::

    E.G.:

    annotator_1, Marvin, 11.3, 15.6
    annotator_1, Maureen, 20, 25.7
    annotator_2, Marvin, 10, 26.3
    annotator_C, Marvin, 12.3, 14

Using the command line tool
---------------------------

``pygamma-agreement``'s command line can be used on one or more files, or a folder
containing several files. Thus, all these commands will work:

.. code-block:: bash

    pygamma-agreement data/my_annotation_file.csv
    pygamma-agreement data/annotation_1.csv data/annotation_2.csv
    pygamma-agreement data/*.csv
    pygamma-agreement data/

The gamma value for each file will be printed out in the console's stdout.
If you wish to easily save the tool's output in a file, you can use the ``--output_csv``
option to save it to a 2-column CSV that will contain each file's gamma value.

.. code-block:: bash

    pygamma-agreement data/*.csv --output_csv gamma_results.csv

Optionally, you can also configure the alpha and beta coefficients used in
the combined categorical dissimilarity. You can also set a different precision level
if you want your gamma's estimation to be more or less precise:

.. code-block:: bash

    pygamma-agreement data/my_annotation_file.csv --alpha 3 --beta 1 --precision_level 0.02

