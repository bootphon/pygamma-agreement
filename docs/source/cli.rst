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

    annotator, annotation, segment_start, segment_end

Thus, for instance:

.. code-block::

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
If you wish to easily save the tool's output in a file, you can use the ``--output-csv`` (or ``-o``)
option to save it to a CSV that will contain each file's gamma value. You can then visualize the
results with tools like ``tabview`` or other spreadsheet software alike.

.. code-block:: bash

    pygamma-agreement data/*.csv --output-csv gamma_results.csv
    pygamma-agreement data/*.csv -o gamma_results.csv

There is also the option of saving the values in the JSON format with the ``--output-json`` (or ``-j``) option.

.. code-block:: bash

    pygamma-agreement data/*.csv --output-json gamma_results.json
    pygamma-agreement data/*.csv -j gamma_results.json


Optionally, you can also configure the alpha and beta coefficients used in
the combined categorical dissimilarity. You can also set a different precision level
if you want your gamma's estimation to be more or less precise:

.. code-block:: bash

    pygamma-agreement data/my_annotation_file.csv --alpha 3 --beta 1 --precision-level 0.02

In addition the gamma agreement, ``pygamma-agreement`` also gives you the option to output the gamma-cat & gamma-k(s)
alternate inter-annotator agreements, with the ``--gamma-cat`` (or ``-g``) and ``--gamma-k`` (or ``-k``) options, or
both.

.. code-block:: bash

    pygamma-agreement -k -g tests/data/*.csv -o test.csv
    OR
    pygamma-agreement -k -g tests/data/*.csv -j test.csv













