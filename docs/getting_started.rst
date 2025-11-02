Getting Started
===============

Welcome to ASTRA! ASTRA is a Python package for Automated model selection using Statistical Testing for Robust Algorithms.

Installation
------------

You can install ASTRA using pip:

.. code-block:: bash

    pip install .

Usage
-----

Once installed, you can use the ``astra`` command-line interface. For example, to run a benchmark:

.. code-block:: bash

    astra benchmark --config configs/example.yml

You can also use ASTRA in your Python code:

.. code-block:: python

    import astra

    astra.benchmark.run(data='path/to/your/data.csv')

For more details, please refer to the :doc:`user_guide`.