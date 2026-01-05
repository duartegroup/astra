Developer Guide
===============

This page details how to contribute to ASTRA.

Setting up the Development Environment
--------------------------------------

It is recommended to use a virtual environment for development.

1. Install the project in editable mode with development dependencies:

   .. code-block:: bash

      pip install -e ".[dev]"

2. Set up `pre-commit` hooks to automatically check code style on commit:

   .. code-block:: bash

      pre-commit install

Running Tests
-------------

Tests are located in the ``tests/`` directory and can be run using `pytest`:

.. code-block:: bash

    pytest

Linting and Formatting
----------------------

This project uses `ruff <https://github.com/astral-sh/ruff>`_ for linting and formatting. You can run it with:

.. code-block:: bash

    ruff check .
    ruff format .

Building the Documentation
--------------------------

The documentation is built using Sphinx.

1. Install documentation dependencies:

   .. code-block:: bash

      pip install -r docs/requirements.txt

2. From the `docs/` directory, run:

   .. code-block:: bash

    make html

The built documentation will be in `docs/_build/html/`.