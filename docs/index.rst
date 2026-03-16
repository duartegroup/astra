.. div:: hero

   .. raw:: html

      <h1 class="hero-title">✦ ASTRA ✦</h1>
      <p class="hero-tagline">Automated model selection using statistical testing for robust algorithms</p>

   .. code-block:: bash

      git clone https://github.com/duartegroup/astra.git && cd astra && pip install .

   |ci-badge| |license-badge| |python-badge|

----

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card::
      :link: getting_started
      :link-type: doc

      :octicon:`rocket;1.5em;sd-text-warning` **Getting Started**
      ^^^
      Install ASTRA and run your first benchmark in minutes.

   .. grid-item-card::
      :link: user_guide
      :link-type: doc

      :octicon:`book;1.5em;sd-text-warning` **User Guide**
      ^^^
      In-depth reference for all CLI commands and options.

   .. grid-item-card::
      :link: api
      :link-type: doc

      :octicon:`code-square;1.5em;sd-text-warning` **API Reference**
      ^^^
      Full documentation of the Python API.

   .. grid-item-card::
      :link: tutorials/index
      :link-type: doc

      :octicon:`mortar-board;1.5em;sd-text-warning` **Tutorials**
      ^^^
      Step-by-step worked examples.

   .. grid-item-card::
      :link: developer_guide
      :link-type: doc

      :octicon:`git-pull-request;1.5em;sd-text-warning` **Developer Guide**
      ^^^
      How to set up a dev environment and contribute.


.. toctree::
   :maxdepth: 2
   :hidden:
   :titlesonly:

   getting_started
   tutorials/index
   user_guide
   api
   developer_guide


.. |ci-badge| image:: https://img.shields.io/github/actions/workflow/status/duartegroup/astra/CI.yaml?branch=main&style=flat-square&label=CI&color=f59e0b
   :target: https://github.com/duartegroup/astra/actions/workflows/CI.yaml
   :alt: CI status

.. |license-badge| image:: https://img.shields.io/badge/License-MIT-f59e0b?style=flat-square
   :target: https://github.com/duartegroup/astra/blob/main/LICENSE
   :alt: MIT License

.. |python-badge| image:: https://img.shields.io/badge/Python-3.11%2B-3776ab?style=flat-square&logo=python&logoColor=white
   :target: https://www.python.org/
   :alt: Python 3.11+
