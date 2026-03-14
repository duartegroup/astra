User Guide
==========

ASTRA provides a command-line interface (CLI) for benchmarking and comparing
machine learning models using statistical testing.

Benchmarking Models
-------------------

Use ``astra benchmark`` to run performance benchmarks across different models:

.. code-block:: bash

    astra benchmark <data>

where ``data`` is the path to a dataset (CSV, pickle, or parquet) containing
input features, the target variable, and a fold column. See
`here <https://github.com/duartegroup/astra/blob/main/astra/data/example_df.csv>`_
for an example.

.. tip::

   You can specify some or all options via a YAML configuration file:

   .. code-block:: bash

      astra benchmark --config <config.yml>

   See `here <https://github.com/duartegroup/astra/blob/main/configs/example.yml>`_
   for the format. Arguments in the config file take precedence over CLI flags.

.. list-table::
   :header-rows: 1
   :widths: 22 14 14 50

   * - Option
     - Type
     - Default
     - Description
   * - ``--name``
     - str
     - data filename
     - Name of the experiment. Results are saved in ``results/<name>/``.
       Used to load cached results if they exist.
   * - ``--features``
     - str
     - Features
     - Name of the column containing the input features.
   * - ``--target``
     - str
     - Target
     - Name of the column containing the target variable.
   * - ``--run_nested_CV``
     - bool
     - False
     - Run nested CV with hyperparameter tuning for the best models.
   * - ``--use_optuna``
     - bool
     - False
     - Use Optuna for hyperparameter tuning instead of grid search.
   * - ``--n_trials``
     - int
     - 100
     - Number of trials for Optuna hyperparameter search.
   * - ``--timeout``
     - int
     - 3600
     - Time limit in seconds for Optuna hyperparameter search.
   * - ``--fold_col``
     - str | list
     - Fold
     - Name(s) of the *0-indexed* fold column(s). Providing a list enables
       nxk-fold CV (n repeats × k folds).
   * - ``--main_metric``
     - str
     - R2
     - Main metric for model selection. Determines whether the task is
       regression or classification.
   * - ``--sec_metrics``
     - list[str]
     - MSE, MAE
     - Secondary metrics to report alongside the main metric.
   * - ``--parametric``
     - str
     - auto
     - Whether to use parametric statistical tests. ``auto`` checks
       assumptions and selects the appropriate test automatically.
   * - ``--impute``
     - str | float | None
     - None
     - Missing value imputation strategy. Choices: ``mean``, ``median``,
       ``knn``, or a numeric constant.
   * - ``--remove_constant``
     - float | None
     - None
     - Remove features whose variance falls below this threshold.
   * - ``--remove_correlated``
     - float | None
     - None
     - Remove features whose correlation with another feature exceeds
       this threshold.
   * - ``--scaler``
     - str | None
     - None
     - Scale features before training. Choices: ``Standard``, ``MinMax``.
   * - ``--n_jobs``
     - int
     - 1
     - Number of parallel jobs for hyperparameter tuning.

.. note::

   Metrics are not case-sensitive. See
   `here <https://github.com/duartegroup/astra/blob/main/astra/metrics.py#L99>`_
   for all available metrics.

.. warning::

   nxk-fold CV (multiple ``--fold_col`` values) does not currently support
   nested CV or final hyperparameter tuning.

By default, ASTRA benchmarks all implemented
`classification <https://github.com/duartegroup/astra/blob/main/astra/models/classification.py>`_
or
`regression <https://github.com/duartegroup/astra/blob/main/astra/models/regression.py>`_
models over default hyperparameter grids. Custom models, starting
hyperparameters, and search spaces can be specified in the configuration file.

The benchmark creates the following files under ``results/<name>/``:

* ``default_CV.pkl`` — CV scores for all models with default hyperparameters.
* ``nested_CV.pkl`` — CV scores for all models with optimised hyperparameters
  (nested grid search).
* ``final_CV.pkl`` — CV scores for the final model with optimised
  hyperparameters.
* ``final_CV_hparam_search.csv`` — Full hyperparameter search results.
* ``final_model.pkl`` — Best model refit on the full dataset.
* ``final_hyperparameters.pkl`` — Optimal hyperparameters.
* ``benchmark.log`` — Log of the benchmarking process.

.. tip::

   All ``.pkl`` files can be loaded with Python's built-in :func:`pickle.load`
   or with :func:`pandas.read_pickle`.

Comparing Models
----------------

Use ``astra compare`` to statistically analyse benchmark results:

.. code-block:: bash

    astra compare <CV_results_path>

where ``CV_results_path`` is either a list of paths to directories each
containing ``final_CV.pkl``, or a single directory containing multiple
``*final_CV.pkl`` files. The model name is inferred from the parent directory
name (list mode) or from the filename prefix (single directory mode).

.. list-table::
   :header-rows: 1
   :widths: 22 14 14 50

   * - Option
     - Type
     - Default
     - Description
   * - ``--main_metric``
     - str
     - —
     - The main metric to use for comparison.
   * - ``--sec_metrics``
     - list[str]
     - —
     - Secondary metrics to report.
   * - ``--parametric``
     - str
     - ``auto``
     - Whether to use parametric statistical tests. ``auto`` checks
       assumptions and selects the appropriate test automatically.
