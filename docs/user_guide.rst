User Guide
==========

This page details how to use ASTRA. ASTRA provides a command-line interface (CLI) for benchmarking and comparing machine learning models.

Benchmarking Models
-------------------

Use `astra benchmark` to run performance benchmarks across different models using

.. code-block:: bash

    astra benchmark <data>

where ``data`` is the path to the dataset to train and evaluate models on. This should be a ``pd.DataFrame``, saved in CSV, pickle, or parquet format (which will be inferred from the file ending, and needs to be ``.csv``, ``.pkl``, ``.pickle``, or ``.parquet``), containing input features, the target variable, and a column indicating which fold a data point belongs to. See `here <../astra/data/example_df.csv>`_ for an example.

**Options:**

* ``--name``: Name of the experiment. Results will be saved in a folder with this name in the 'results' directory. Will be used to load cached results if they exist. Default: ``data`` file name (without the extension).
* ``--features``: Name of the column containing the features. Default: Features.
* ``--target``: Name of the column containing the target. Default: Target.
* ``--run_nested_CV``: Whether to run nested CV with hyperparameter tuning for the best models. Default: False.
* ``--use_optuna``: Whether to use Optuna for hyperparameter tuning. Default: False.
* ``--n_trials``: Number of trials for Optuna hyperparameter search. Default: 100.
* ``--timeout``: Time limit (in seconds) for Optuna hyperparameter search. Default: 3600.
* ``--fold_col``: Name(s) of the column(s) containing the *0-indexed* CV fold number(s). If a list is provided, models will be benchmarked in an nxk-fold CV, where n is the number of repeats and k is the number of folds. If a single string is provided, it will be treated as a single fold column. nxk-fold CV does not currently support nested CV and final hyperparameter tuning. Default: Fold.
* ``--main_metric``: Main metric to use for model selection. This will be used to infer the prediction task (classification or regression). Default: R2.
* ``--sec_metrics``: Secondary metrics to use for model selection. Default: MSE, MAE.
* ``--parametric``: Whether to use parametric statistical tests for model comparison. If 'auto' (default), the assumptions of parametric tests will be checked, and parametric tests will be used if the assumptions are met.
* ``--impute``: Method to use for imputing missing values. If None, no imputation will be performed. Valid choices are 'mean', 'median', 'knn', or a float or int value for constant imputation.
* ``--remove_constant``: If specified, features with variance below this threshold will be removed. If None, no features are removed.
* ``--remove_correlated``: If specified, features with correlation above this threshold will be removed. If None, no features are removed.
* ``--scaler``: Type of scaler to use, if the data is to be scaled first. Valid choices are 'Standard' and 'MinMax'. Default: None.
* ``--n_jobs``: Number of jobs to run in parallel for hyperparameter tuning. Default: 1.

.. note::

    Metrics are not case-sensitive. See `here <../astra/metrics.py>`_ for an overview of available metrics.

Alternatively, you can specify some or all arguments using a configuration file:

.. code-block:: bash

    astra benchmark --config <path to config file>

See `here <../configs/example.yml>`_ for an example. Arguments in the configuration file will override CLI arguments.

By default, ASTRA will benchmark all implemented `classification <../astra/models/classification.py>`_ or `regression <../astra/models/regression.py>`_ models, and search over default hyperparameter grids. You can specify which models to consider, custom starting hyperparameters, and custom hyperparameter search spaces in the configuration file. When using Optuna, custom hyperparameters will be converted into integer, float, or categorical Optuna distributions.

The benchmark script will create the following files under ``results/<name>``:

* ``default_CV.pkl``: A dictionary containing CV scores of the main and secondary metrics for all models using default hyperparameters.
* ``nested_CV.pkl``: A dictionary containing CV scores of the main and secondary metrics for all models with optimised hyperparameters using nested grid-search.
* ``final_CV.pkl``: A dictionary containing CV scores of the main and secondary metrics for the final model with optimised hyperparameters.
* ``final_CV_hparam_search.csv``: Final hyperparameter search results (``cv_results_`` of ``GridSearchCV`` or ``trials_dataframe`` of ``OptunaSearchCV``).
* ``final_model.pkl``: The best performing model, refit on the whole dataset.
* ``final_hyperparameters.pkl``: A dictionary of the optimal hyperparameters.

Comparing Models
----------------

Use `astra compare` to statistically analyse benchmark results:

.. code-block:: bash

    astra compare <CV_results_path>

where ``CV_results_path`` is a list of paths to directories containing CV results, or the path to a single directory containing all CV results. CV results should be pickled dictionaries with metrics as keys and lists of scores as values, for example, ``final_CV.pkl`` returned by `astra benchmark`, and ending with ``final_CV.pkl``. The model name will be the parent directory if passing a list of paths, or the file name (minus the ``final_CV.pkl``) if passing a single directory.

**Options:**

* ``--main_metric``: The main metric to use for comparison.
* ``--sec_metrics``: Secondary metrics to use for comparison.
* ``--parametric``: Whether to use parametric statistical tests for model comparison. If 'auto' (default), the assumptions of parametric tests will be checked, and parametric tests will be used if the assumptions are met.