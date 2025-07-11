Automated model selection using Statistical Testing for Robust Algorithms (ASTRA)
==============================
[//]: # (Badges)
[![GitHub Actions Build Status](https://github.com/duartegroup/astra/workflows/CI/badge.svg)](https://github.com/duartegroup/astra/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/duartegroup/ASTRA/branch/main/graph/badge.svg)](https://codecov.io/gh/duartegroup/ASTRA/branch/main)

## Installation

To install the package from source:

```bash
git clone 
cd astra
pip install -e .
```

## Command Line Interface

ASTRA provides two main commands: `benchmark` and `compare`.

### Benchmarking Models

Use `astra benchmark` to run performance benchmarks across different models using

```bash
astra benchmark <data>
```
where `data` is the path to the dataset to train and evaluate models on. This should be a `pd.DataFrame`, saved in CSV, pickle, or parquet format (which will be inferred from the file ending, and needs to be `.csv`, `.pkl`, `.pickle`, or `.parquet`). If the data is not prefeaturised and presplit, it should contain a column called 'SMILES'. See [here](astra/data/example_df.csv) for an example.

Options:
- `--name`: Name of the experiment. Results will be saved in a folder with this name in the 'results' directory. Will be used to load cached results if they exist. Default: `data` file name (without the extension).
- `--features`: Name of the column containing the features. Default: Features.
- `--target`: Name of the column containing the target. Default: Target.
- `--run_nested_CV`: Whether or not to run nested CV with hyperparameter tuning for the best models. Default: False.
- `--fold_col`: Name of the column containing the *0-indexed* CV fold number. Default: Fold.
- `--main_metric`: Main metric to use for model selection. This will be used to infer the prediction task (classification or regression). Default: R2.
- `--sec_metrics`: Secondary metrics to use for model selection. Default: MSE, MAE.
- `--split`: Type of split to use, if the data is to be resplit first. Valid choices are 'Scaffold' and 'Fingerprint'. Results (fold number) will be saved in a column called 'Fold'. Default: None (i.e., don't resplit).
- `--n_folds`: Number of folds to split the data into, if the data is to be resplit first. Default: 5.
- `--fingerprint`: Type of fingerprint to use, if the data is to be featurised first. Valid choices are 'Morgan', 'Avalon', 'RDKit', 'MACCS', 'AtomPair', 'TopTorsion'. Results will be saved in a column called 'Features'. For Morgan fingerprints, specify the radius and fingerprint size as 'Morgan_{radius}_{fpsize}'. Default: None.
- `--incl_RDKit_feats`: Whether or not to include RDKit features, if the data is to be featurised first. If 'fingerprint' isn't specified, this argument is ignored.
- `--scaler`: Type of scaler to use, if the data is to be scaled first. Valid choices are 'Standard' and 'MinMax'. Default: None.
- `--n_jobs`: Number of jobs to run in parallel for hyperparameter tuning. Default: 1.

**Note**: Metrics are not case-sensitive. See [here](astra/metrics.py) for an overview of available metrics.

The benchmark script will create the following files under `results/<name>`:
- `default_CV.pkl`: A dictionary containing CV scores of the main and secondary metrics for all models using default hyperparameters.
- `nested_CV.pkl`: A dictionary containing CV scores of the main and secondary metrics for all models with optimised hyperparameters using nested grid-search.
- `final_CV_results.csv`: Final CV results (`cv_results_` of `GridSearchCV`).
- `final_model.pkl`: The best performing model, refit on the whole dataset.
- `final_hyperparameters.pkl`: A dictionary of the optimal hyperparameters.

### Comparing Results

Use `astra compare` to statistically analyse benchmark results:

```bash
astra compare <CV_results_path>
```
where `CV_results_path` is the path to the directory containing the CV results.

Options:
- `--main_metric`: The main metric to use for comparison.
- `--sec_metrics`: Secondary metrics to use for comparison.

## Requirements
- Python >=3.11
- Core dependencies (automatically installed):
  - numpy
  - pandas
  - lightgbm
  - pingouin
  - xgboost
  - torch
  - rdkit
  - torchaudio
  - torchvision
  - catboost
  - deepchem
  - scikit-learn
  - scikit-posthocs

### Copyright

Copyright (c) 2024, Wojtek Treyde


#### Acknowledgements
 
Project based on the 
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.10.
