"""
Description
-----------
This module contains functions for dataset splitting.
"""

import pandas as pd
import deepchem as dc
import numpy as np


def get_splits(data: pd.DataFrame, split: str, n_folds: int = 5) -> pd.DataFrame:
    """
    Split data into n_folds using the specified split method and the DeepChem library.

    Parameters
    ----------
    data : pd.DataFrame
        The dataset to split.
    split : str
        The splitting method to use. Options are 'Scaffold' and 'Fingerprint'.
    n_folds : int, default=5
        The number of folds to split the data into.

    Returns
    -------
    pd.DataFrame
        The dataset with an additional column 'fold' containing the fold number for each sample.
    """
    assert n_folds > 1, "n_folds must be greater than 1"
    assert "SMILES" in data.columns, "SMILES column not found in data"

    smiles = data.SMILES
    Xs = np.zeros(len(data))
    dc_dataset = dc.data.DiskDataset.from_numpy(X=Xs, ids=smiles)

    if split == "Scaffold":
        splitter = dc.splits.ScaffoldSplitter()
    elif split == "Fingerprint":
        splitter = dc.splits.FingerprintSplitter()
    else:
        raise ValueError("Invalid split method")

    folds: list[tuple[dc.data.Dataset, dc.data.Dataset]] = splitter.k_fold_split(
        dc_dataset, n_folds
    )
    test_smiles = [test.ids for _, test in folds]
    fold_assignments = [
        i for s in smiles for i, fold in enumerate(test_smiles) if s in fold
    ]

    assert len(fold_assignments) == len(data), "Not all samples were assigned to a fold"
    data["fold"] = fold_assignments

    return data
