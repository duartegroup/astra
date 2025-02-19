"""
Description
-----------
This module contains functions to calculate molecular descriptors and fingerprints using the RDKit library.

Functions
---------
RDKit_descriptors(smiles_list)
    Calculate RDKit molecular descriptors for a list of SMILES strings.
get_fingerprints(smiles_list, fp_type, radius=2, fpsize=1024)
    Calculate fingerprints for a list of SMILES strings.
"""

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Avalon.pyAvalonTools import GetAvalonFP
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator


def RDKit_descriptors(smiles_list: list[str]) -> np.ndarray:
    """
    Calculate RDKit molecular descriptors for a list of SMILES strings.

    Parameters
    ----------
    smiles_list : list of str
        A list of SMILES strings.

    Returns
    -------
    np.ndarray
        A NumPy array of molecular descriptors.
    """
    mols = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]
    calc = MolecularDescriptorCalculator([x[0] for x in Descriptors._descList])
    desc_names = calc.GetDescriptorNames()

    descriptors = []
    for mol in mols:
        if mol is not None:
            mol = Chem.AddHs(mol)
            d = calc.CalcDescriptors(mol)
            descriptors.append(d)
        else:
            descriptors.append([None] * len(desc_names))
    return np.array(descriptors)


def get_fingerprints(
    smiles_list: list[str],
    fp_type: str,
    radius: int | None = 2,
    fpsize: int | None = 1024,
) -> np.ndarray:
    """
    Calculate fingerprints for a list of SMILES strings.

    Parameters
    ----------
    smiles_list : list of str
        A list of SMILES strings.
    fp_type : str
        Type of fingerprint to compute. Valid choices are:
        ['Morgan', 'Avalon', 'RDKit', 'MACCS', 'AtomPair', 'TopTorsion'].
    radius : int or None, default=None
        The radius of the Morgan fingerprint.
    fpsize : int or None, default=None
        The size of the Morgan fingerprint.

    Returns
    -------
    np.ndarray
        A NumPy array of fingerprints.
    """
    if fp_type == "Morgan":
        assert radius is not None and fpsize is not None
        fps = np.array(
            [
                np.array(
                    rdMolDescriptors.GetMorganFingerprintAsBitVect(
                        Chem.MolFromSmiles(s),
                        radius=radius,
                        nBits=fpsize,
                    )
                )
                for s in smiles_list
            ]
        )
    elif fp_type == "Avalon":
        fps = np.array(
            [np.array(GetAvalonFP(Chem.MolFromSmiles(s))) for s in smiles_list]
        )
    elif fp_type == "RDKit":
        fps = np.array(
            [np.array(Chem.RDKFingerprint(Chem.MolFromSmiles(s))) for s in smiles_list]
        )
    elif fp_type == "MACCS":
        fps = np.array(
            [
                np.array(
                    rdMolDescriptors.GetMACCSKeysFingerprint(Chem.MolFromSmiles(s))
                )
                for s in smiles_list
            ]
        )
    elif fp_type == "AtomPair":
        fps = np.array(
            [
                np.array(
                    rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(
                        Chem.MolFromSmiles(s)
                    )
                )
                for s in smiles_list
            ]
        )
    elif fp_type == "TopTorsion":
        fps = np.array(
            [
                np.array(
                    rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(
                        Chem.MolFromSmiles(s)
                    )
                )
                for s in smiles_list
            ]
        )
    else:
        raise ValueError("Invalid fingerprint type.")

    return fps
