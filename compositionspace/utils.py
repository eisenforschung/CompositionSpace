"""Utility functions."""

import os
import numpy as np
import h5py
from ase.data import chemical_symbols


# numerics
EPSILON = 1.0e-6
APT_UINT = np.uint64
PRNG_SEED = 42


def ceil_to_multiple(number, multiple):
    return multiple * np.ceil(number / multiple)


def floor_to_multiple(number, multiple):
    return multiple * np.floor(number / multiple)


def get_file_size(file_path: str = ""):
    print(f"{np.around(os.path.getsize(file_path)/1024/1024, decimals=3)} MiB")


def get_chemical_element_multiplicities(ion_name: str, verbose: bool = False) -> dict:
    """Convert human-readable ionname with possible charge information to multiplicity dict."""
    chrg_agnostic_ion_name = ion_name.replace("+", "").replace("-", "").strip()

    multiplicities: dict = {}
    for symbol in chrg_agnostic_ion_name.split():
        if symbol in chemical_symbols[1::]:
            if symbol in multiplicities:
                multiplicities[symbol] += 1
            else:
                multiplicities[symbol] = 1
    if verbose:
        print(f"\t{chrg_agnostic_ion_name}")
        print(f"\t{len(multiplicities)}")
        print(f"\t{multiplicities}")
    return multiplicities


def get_composition_matrix(file_path: str, entry_id: int = 1):
    """Compute (n_ions, n_chemical_class) composition matrix from per-class counts."""
    with h5py.File(file_path, "r") as h5r:
        src = f"/entry{entry_id}/voxelization"
        # element0 is not reported, these are ions of the unknown type
        # element1, ... element<<n>>
        n_chem_classes = sum(
            1 for grpnm in h5r[f"{src}"] if grpnm.startswith("element")
        )
        print(f"Composition matrix has {n_chem_classes} elements")

        total_cnts = np.asarray(h5r[f"{src}/weight"][:], APT_UINT)
        composition_matrix = np.zeros(
            [np.shape(total_cnts)[0], n_chem_classes + 1], np.float64
        )
        for grpnm in h5r[f"{src}"]:
            if grpnm.startswith("element"):
                chem_class_idx = int(grpnm.replace("element", ""))
                print(f"Populating composition table for element{chem_class_idx}")
                etyp_cnts = np.asarray(h5r[f"{src}/{grpnm}/weight"][:], APT_UINT)

                if np.shape(etyp_cnts) == np.shape(total_cnts):
                    # cumsum_cnts += etyp_cnts
                    composition_matrix[:, chem_class_idx] = np.divide(
                        np.asarray(etyp_cnts, np.float64),
                        np.asarray(total_cnts, np.float64),
                        out=composition_matrix[:, chem_class_idx],
                        where=total_cnts >= 1,
                    )
                else:
                    raise ValueError(
                        f"Groupname {grpnm}, length of counts array for element{chem_class_idx} needs to be the same as of counts!"
                    )
        return composition_matrix, n_chem_classes


# exemplar code for testing some functions
# ceil to a multiple of 1.5
# print(ceil_to_multiple(23.0000000000000000000000000000000000000, 1.5))
# floor to a multiple of 1.5
# print(floor_to_multiple(-23.0000000000000000000000000000000000000, 1.5))
