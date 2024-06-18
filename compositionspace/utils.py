"""Utility functions."""

import os
import numpy as np
from ase.data import chemical_symbols


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


# numerics
EPSILON = 1.0e-6
APT_UINT = np.uint64


# exemplar code for testing some functions
# ceil to a multiple of 1.5
# print(ceil_to_multiple(23.0000000000000000000000000000000000000, 1.5))
# floor to a multiple of 1.5
# print(floor_to_multiple(-23.0000000000000000000000000000000000000, 1.5))
