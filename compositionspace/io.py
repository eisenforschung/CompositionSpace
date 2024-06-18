"""Utility functions to load (atom probe) data from files."""

import h5py
import numpy as np
from ase.data import chemical_symbols


def get_reconstructed_positions(file_path: str = "", verbose: bool = False):
    """Get (n, 3) array of reconstructed positions."""
    with h5py.File(file_path, "r") as h5r:
        trg = "/entry1/atom_probe/reconstruction/reconstructed_positions"
        xyz = h5r[trg][:, :]
        print(
            f"Load reconstructed positions shape {np.shape(xyz)}, type {type(xyz)}, dtype {xyz.dtype}"
        )
        return (xyz, "nm")


def get_ranging_info(file_path: str = "", verbose: bool = False):
    """Get dictionary of iontypes with human-readable name and identifier."""
    with h5py.File(file_path, "r") as h5r:
        trg = "/entry1/atom_probe/ranging/peak_identification"
        n_ion_types = len(h5r[trg])
        iontypes: dict = {}
        for ion_id in np.arange(0, n_ion_types):
            iontypes[f"ion{ion_id}"] = (
                h5r[f"{trg}/ion{ion_id}/name"][()].decode("utf8"),
                np.uint8(ion_id),
            )
        print(f"{n_ion_types} iontypes distinguished:")
        if verbose:
            for key, val in iontypes.items():
                print(f"\t{key}, {val}")
        chrg_agnostic_iontypes: dict = {}
        elements = set()
        for key, value in iontypes.items():
            chrg_agnostic_name = value[0].replace("+", "").replace("-", "").strip()
            if chrg_agnostic_name in chrg_agnostic_iontypes:
                chrg_agnostic_iontypes[chrg_agnostic_name].append(value[1])
            else:
                chrg_agnostic_iontypes[chrg_agnostic_name] = [value[1]]
            symbols = chrg_agnostic_name.split()
            for symbol in symbols:
                if symbol in chemical_symbols[1::]:
                    elements.add(symbol)
        print(f"{len(chrg_agnostic_iontypes)} charge-agnostic iontypes distinguished:")
        if verbose:
            for key, val in chrg_agnostic_iontypes.items():
                print(f"\t{key}, {val}")
        print(f"{len(elements)} elements distinguished:")
        if verbose:
            for symbol in elements:
                print(symbol)
        return iontypes, chrg_agnostic_iontypes, elements


def get_iontypes(file_path: str = "", verbose: bool = False):
    """Get (n,) array of ranged iontype."""
    with h5py.File(file_path, "r") as h5r:
        trg = "/entry1/iontypes/iontypes"
        ityp = h5r[trg][:]
        print(
            f"Load ranged iontypes shape {np.shape(ityp)}, type {type(ityp)}, dtype {ityp.dtype}"
        )
        return (ityp, None)
