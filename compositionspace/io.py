"""Utility functions to load (atom probe) data from files."""

import h5py
import numpy as np


def get_reconstructed_positions(file_path: str = "", verbose: bool = False):
    """Get (n, 3) array of reconstructed positions."""
    with h5py.File(file_path, "r") as h5r:
        trg = "/entry1/atom_probe/reconstruction/reconstructed_positions"
        xyz = h5r[trg][:, :]
        print(f"Load reconstructed positions shape {np.shape(xyz)}, type {type(xyz)}, dtype {xyz.dtype}")
        return (xyz, "nm")


def get_iontypes_info(file_path: str = "", verbose: bool = False):
    """Get dictionary of iontypes with human-readable name and identifier."""
    with h5py.File(file_path, "r") as h5r:
        trg = "/entry1/atom_probe/ranging/peak_identification"
        n_ion_types = len(h5r[trg])
        iontypes = {}
        for ion_id in np.arange(0, n_ion_types):
            iontypes[f"ion{ion_id}"] = (h5r[f"{trg}/ion{ion_id}/name"][()].decode("utf8"), np.uint8(ion_id))
        print(f"Load iontypes information {n_ion_types} types distinguished")
        if verbose:
            for key, val in iontypes.items():
                print(f"\t{key}, {val}")
        return iontypes


def get_iontypes(file_path: str = "", verbose: bool = False):
    """Get (n,) array of ranged iontype."""
    with h5py.File(file_path, "r") as h5r:
        trg = "/entry1/iontypes/iontypes"
        ityp = h5r[trg][:]
        print(f"Load ranged iontypes shape {np.shape(ityp)}, type {type(ityp)}, dtype {ityp.dtype}")
        return (ityp, None)
