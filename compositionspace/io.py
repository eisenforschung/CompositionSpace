"""Utility functions to load (atom probe) data from files."""

import h5py
import numpy as np
from ase.data import chemical_symbols
from ifes_apt_tc_data_modeling.apt.apt6_reader import ReadAptFileFormat
from ifes_apt_tc_data_modeling.pos.pos_reader import ReadPosFileFormat
from ifes_apt_tc_data_modeling.rrng.rrng_reader import ReadRrngFileFormat
from ifes_apt_tc_data_modeling.utils.definitions import MQ_EPSILON

# load information from HDF5 files that are formatted according to NeXus NXapm
# application definition such content is written e.g. by tools from the paraprobe-toolbox
# such as paraprobe-transcoder and paraprobe-ranger
# however as this creates an implicit
# dependency on paraprobe-toolbox alternative I/O functions are offered to read
# from classical file formats of the community APT and RRNG files to be specific


def get_reconstructed_positions(file_path: str = ""):
    """Get (n, 3) array of reconstructed positions."""
    if file_path.lower().endswith(".apt"):
        apt = ReadAptFileFormat(file_path)
        # print(apt.get_metadata_table())
        xyz = apt.get_reconstructed_positions()
        #print(
        #    f"Load reconstructed positions shape {np.shape(xyz.values)}, type {type(xyz.values)}, dtype {xyz.values.dtype}"
        #)
        return (xyz.values, "nm")
    elif file_path.lower().endswith(".pos"):
        pos = ReadPosFileFormat(file_path)
        xyz = pos.get_reconstructed_positions()
        #print(
        #    f"Load reconstructed positions shape {np.shape(xyz.values)}, type {type(xyz.values)}, dtype {xyz.values.dtype}"
        #)
        return (xyz.values, "nm")
    else:
        with h5py.File(file_path, "r") as h5r:
            trg = "/entry1/atom_probe/reconstruction/reconstructed_positions"
            xyz = h5r[trg][:, :]
            #print(
            #    f"Load reconstructed positions shape {np.shape(xyz)}, type {type(xyz)}, dtype {xyz.dtype}"
            #)
            return (xyz, "nm")


def get_ranging_info(file_path: str = "", verbose: bool = False):
    """Get dictionary of iontypes with human-readable name and identifier."""
    n_ion_types = 0
    iontypes: dict = {}
    if file_path.lower().endswith(".rrng"):
        rrng = ReadRrngFileFormat(file_path, unique=True, verbose=False)
        n_ion_types = 1 + len(rrng.rrng["molecular_ions"])  # 1 + for the unknown type!
        # add the unknown iontype
        iontypes["ion0"] = ("unknown", np.uint8(0), np.float64([0.0, MQ_EPSILON]))
        #print(f'{iontypes["ion0"]}')
        for ion_id, mion in enumerate(rrng.rrng["molecular_ions"]):
            iontypes[f"ion{ion_id + 1}"] = (
                mion.name.values,
                np.uint8(ion_id + 1),
                mion.ranges.values.flatten(),
            )
            #print(f"{iontypes[f'ion{ion_id + 1}']}")
    else:
        with h5py.File(file_path, "r") as h5r:
            trg = "/entry1/atom_probe/ranging/peak_identification"
            n_ion_types = len(h5r[trg])
            # unknown ion with default name ion0 is already included by default!
            for ion_id in np.arange(0, n_ion_types):
                iontypes[f"ion{ion_id}"] = (
                    h5r[f"{trg}/ion{ion_id}/name"][()].decode("utf8"),
                    np.uint8(ion_id),
                    h5r[f"{trg}/ion{ion_id}/mass_to_charge_range"][:, :],
                )
                #print(f"{iontypes[f'ion{ion_id}']}")

    #print(f"{n_ion_types} iontypes distinguished:")
    #if verbose:
    #    for key, val in iontypes.items():
    #        print(f"\t{key}, {val}")
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
    #print(f"{len(chrg_agnostic_iontypes)} charge-agnostic iontypes distinguished:")
    #if verbose:
    #    for key, val in chrg_agnostic_iontypes.items():
    #        print(f"\t{key}, {val}")
    #print(f"{len(elements)} elements distinguished:")
    lex_asc_elements = np.sort(list(elements), kind="stable")
    #if verbose:
    #    for symbol in lex_asc_elements:
    #        print(symbol)
    return iontypes, chrg_agnostic_iontypes, lex_asc_elements


def get_iontypes(file_path: str = "", iontypes: dict = {}):
    """Get (n,) array of ranged iontype."""
    ityp = None
    mq = None
    if file_path.lower().endswith(".apt"):
        if not isinstance(iontypes, dict) or len(iontypes) == 0:
            raise KeyError(
                f"Passing ranging definitions is required when working with APT files!"
            )
        apt = ReadAptFileFormat(file_path)
        # print(apt.get_metadata_table())
        mq = apt.get_mass_to_charge_state_ratio()
    elif file_path.lower().endswith(".pos"):
        if not isinstance(iontypes, dict) or len(iontypes) == 0:
            raise KeyError(
                f"Passing ranging definitions is required when working with POS files!"
            )
        pos = ReadPosFileFormat(file_path)
        # print(apt.get_metadata_table())
        mq = pos.get_mass_to_charge_state_ratio()
    if mq is not None:
        # range on-the-fly using information
        n_ions = np.shape(mq.values)[0]
        ityp = np.zeros((n_ions,), np.uint8)
        for key, value in iontypes.items():
            if key != "ion0":
                ion_id = value[1]
                low = value[2][0]
                high = value[2][1]
                #print(f"Ranging {ion_id} with [{low}, {high}] ...")
                msk = np.argwhere((mq.values >= low) & (mq.values <= high))
                #print(f"{np.shape(msk)}, {msk[0:5]}, {msk[-5:]}")
                ityp[msk] = ion_id
    else:
        with h5py.File(file_path, "r") as h5r:
            trg = "/entry1/iontypes/iontypes"
            ityp = h5r[trg][:]
            #print(
            #    f"Load ranged iontypes shape {np.shape(ityp)}, type {type(ityp)}, dtype {ityp.dtype}"
            #)
    return ityp
