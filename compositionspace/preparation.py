"""Run step 1 of the workflow."""

import datetime as dt
import os

import flatdict as fd
import h5py
import numpy as np
import yaml


from compositionspace import __nexus__version__, __nexus__version__hash__, __version__
from compositionspace.io import (
    get_iontypes,
    get_ranging_info,
    get_reconstructed_positions,
)
from compositionspace.utils import (
    APT_UINT,
    ceil_to_multiple,
    floor_to_multiple,
    get_chemical_element_multiplicities,
)

# https://stackoverflow.com/questions/47182183/pandas-chained-assignment-warning-exception-handling
# pd.options.mode.chained_assignment = None


class ProcessPreparation:
    def __init__(
        self,
        config_file_path: str = "",
        results_file_path: str = "",
        entry_id: int = 1,
        verbose: bool = False,
    ):
        """Initialize the class."""
        self.config = {}
        if os.path.exists(config_file_path):
            with open(config_file_path, "r") as yml:
                self.config = fd.FlatDict(yaml.safe_load(yml), delimiter="/")
        elif isinstance(config_file_path, dict):
            self.config = fd.FlatDict(config_file_path, delimiter="/")
        else:
            raise IOError(f"File {config_file_path} does not exist!")
        self.config["config_file_path"] = config_file_path
        self.config["results_file_path"] = results_file_path
        self.config["entry_id"] = entry_id
        self.verbose = verbose
        self.version = __version__
        self.voxel_identifier = None
        self.n_ions = 0
        self.itypes: dict = {}
        self.elements: set = set()

    def init_ranging(self, itypes: dict, elements: set):
        """Get metadata about itypes and elements."""
        self.itypes = itypes
        self.elements = elements

    def write_init_results(self):
        """Init the NeXus/HDF5 results file."""
        h5w = h5py.File(self.config["results_file_path"], "w")
        h5w.attrs["NX_class"] = "NXroot"
        h5w.attrs["file_name"] = self.config["results_file_path"]
        h5w.attrs["file_time"] = dt.datetime.now(
            dt.timezone.utc
        ).isoformat()  # .replace("+00:00", "Z")
        # /@file_update_time
        h5w.attrs["NeXus_repository"] = (
            f"https://github.com/FAIRmat-NFDI/nexus_definitions/blob/{__nexus__version__hash__}"
        )
        h5w.attrs["NeXus_version"] = __nexus__version__
        h5w.attrs["HDF5_version"] = ".".join(map(str, h5py.h5.get_libversion()))
        h5w.attrs["h5py_version"] = h5py.__version__

        trg = f"/entry{self.config['entry_id']}"
        grp = h5w.create_group(trg)
        grp.attrs["NX_class"] = "NXentry"
        dst = h5w.create_dataset(
            f"{trg}/definition", data="NXapm_compositionspace_results"
        )
        trg = f"/entry{self.config['entry_id']}/program"
        grp = h5w.create_group(trg)
        grp.attrs["NX_class"] = "NXprogram"
        dst = h5w.create_dataset(f"{trg}/program", data="compositionspace")
        dst.attrs["version"] = __version__
        dst.attrs["url"] = (
            f"https://github.com/eisenforschung/CompositionSpace/releases/tag/{__version__}"
        )
        h5w.close()

    def define_voxelization_grid(self, xyz):
        """Define grid with which to discretize reconstructed ion positions and voxelize."""
        # initialize extent (number of cells) along x, y, z axes
        self.n_ions = np.shape(xyz)[0]
        self.extent = [0, 0, 0]
        self.origin = None
        # initialize min, max bounds for x, y, z
        self.aabb3d = np.reshape(
            [
                np.finfo(np.float32).max,
                np.finfo(np.float32).min,
                np.finfo(np.float32).max,
                np.finfo(np.float32).min,
                np.finfo(np.float32).max,
                np.finfo(np.float32).min,
            ],
            (3, 2),
            order="C",
        )
        if self.verbose:
            print(self.aabb3d)
        n_ions = np.shape(xyz)[0]
        self.voxel_identifier = np.asarray(np.zeros(n_ions), APT_UINT)
        print(f"shape {np.shape(self.voxel_identifier)}")
        dedge = self.config["voxelization/edge_length"]  # cubic voxels, nm
        for dim in [0, 1, 2]:  # 0 -> x, 1 -> y, 2 -> z
            print(f"dim {dim}")
            self.aabb3d[dim, 0] = floor_to_multiple(
                np.min((self.aabb3d[dim, 0], np.min(xyz[:, dim]))), dedge
            ) - (2.0 * dedge)
            print(
                f"\tnp.min(xyz[:, axis_id]) {np.min(xyz[:, dim])} >>>> {self.aabb3d[dim, 0]}"
            )
            self.aabb3d[dim, 1] = ceil_to_multiple(
                np.max((self.aabb3d[dim, 1], np.max(xyz[:, dim]))), dedge
            ) + (2.0 * dedge)
            print(
                f"\tnp.max(xyz[:, axis_id]) {np.max(xyz[:, dim])} >>>> {self.aabb3d[dim, 1]}"
            )
            self.extent[dim] = APT_UINT(
                (self.aabb3d[dim, 1] - self.aabb3d[dim, 0]) / dedge
            )
            print(
                f"\tself.aabb3d axis_id {dim}, {self.aabb3d[dim, :]}, extent {self.extent[dim]}"
            )
            bins = np.linspace(
                self.aabb3d[dim, 0] + dedge,
                self.aabb3d[dim, 0] + (self.extent[dim] * dedge),
                num=self.extent[dim],
                endpoint=True,
            )
            # print(f"\t{bins}")
            if dim == 0:
                self.voxel_identifier = self.voxel_identifier + (
                    np.asarray(np.digitize(xyz[:, dim], bins, right=True), APT_UINT) * 1
                )
            elif dim == 1:
                self.voxel_identifier = self.voxel_identifier + (
                    np.asarray(np.digitize(xyz[:, dim], bins, right=True), APT_UINT)
                    * APT_UINT(self.extent[0])
                )
            elif dim == 2:
                self.voxel_identifier = self.voxel_identifier + (
                    np.asarray(np.digitize(xyz[:, dim], bins, right=True), APT_UINT)
                    * APT_UINT(self.extent[0])
                    * APT_UINT(self.extent[1])
                )
        if np.prod(self.extent) >= 1:
            print(f"np.max(self.voxel_identifier) {np.max(self.voxel_identifier)}")
            print(f"np.prod(self.extent) {np.prod(self.extent)}")
        else:
            raise ValueError("Voxelization grid has no cell!")
        self.extent = np.asarray(self.extent, np.uint64)
        self.origin = np.asarray(
            [self.aabb3d[0, 0], self.aabb3d[1, 0], self.aabb3d[2, 0]], np.float64
        )

    def define_lookup_table(self, itypes, evaporation_id: bool = False):
        """Define a lookup table for fast summary statistics of voxel contributions."""
        if evaporation_id:
            ion_struct = [
                ("iontype", np.uint8),
                ("voxel_id", APT_UINT),
                ("evap_id", APT_UINT),
            ]
        else:
            ion_struct = [("iontype", np.uint8), ("voxel_id", APT_UINT)]
        n_ions = np.shape(itypes)[0]
        self.lu_ityp_voxel_id_evap_id = np.zeros(n_ions, dtype=ion_struct)
        self.lu_ityp_voxel_id_evap_id["iontype"] = itypes[:]
        self.lu_ityp_voxel_id_evap_id["voxel_id"] = self.voxel_identifier
        # del voxel_identifier
        if evaporation_id:
            self.lu_ityp_voxel_id_evap_id["evap_id"] = np.asarray(
                np.linspace(1, n_ions, num=n_ions, endpoint=True), APT_UINT
            )
        # we sort this LU by iontype such that we can get all ids of voxels contributing to this iontype
        if evaporation_id:
            self.lu_ityp_voxel_id_evap_id = np.sort(
                self.lu_ityp_voxel_id_evap_id,
                kind="stable",
                order=["iontype", "voxel_id", "evap_id"],
            )
        else:
            self.lu_ityp_voxel_id_evap_id = np.sort(
                self.lu_ityp_voxel_id_evap_id,
                kind="stable",
                order=["iontype", "voxel_id"],
            )

    def write_voxelization_grid_info(self):
        """Write metadata that detail the discretization grid to NeXus/HDF5."""
        if not os.path.isfile(self.config["results_file_path"]):
            raise IOError(
                f"Results file {self.config['results_file_path']} does not exist!"
            )
        h5w = h5py.File(self.config["results_file_path"], "a")
        trg = f"/entry{self.config['entry_id']}/voxelization"
        grp = h5w.create_group(trg)
        grp.attrs["NX_class"] = "NXprocess"
        dst = h5w.create_dataset(f"{trg}/sequence_index", data=np.uint64(1))

        trg = f"/entry{self.config['entry_id']}/voxelization/cg_grid"
        grp = h5w.create_group(trg)
        grp.attrs["NX_class"] = "NXcg_grid"
        dst = h5w.create_dataset(f"{trg}/dimensionality", data=np.uint64(3))
        c = np.prod(self.extent)
        dst = h5w.create_dataset(f"{trg}/cardinality", data=np.uint64(c))
        dst = h5w.create_dataset(f"{trg}/origin", data=self.origin)
        dst.attrs["units"] = "nm"
        dst = h5w.create_dataset(f"{trg}/symmetry", data="cubic")
        dedge = self.config["voxelization/edge_length"]
        dst = h5w.create_dataset(
            f"{trg}/cell_dimensions",
            data=np.asarray([dedge, dedge, dedge], np.float64),
        )
        dst.attrs["units"] = "nm"
        dst = h5w.create_dataset(f"{trg}/extent", data=self.extent)
        identifier_offset = 0  # we count cells starting from this value
        dst = h5w.create_dataset(
            f"{trg}/identifier_offset", data=np.uint64(identifier_offset)
        )

        voxel_id = identifier_offset
        position = np.zeros([c, 3], np.float64)
        for k in np.arange(0, self.extent[2]):
            z = self.aabb3d[2, 0] + (0.5 + k) * dedge
            for j in np.arange(0, self.extent[1]):
                y = self.aabb3d[1, 0] + (0.5 + j) * dedge
                for i in np.arange(0, self.extent[0]):
                    x = self.aabb3d[0, 0] + (0.5 + i) * dedge
                    position[voxel_id, :] = [x, y, z]
                    voxel_id += 1
        dst = h5w.create_dataset(
            f"{trg}/position", compression="gzip", compression_opts=1, data=position
        )
        dst.attrs["units"] = "nm"
        del position

        voxel_id = identifier_offset
        coordinate = np.zeros([c, 3], np.uint64)
        for k in np.arange(0, self.extent[2]):
            for j in np.arange(0, self.extent[1]):
                for i in np.arange(0, self.extent[0]):
                    coordinate[voxel_id, :] = [i, j, k]
                    voxel_id += 1
        dst = h5w.create_dataset(
            f"{trg}/coordinate",
            compression="gzip",
            compression_opts=1,
            data=coordinate,
        )
        del coordinate
        h5w.close()

    def write_voxelization_results(self):
        """Perform voxelization and write results to NeXus/HDF5."""
        if not os.path.isfile(self.config["results_file_path"]):
            raise IOError(
                f"Results file {self.config['results_file_path']} does not exist!"
            )

        c = np.prod(self.extent)
        elem_id = {}
        elem_cnts = {}
        for idx, symbol in enumerate(self.elements):
            elem_cnts[symbol] = np.zeros(c, APT_UINT)
            elem_id[symbol] = idx
        if self.verbose:
            print(elem_cnts)
            print(elem_id)
        total_cnts = np.zeros(c, APT_UINT)

        h5w = h5py.File(self.config["results_file_path"], "a")
        trg = f"/entry{self.config['entry_id']}/voxelization/cg_grid"
        dst = h5w.create_dataset(
            f"{trg}/voxel_identifier",
            compression="gzip",
            compression_opts=1,
            data=self.voxel_identifier,
        )

        for ityp, tpl in self.itypes.items():
            if ityp == "ion0":
                continue
            print(f"{ityp}, {tpl}:")
            multiplicities = get_chemical_element_multiplicities(tpl[0], verbose=True)

            inds = np.argwhere(self.lu_ityp_voxel_id_evap_id["iontype"] == tpl[1])
            offsets = (np.min(inds), np.max(inds))
            for symbol, cnts in multiplicities.items():
                for offset in np.arange(offsets[0], offsets[1] + 1):
                    idx = self.lu_ityp_voxel_id_evap_id["voxel_id"][offset]
                    elem_cnts[symbol][idx] += cnts
                    # offsets are inclusive [min, max] indices to use on lu_ityp_voxel_id_evap_id !
                    # alternatively, one could make two loops where in the first an offset lookup table is generated
                    # after this point one can drop the iontype and evap_id columns from the lu_ityp_voxel_id_evap_id lookup table

        for symbol in elem_cnts:
            # atom/molecular ion-type-specific contribution/intensity/count in each voxel/cell
            trg = f"/entry{self.config['entry_id']}/voxelization/element{elem_id[symbol] + 1}"
            print(f"{trg}, {symbol}")
            grp = h5w.create_group(f"{trg}")
            grp.attrs["NX_class"] = "NXion"
            dst = h5w.create_dataset(f"{trg}/name", data=str(symbol))
            dst = h5w.create_dataset(
                f"{trg}/weight",
                compression="gzip",
                compression_opts=1,
                data=elem_cnts[symbol],
            )
            total_cnts += elem_cnts[symbol]
            print(
                f"symbol {symbol}, idx {elem_id[symbol]}, np.sum(elem_cnts[symbol]) {np.sum(elem_cnts[symbol])}, np.sum(total_cnts) {np.sum(total_cnts)}"
            )
        print(f"n_ions {self.n_ions}")

        # total atom/molecular ion contribution/intensity/count in each voxel/cell
        trg = f"/entry{self.config['entry_id']}/voxelization"
        dst = h5w.create_dataset(
            f"{trg}/weight", compression="gzip", compression_opts=1, data=total_cnts
        )
        h5w.close()

    def run(self, recon_file_path: str, range_file_path: str):
        xyz_val, xyz_unit = get_reconstructed_positions(recon_file_path)
        if recon_file_path.lower().endswith(
            (".apt", ".pos")
        ) and range_file_path.lower().endswith(".rrng"):
            ityp_info, nochrg_ityp_info, elements = get_ranging_info(
                range_file_path, verbose=True
            )
            ityp_val = get_iontypes(recon_file_path, ityp_info)
        elif recon_file_path.lower().endswith(
            ".nxs"
        ) and range_file_path.lower().endswith(".nxs"):
            ityp_info, nochrg_ityp_info, elements = get_ranging_info(
                recon_file_path, verbose=True
            )
            ityp_val = get_iontypes(range_file_path)
        else:
            raise IOError(
                f"((APT or POS) and RRNG) or (NXS and NXS) are the only supported combinations!"
            )

        self.init_ranging(ityp_info, elements)
        self.write_init_results()
        self.define_voxelization_grid(xyz_val)
        self.define_lookup_table(ityp_val)
        self.write_voxelization_grid_info()
        self.write_voxelization_results()

        # For a large number of voxels, say a few million and dozens of iontypes storing all
        # ityp_weights in main memory might not be useful, instead these should be stored in the HDF5 file
        # inside the loop and ones the loop is completed, i.e. each total weight for each voxel known
        # we should update the data in the HDF5 file, alternatively one could also just store the
        # weights instead of the compositions and then compute the composition with a linear in c*ityp time
        # complex division, there are even more optimizations one could do, but probably using
        # multithreading would be a good start before dwelling deeper already this code here is
        # faster than the original one despite the fact that it works on the entire portland wang
        # dataset with 4.868 mio ions, while the original test dataset includes only 1.75 mio ions
        # the top part of the dataset also the code is much shorter to read and eventually even
        # more robust wrt to how ions are binned with the rectangular transfer function
        # one should say that this particular implementation (like the original) one needs
        # substantial modification when one considers a delocalization kernel which spreads
        # the weight of an ion into the neighboring voxels, this is what paraprobe-nanochem does
        # one can easily imagine though that the results of this voxelization step can both be
        # fed into the composition clustering step and here is then also the clear connection
        # where the capabilities for e.g. the APAV open-source Python library end and Alaukik's
        # ML/AI work really shines, in fact until now all code including the slicing could have
        # equally been achieved with paraprobe-nanochem.
        # Also the excessive reimplementation of file format I/O functions in datautils should
        # be removed. There is an own Python library for just doing that more robustly and
        # capable of handling all sorts of molecular ions and charge state analyses included
