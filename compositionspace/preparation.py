
import os
import yaml
import numpy as np
import h5py
import datetime as dt
from compositionspace.get_gitrepo_commit import get_repo_last_commit
from compositionspace.utils import floor_to_multiple, ceil_to_multiple

# https://stackoverflow.com/questions/47182183/pandas-chained-assignment-warning-exception-handling
# pd.options.mode.chained_assignment = None


class ProcessPreparation:
    def __init__(self, 
                 config_file_path: str = "", 
                 results_file_path: str = "",
                 entry_id: int = 1,
                 verbose: bool = False):
        # why should inputfile be a dictionary, better always document changes made in file
        self.config = {}
        with open(config_file_path, "r") as yml:
            self.config = yaml.safe_load(yml)  # TODO try, except
        self.config["config_file_path"] = config_file_path
        self.config["results_file_path"] = results_file_path
        self.config["entry_id"] = entry_id
        self.verbose = verbose
        self.version = get_repo_last_commit()
        # if not os.path.exists(self.config['output_path']):
        #     os.mkdir(self.config['output_path'])
        self.voxel_identifier = None
        self.n_ions = 0
        self.aabb3d = None
        self.extent = None
        self.lu_ityp_voxel_id_evap_id = None

    def write_init_results(self):
        """Init a NeXus results file."""
        with h5py.File(self.config["results_file_path"], "w") as h5w:
            h5w.attrs["NX_class"] = "NXroot"
            h5w.attrs["file_name"] = self.config["results_file_path"]
            h5w.attrs["file_time"] = dt.datetime.now(dt.timezone.utc).isoformat()  # .replace("+00:00", "Z")
            # /@file_update_time
            h5w.attrs["NeXus_repository"] = "TODO"  # f"https://github.com/FAIRmat-NFDI/nexus_definitions/blob/get_nexus_version_hash()"
            h5w.attrs["NeXus_version"] = "TODO"  # f"get_nexus_version()"
            h5w.attrs["HDF5_version"] = ".".join(map(str, h5py.h5.get_libversion()))
            h5w.attrs["h5py_version"] = h5py.__version__

            trg = f"/entry{self.config['entry_id']}"
            grp = h5w.create_group(trg)
            grp.attrs["NX_class"] = "NXentry"
            dst = h5w.create_dataset(f"{trg}/definition", data="NXapm_compositionspace_results")
            trg = f"/entry{self.config['entry_id']}/program"
            grp = h5w.create_group(trg)
            grp.attrs["NX_class"] = "NXprogram"
            dst = h5w.create_dataset(f"{trg}/program", data="compositionspace")
            dst.attrs["version"] = get_repo_last_commit()

    def define_voxelization_grid(self, xyz):
        column_names = ["x", "y", "z"]
        # initialize extent (number of cells) along x, y, z axes
        self.n_ions = np.shape(xyz)[0]
        self.extent = [0, 0, 0]
        # initialize min, max bounds for x, y, z
        self.aabb3d = np.reshape([np.finfo(np.float32).max, np.finfo(np.float32).min,
                np.finfo(np.float32).max, np.finfo(np.float32).min,
                np.finfo(np.float32).max, np.finfo(np.float32).min], (3, 2), order="C")
        if self.verbose:
            print(self.aabb3d)
        n_ions = np.shape(xyz)[0]
        self.voxel_identifier = np.asarray(np.zeros(n_ions), np.uint32)
        print(f"shape {np.shape(self.voxel_identifier)}")
        # edge length of cubic cells/voxels in nm
        dedge = self.config["voxel_edge_length"]
        for axis_id in [0, 1, 2]:
            column_name = column_names[axis_id]
            # i = np.asarray(df_lst[0].loc[:, column_name], np.float32)
            self.aabb3d[axis_id, 0] = floor_to_multiple(np.min((self.aabb3d[axis_id, 0], np.min(xyz[:, axis_id]))), dedge)
            self.aabb3d[axis_id, 1] = ceil_to_multiple(np.max((self.aabb3d[axis_id, 1], np.max(xyz[:, axis_id]))), dedge)
            self.extent[axis_id] = np.uint32((self.aabb3d[axis_id, 1] - self.aabb3d[axis_id, 0]) / dedge)
            print(f"self.aabb3d {self.aabb3d[axis_id, :]}, extent {self.extent[axis_id]}")
            bins = np.linspace(self.aabb3d[axis_id, 0] + dedge, self.aabb3d[axis_id, 0] + (self.extent[axis_id] * dedge), num=self.extent[axis_id], endpoint=True)
            print(bins)
            if axis_id == 0:
                self.voxel_identifier = self.voxel_identifier + (np.asarray(np.digitize(xyz[:, axis_id], bins, right=True), np.uint32) * 1)
            elif axis_id == 1:
                self.voxel_identifier = self.voxel_identifier + (np.asarray(np.digitize(xyz[:, axis_id], bins, right=True), np.uint32) * np.uint32(self.extent[0]))
            else:
                self.voxel_identifier = self.voxel_identifier + (np.asarray(np.digitize(xyz[:, axis_id], bins, right=True), np.uint32) * np.uint32(self.extent[0]) * np.uint32(self.extent[1]))
        if self.verbose:
            print(self.voxel_identifier[0:10])
        print(np.max(self.voxel_identifier))

    def define_lookup_table(self, itypes):
        """Define a lookup table for summary statistics on voxel composition fast."""
        ion_struct = [('iontype', np.uint8), ('voxel_id', np.uint32), ('evap_id', np.uint32)]
        n_ions = np.shape(itypes)[0]
        self.lu_ityp_voxel_id_evap_id = np.zeros(n_ions, dtype=ion_struct)
        self.lu_ityp_voxel_id_evap_id["iontype"] = itypes[:]
        self.lu_ityp_voxel_id_evap_id["voxel_id"] = self.voxel_identifier
        # del voxel_identifier
        self.lu_ityp_voxel_id_evap_id["evap_id"] = np.asarray(np.linspace(1, n_ions, num=n_ions, endpoint=True), np.uint32)
        if self.verbose:
            print(self.lu_ityp_voxel_id_evap_id[0:10])
        self.lu_ityp_voxel_id_evap_id = np.sort(self.lu_ityp_voxel_id_evap_id, kind="stable", order=["iontype", "voxel_id", "evap_id"])
        if self.verbose:
            print(self.lu_ityp_voxel_id_evap_id[0:10])
            print(self.lu_ityp_voxel_id_evap_id[-10::])

    def write_voxelization_grid_info(self):
        # voxelization
        if not os.path.isfile(self.config["results_file_path"]):
            raise LookupError(f"Results file {self.config['results_file_path']} has not been instantiated!")

        with h5py.File(self.config["results_file_path"], "a") as h5w:
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
            dst = h5w.create_dataset(f"{trg}/origin", data=np.asarray([self.aabb3d[0, 0], self.aabb3d[1, 0], self.aabb3d[2, 0]], np.float64))
            dst.attrs["units"] = "nm"
            dst = h5w.create_dataset(f"{trg}/symmetry", data="cubic")
            dedge = self.config["voxel_edge_length"]
            dst = h5w.create_dataset(f"{trg}/cell_dimensions", data=np.asarray([dedge, dedge, dedge], np.float64))
            dst.attrs["units"] = "nm"
            dst = h5w.create_dataset(f"{trg}/extent", data=np.asarray(self.extent, np.uint32))  # max. 2*32 cells
            identifier_offset = 0
            dst = h5w.create_dataset(f"{trg}/identifier_offset", data=np.uint64(identifier_offset))  # start counting cells from 0

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
            dst = h5w.create_dataset(f"{trg}/position", compression="gzip", compression_opts=1, data=position)
            dst.attrs["units"] = "nm"
            del position

            voxel_id = identifier_offset
            coordinate = np.zeros([c, 3], np.uint32)
            for k in np.arange(0, self.extent[2]):
                for j in np.arange(0, self.extent[1]):
                    for i in np.arange(0, self.extent[0]):
                        coordinate[voxel_id, :] = [i, j, k]
                        voxel_id += 1
            dst = h5w.create_dataset(f"{trg}/coordinate", compression="gzip", compression_opts=1, data=coordinate)
            del coordinate

    def write_voxelization_results(self, ityp_info: dict):
        with h5py.File(self.config["results_file_path"], "a") as h5w:
            trg = f"/entry{self.config['entry_id']}/voxelization/cg_grid"
            dst = h5w.create_dataset(f"{trg}/voxel_identifier", compression="gzip", compression_opts=1, data=self.voxel_identifier)

            c = np.prod(self.extent)
            print(f"")
            # now just add weight/counts for a the iontype-specific part of the lookup-table
            print(f"Cardinality is {c} means we have to visit so that many entries in the lookup table "
                  f"{np.sum(self.lu_ityp_voxel_id_evap_id['iontype'] == 0)} but by virtue of construction "
                  f"of the lookup table all the indices will be close in cache.")
            total_weights = np.zeros(c, np.float64)
            for ityp in np.arange(0, len(ityp_info)):
                inds = np.argwhere(self.lu_ityp_voxel_id_evap_id["iontype"] == ityp)
                offsets = (np.min(inds), np.max(inds))
                # print(f"offsets {offsets}")
                # these are inclusive [min, max] array indices to use on lu_ityp_voxel_id_evap_id !
                
            # alternatively one could make two loops where in the first an offset lookup table is generated
            # after this point one can drop the iontype and evap_id columns from the lu_ityp_voxel_id_evap_id lookup table
                ityp_weights = np.zeros(c, np.float64)
                for offset in np.arange(offsets[0], offsets[1] + 1):
                    idx = self.lu_ityp_voxel_id_evap_id["voxel_id"][offset]
                    ityp_weights[idx] += 1.
                # print(f"ityp {ityp}, np.sum(ityp_weights) {np.sum(ityp_weights)}")
                
                # atom/molecular ion-type-specific contribution/intensity/count in each voxel/cell
                trg = f"/entry{self.config['entry_id']}/voxelization/ion{ityp}"
                grp = h5w.create_group(f"{trg}")
                grp.attrs["NX_class"] = "NXion"
                dst = h5w.create_dataset(f"{trg}/name", data=ityp_info[f"ion{ityp}"][0])
                dst = h5w.create_dataset(f"{trg}/weight", compression="gzip", compression_opts=1, data=ityp_weights)
                dst.attrs["units"] = "a.u."  
                
                total_weights += ityp_weights
                print(f"ityp {ityp}, np.sum(total_weights) {np.sum(total_weights)}")
            print(f"cardinality of cg_grid {c}, n_ions {self.n_ions}")

            # total atom/molecular ion contribution/intensity/count in each voxel/cell
            trg = f"/entry{self.config['entry_id']}/voxelization"
            dst = h5w.create_dataset(f"{trg}/total", compression="gzip", compression_opts=1, data=total_weights)
            dst.attrs["units"] = "a.u."

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