"""Run step 4 of the workflow."""

import os

import flatdict as fd
import h5py
import numpy as np
import yaml
from sklearn.cluster import DBSCAN

from compositionspace.get_gitrepo_commit import get_repo_last_commit
from compositionspace.utils import APT_UINT


class ProcessClustering:
    def __init__(
        self,
        config_file_path: str = "",
        results_file_path: str = "",
        entry_id: int = 1,
        verbose: bool = False,
    ):
        """Initialize the class."""
        # why should inputfile be a dictionary, better always document changes made in file
        self.config = {}
        if os.path.exists(config_file_path):
            with open(config_file_path, "r") as yml:
                self.config = fd.FlatDict(yaml.safe_load(yml), delimiter="/")
        else:
            raise IOError(f"File {config_file_path} does not exist!")
        if not os.path.exists(results_file_path):
            raise IOError(f"File {results_file_path} does not exist!")
        if entry_id < 1:
            raise ValueError(f"entry_id needs to be at least 1 !")
        self.config["config_file_path"] = config_file_path
        self.config["results_file_path"] = results_file_path
        self.config["entry_id"] = entry_id
        self.verbose = verbose
        self.version = get_repo_last_commit()

    def run(self):
        """Perform DBScan clustering for each Gaussian mixture model result."""
        eps = self.config["clustering/dbscan/eps"]
        min_samples = self.config["clustering/dbscan/min_samples"]
        print(f"DBScan configuration: eps {eps} nm, min_samples {min_samples}")

        h5r = h5py.File(self.config["results_file_path"], "r")
        ic_results_group_names = list(
            h5r[f"/entry{self.config['entry_id']}/segmentation/ic_opt"].keys()
        )
        print(ic_results_group_names)
        h5r.close()

        h5w = h5py.File(self.config["results_file_path"], "a")
        trg = f"/entry{self.config['entry_id']}/clustering"
        grp = h5w.create_group(trg)
        grp.attrs["NX_class"] = "NXprocess"
        sequence_idx = 4
        if self.config["autophase/use"]:
            sequence_idx += 1
        dst = h5w.create_dataset(f"{trg}/sequence_index", data=np.uint64(sequence_idx))
        trg = f"/entry{self.config['entry_id']}/clustering/ic_opt"
        grp = h5w.create_group(trg)
        grp.attrs["NX_class"] = "NXobject"
        h5w.close()

        # n_ic_runs = sum(1 for grpnm in ic_results_group_names if grpnm.startswith("cluster_analysis"))
        for grpnm in ic_results_group_names:
            print(grpnm)
            if not grpnm.startswith("cluster_analysis"):
                continue
            ic_run_id = int(grpnm.replace("cluster_analysis", ""))
            print(f"ic_run_id {ic_run_id} >>>>")

            # using here explicitly blocking calls for open and close as working with a "with h5py.File ..." ran in conflicts
            h5r = h5py.File(self.config["results_file_path"], "r")
            phase_identifier = h5r[
                f"/entry{self.config['entry_id']}/segmentation/ic_opt/cluster_analysis{ic_run_id}/y_pred"
            ][:]
            all_vxl_pos = h5r[
                f"/entry{self.config['entry_id']}/voxelization/cg_grid/position"
            ][:, :]
            print(
                f"np.shape(all_vxl_pos) {np.shape(all_vxl_pos)} list(set(phase_identifier) {list(set(phase_identifier))}"
            )
            n_max_phase_identifier = np.max(tuple(set(phase_identifier)))
            h5r.close()

            h5w = h5py.File(self.config["results_file_path"], "a")
            trg = f"/entry{self.config['entry_id']}/clustering/ic_opt/cluster_analysis{ic_run_id}"
            grp = h5w.create_group(trg)
            grp.attrs["NX_class"] = "NXprocess"
            h5w.close()

            # generate summary representation how the voxel

            for target_phase in np.arange(
                0, n_max_phase_identifier + 1, dtype=np.uint32
            ):
                print(f"\tLoop {target_phase}")
                if target_phase > n_max_phase_identifier:
                    raise ValueError(
                        f"Argument target_phase needs to be <= {n_max_phase_identifier} !"
                    )
                trg_vxl_pos = all_vxl_pos[phase_identifier == target_phase, :]
                trg_vxl_idx = np.asarray(
                    np.linspace(
                        0,
                        np.shape(all_vxl_pos)[0],
                        num=np.shape(all_vxl_pos)[0],
                        endpoint=True,
                    ),
                    APT_UINT,
                )[phase_identifier == target_phase]
                print(f"\tnp.shape(trg_vxl_pos) {np.shape(trg_vxl_pos)}")
                print(f"\tnp.shape(trg_vxl_idx) {np.shape(trg_vxl_idx)}")

                db = DBSCAN(
                    eps=eps,
                    min_samples=min_samples,
                    metric="euclidean",
                    algorithm="kd_tree",
                    leaf_size=10,
                    p=None,
                    n_jobs=-1,
                ).fit(trg_vxl_pos)
                # print(np.unique(db.core_sample_indices_))
                # core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
                # core_samples_mask[db.core_sample_indices_] = True
                # labels = db.labels_
                print(f"\t{len(np.unique(db.labels_))}")
                print(f"\ttype(db.labels_) {type(db.labels_)} dtype {db.labels_.dtype}")
                print(np.unique(db.labels_))

                h5w = h5py.File(self.config["results_file_path"], "a")
                trg = f"/entry{self.config['entry_id']}/clustering/ic_opt/cluster_analysis{ic_run_id}/dbscan{target_phase}"
                grp = h5w.create_group(trg)
                grp.attrs["NX_class"] = "NXprocess"
                dst = h5w.create_dataset(f"{trg}/epsilon", data=np.float64(eps))
                dst.attrs["units"] = "nm"
                dst = h5w.create_dataset(
                    f"{trg}/min_samples", data=np.uint32(min_samples)
                )
                dst = h5w.create_dataset(
                    f"{trg}/label",
                    compression="gzip",
                    compression_opts=1,
                    data=np.asarray(db.labels_, np.int64),
                )
                dst = h5w.create_dataset(
                    f"{trg}/voxel",
                    compression="gzip",
                    compression_opts=1,
                    data=np.asarray(trg_vxl_idx, APT_UINT),
                )
                h5w.close()

                del trg_vxl_pos
                del trg_vxl_idx
                del db
