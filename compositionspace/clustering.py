import os
import yaml
import numpy as np
import h5py
from sklearn.cluster import DBSCAN
from compositionspace.get_gitrepo_commit import get_repo_last_commit


class ProcessClustering:
    def __init__(self,
                 config_file_path: str = "",
                 results_file_path: str = "",
                 entry_id: int = 1,
                 verbose: bool = False):
        # why should inputfile be a dictionary, better always document changes made in file
        self.config = {}
        if os.path.exists(config_file_path):
            with open(config_file_path, "r") as yml:
                self.config = yaml.safe_load(yml)
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

    def run_and_write_results(self):
        n_ic_cluster = self.config["n_sel_ic_cluster"]
        eps = self.config["ml_models"]["DBScan"]["eps"]
        min_samples = self.config["ml_models"]["DBScan"]["min_samples"]
        print(f"n_ic_cluster {n_ic_cluster}, eps {eps} nm, min_samples {min_samples}")

        h5r = h5py.File(self.config["results_file_path"], "r")
        phase_identifier = h5r[f"/entry{self.config['entry_id']}/segmentation/ic_opt/cluster_analysis{n_ic_cluster}/y_pred"][:]
        all_vxl_pos = h5r[f"/entry{self.config['entry_id']}/voxelization/cg_grid/position"][:,:]
        print(f"np.shape(all_vxl_pos) {np.shape(all_vxl_pos)} list(set(phase_identifier) {list(set(phase_identifier))}")
        n_max_phase_identifier = np.max(tuple(set(phase_identifier)))
        h5r.close()

        for target_phase in np.arange(0, n_max_phase_identifier + 1):
            print(f"Loop {target_phase}")
            if target_phase > n_max_phase_identifier:
                raise ValueError(f"Argument target_phase needs to be <= {n_max_phase_identifier} !")
            trg_vxl_pos = all_vxl_pos[phase_identifier == target_phase, :]
            print(f"np.shape(trg_vxl_pos) {np.shape(trg_vxl_pos)}")

            db = DBSCAN(eps=eps,
                        min_samples=min_samples,
                        metric="euclidean",
                        algorithm="kd_tree",
                        leaf_size=10,
                        p=None,
                        n_jobs=-1).fit(trg_vxl_pos)
            # print(np.unique(db.core_sample_indices_))
            # core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
            # core_samples_mask[db.core_sample_indices_] = True
            # labels = db.labels_
            print(len(np.unique(db.labels_)))
            print(f"type(db.labels_) {type(db.labels_)} dtype {db.labels_.dtype}")
            print(np.unique(db.labels_))

            h5w = h5py.File(self.config["results_file_path"], "a")
            trg = f"/entry{self.config['entry_id']}/clustering/cluster_analysis{target_phase}"
            grp = h5w.create_group(trg)
            grp.attrs["NX_class"] = "NXprocess"
            dst = h5w.create_dataset(f"{trg}/epsilon", data=np.float64(eps))
            dst.attrs["units"] = "nm"
            dst = h5w.create_dataset(f"{trg}/min_samples", data=np.uint32(min_samples))
            dst = h5w.create_dataset(f"{trg}/labels", compression="gzip", compression_opts=1, data=np.asarray(db.labels_, np.int64))
            h5w.close()

            del trg_vxl_pos
            del db
