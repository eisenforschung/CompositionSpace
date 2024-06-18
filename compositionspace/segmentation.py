import os
import h5py
import yaml
import numpy as np
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from compositionspace.get_gitrepo_commit import get_repo_last_commit
from compositionspace.utils import EPSILON, APT_UINT


class ProcessSegmentation:
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
                self.config = yaml.safe_load(yml)
        else:
            raise IOError(f"File {config_file_path} does not exist!")
        if os.path.exists(results_file_path):
            self.config["results_file_path"] = results_file_path
        else:
            raise IOError(f"File {results_file_path} does not exist!")
        self.config["entry_id"] = entry_id
        self.verbose = verbose
        self.version = get_repo_last_commit()
        self.n_chem_classes = 0
        self.composition_matrix = None
        self.X_train = None

    def get_composition_matrix(self):
        """Compute (n_ions, n_chemical_class) composition matrix from per-class counts."""
        self.composition_matrix = None
        with h5py.File(self.config["results_file_path"], "r") as h5r:
            src = f"/entry{self.config['entry_id']}/voxelization"
            self.n_chem_classes = sum(
                1 for grpnm in h5r[f"{src}"] if grpnm.startswith("element")
            )
            print(f"Composition matrix has {self.n_chem_classes} chemical classes")

            total_cnts = np.asarray(h5r[f"{src}/counts"][:], np.float64)
            self.composition_matrix = np.zeros(
                [np.shape(total_cnts)[0], self.n_chem_classes + 1], np.float64
            )

            for grpnm in h5r[f"{src}"]:
                if grpnm.startswith("element"):
                    chem_class_idx = int(grpnm.replace("element", ""))
                    etyp_cnts = np.asarray(h5r[f"{src}/{grpnm}/counts"][:], np.float64)
                    if np.shape(etyp_cnts) == np.shape(total_cnts):
                        self.composition_matrix[:, chem_class_idx] = np.divide(
                            etyp_cnts,
                            total_cnts,
                            out=self.composition_matrix[:, chem_class_idx],
                            where=total_cnts >= (1.0 - EPSILON),
                        )
                    else:
                        raise ValueError(
                            f"Groupname {grpnm}, length of counts array for chemical class {chem_class_idx} needs to be the same as of counts!"
                        )

    def perform_pca_and_write_results(self):
        """Perform PCA of n_chemical_class-dimensional correlation."""
        self.get_composition_matrix()
        # TODO:export composition matrix here

        self.X_train = None
        self.X_train = self.composition_matrix
        PCAObj = PCA(n_components=self.n_chem_classes)
        PCATrans = PCAObj.fit_transform(self.X_train)
        PCACumsumArr = np.cumsum(PCAObj.explained_variance_ratio_)

        with h5py.File(self.config["results_file_path"], "a") as h5w:
            trg = f"/entry{self.config['entry_id']}/segmentation"
            grp = h5w.create_group(trg)
            grp.attrs["NX_class"] = "NXprocess"
            trg = f"/entry{self.config['entry_id']}/segmentation/pca"
            grp = h5w.create_group(trg)
            grp.attrs["NX_class"] = "NXprocess"
            dst = h5w.create_dataset(f"{trg}/sequence_index", data=np.uint32(2))
            trg = f"/entry{self.config['entry_id']}/segmentation/pca/result"
            grp = h5w.create_group(trg)
            grp.attrs["NX_class"] = "NXdata"
            grp.attrs["axes"] = "axis_pca_dimension"
            grp.attrs["axis_pca_dimension"] = np.uint32(0)
            grp.attrs["signal"] = "axis_explained_variance"
            # further attributes, to render it a proper NeXus NXdata object
            axis_dim = np.asarray(
                np.linspace(
                    0, self.n_chem_classes - 1, num=self.n_chem_classes, endpoint=True
                ),
                np.uint32,
            )
            dst = h5w.create_dataset(
                f"{trg}/axis_pca_dimension",
                compression="gzip",
                compression_opts=1,
                data=axis_dim,
            )
            dst.attrs["long_name"] = "Dimension"
            axis_expl_var = np.asarray(PCACumsumArr, np.float64)
            dst = h5w.create_dataset(
                f"{trg}/axis_explained_variance",
                compression="gzip",
                compression_opts=1,
                data=axis_expl_var,
            )
            dst.attrs["long_name"] = "Explained variance"

    def perform_bics_minimization_and_write_results(self):
        """Perform Gaussian mixture model supervised ML with (Bayesian) IC minimization."""
        self.get_composition_matrix()

        self.X_train = None
        self.X_train = self.composition_matrix

        with h5py.File(self.config["results_file_path"], "a") as h5w:
            trg = f"/entry{self.config['entry_id']}/segmentation/ic_opt"  # information criterion optimization (minimization)
            grp = h5w.create_group(trg)
            grp.attrs["NX_class"] = "NXprocess"
            dst = h5w.create_dataset(f"{trg}/sequence_index", data=np.uint32(3))

        # gm_scores = []
        aics = []
        bics = []  # y_pred are stored directly into the HDF5 file
        n_clusters_queue = list(range(1, self.config["n_max_ic_cluster"] + 1))
        for n_bics_cluster in n_clusters_queue:
            # why does the following result look entirely different by orders of magnitude if you change range to np.arange and drop the list creation?
            # floating point versus integer numbers, this needs to be checked !!!
            # again !!! even though now we are using list and range again the result appear random!!!???
            # run sequentially first to assure
            print(f"GaussianMixture ML analysis with n_cluster {int(n_bics_cluster)}")
            gm = GaussianMixture(n_components=int(n_bics_cluster), verbose=0)
            gm.fit(self.X_train)
            y_pred = gm.predict(self.composition_matrix)
            # gm_scores.append(homogeneity_score(y, y_pred))
            aics.append(gm.aic(self.composition_matrix))
            bics.append(gm.bic(self.composition_matrix))

            with h5py.File(self.config["results_file_path"], "a") as h5w:
                trg = f"/entry{self.config['entry_id']}/segmentation/ic_opt/cluster_analysis{n_bics_cluster - 1}"
                grp = h5w.create_group(trg)
                grp.attrs["NX_class"] = "NXprocess"
                dst = h5w.create_dataset(
                    f"{trg}/n_ic_cluster", data=np.uint64(n_bics_cluster)
                )
                dst = h5w.create_dataset(
                    f"{trg}/y_pred",
                    compression="gzip",
                    compression_opts=1,
                    data=np.asarray(y_pred, APT_UINT),
                )
        # all clusters processed TODO: take advantage of trivial parallelism here

        with h5py.File(self.config["results_file_path"], "a") as h5w:
            trg = f"/entry{self.config['entry_id']}/segmentation/ic_opt/summary"
            grp = h5w.create_group(trg)
            grp.attrs["NX_class"] = "NXdata"
            grp.attrs["axes"] = "axis_dimension"
            grp.attrs["axis_dimension"] = np.uint32(0)
            # grp.attrs["signal"] = "axis_aic"  # Akaike information criterion
            grp.attrs["signal"] = "axis_bic"  # Bayes information criterion
            grp.attrs["auxiliary_signals"] = ["axis_aic"]
            dst = h5w.create_dataset(
                f"{trg}/title", data="Information criterion minimization"
            )

            # further attributes to render it a proper NeXus NXdata object
            axis_dim = np.asarray(
                np.linspace(
                    1,
                    self.config["n_max_ic_cluster"],
                    num=self.config["n_max_ic_cluster"],
                    endpoint=True,
                ),
                APT_UINT,
            )
            dst = h5w.create_dataset(
                f"{trg}/axis_dimension",
                compression="gzip",
                compression_opts=1,
                data=axis_dim,
            )
            dst.attrs["long_name"] = "Number of cluster"
            dst = h5w.create_dataset(
                f"{trg}/axis_aic",
                compression="gzip",
                compression_opts=1,
                data=np.asarray(aics, np.float64),
            )
            # dst.attrs["long_name"] = "Akaike information criterion", NX_DIMENSIONLESS
            dst = h5w.create_dataset(
                f"{trg}/axis_bic",
                compression="gzip",
                compression_opts=1,
                data=np.asarray(bics, np.float64),
            )
            dst.attrs["long_name"] = (
                "Information criterion value"  # "Bayes information criterion", NX_DIMENSIONLESS
            )

    def run(self):
        """Run step 2 and 3 of the workflow."""
        self.perform_pca_and_write_results()
        self.perform_bics_minimization_and_write_results()

    # inspect version prior nexus-io feature branch was merged for generate_plots and plot3d
