import os

import flatdict as fd
import h5py
import numpy as np
import yaml
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

from compositionspace.get_gitrepo_commit import get_repo_last_commit
from compositionspace.utils import APT_UINT, PRNG_SEED, get_composition_matrix
from compositionspace.visualization import decorate_path_to_default_plot


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
                self.config = fd.FlatDict(yaml.safe_load(yml), delimiter="/")
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

    def perform_pca_and_write_results(self):
        """Perform PCA of n_chemical_class-dimensional correlation."""
        self.composition_matrix, self.n_chem_classes = get_composition_matrix(
            self.config["results_file_path"], self.config["entry_id"]
        )

        self.X_train = self.composition_matrix
        PCAObj = PCA(n_components=self.n_chem_classes)
        PCATrans = PCAObj.fit_transform(self.X_train)
        PCACumsumArr = np.cumsum(PCAObj.explained_variance_ratio_)

        h5w = h5py.File(self.config["results_file_path"], "a")
        trg = f"/entry{self.config['entry_id']}/segmentation"
        grp = h5w.create_group(trg)
        grp.attrs["NX_class"] = "NXprocess"
        trg = f"/entry{self.config['entry_id']}/segmentation/pca"
        grp = h5w.create_group(trg)
        grp.attrs["NX_class"] = "NXprocess"
        sequence_idx = 2
        if self.config["autophase/use"]:
            sequence_idx += 1
        dst = h5w.create_dataset(f"{trg}/sequence_index", data=np.uint64(sequence_idx))
        trg = f"/entry{self.config['entry_id']}/segmentation/pca/result"
        grp = h5w.create_group(trg)
        grp.attrs["NX_class"] = "NXdata"
        grp.attrs["axes"] = "axis_pca_dimension"
        grp.attrs["axis_pca_dimension_indices"] = np.uint64(0)
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
        h5w.close()

    def perform_bics_minimization_and_write_results(self):
        """Perform Gaussian mixture model supervised ML with (Bayesian) IC minimization."""
        self.composition_matrix, self.n_chem_classes = get_composition_matrix(
            self.config["results_file_path"], self.config["entry_id"]
        )

        h5w = h5py.File(self.config["results_file_path"], "a")
        trg = f"/entry{self.config['entry_id']}/segmentation/ic_opt"  # information criterion optimization (minimization)
        grp = h5w.create_group(trg)
        grp.attrs["NX_class"] = "NXprocess"
        sequence_idx = 3
        if self.config["autophase/use"]:
            sequence_idx += 1
        dst = h5w.create_dataset(f"{trg}/sequence_index", data=np.uint64(sequence_idx))
        h5w.close()

        aics = []
        bics = []
        n_clusters_queue = list(
            range(1, self.config["segmentation/n_max_ic_cluster"] + 1)
        )
        for n_bics_cluster in n_clusters_queue:
            X_train = None
            C_mod = None
            if self.config["autophase/use"]:
                print("Using results with automated phase assignment")
                with h5py.File(self.config["results_file_path"], "r") as h5r:
                    trg = f"/entry{self.config['entry_id']}/autophase/result/axis_feature_identifier"
                    if trg in h5r:
                        descending_indices = h5r[trg][:]
                        # print(descending_indices)
                        n_trunc = self.config["autophase/trunc_species"]
                        # print(n_trunc)
                        X_train = np.zeros(
                            (np.shape(self.composition_matrix)[0], n_trunc), np.float64
                        )
                        C_mod = np.zeros(
                            (np.shape(self.composition_matrix)[0], n_trunc), np.float64
                        )
                        for j, idx in enumerate(descending_indices[0:n_trunc]):
                            X_train[:, j] = self.composition_matrix[:, idx]
                            C_mod[:, j] = self.composition_matrix[:, idx]
                    else:
                        raise KeyError(
                            f"{trg} does not exist in {self.config["results_file_path"]} !"
                        )
            else:
                print("Using results without automated phase assignment")
                X_train = self.composition_matrix
                C_mod = self.composition_matrix
            print(f"np.shape(X_train) {np.shape(X_train)}")

            # why does the following result look entirely different by orders of magnitude if you change range to np.arange and drop the list creation?
            # floating point versus integer numbers, this needs to be checked !!!
            # again !!! even though now we are using list and range again the result appear random!!!???
            # run sequentially first to assure
            print(f"GaussianMixture ML analysis with n_cluster {int(n_bics_cluster)}")
            gm = GaussianMixture(
                n_components=int(n_bics_cluster), random_state=PRNG_SEED, verbose=0
            )
            gm.fit(X_train)
            y_pred = gm.predict(C_mod)
            # gm_scores.append(homogeneity_score(y, y_pred))
            aics.append(gm.aic(C_mod))
            bics.append(gm.bic(C_mod))

            h5w = h5py.File(self.config["results_file_path"], "a")
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
            h5w.close()

            del X_train
            del C_mod
        # all clusters processed TODO: take advantage of trivial parallelism here

        h5w = h5py.File(self.config["results_file_path"], "a")
        trg = f"/entry{self.config['entry_id']}/segmentation/ic_opt/result"
        grp = h5w.create_group(trg)
        grp.attrs["NX_class"] = "NXdata"
        grp.attrs["axes"] = "axis_dimension"
        grp.attrs["axis_dimension_indices"] = np.uint64(0)
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
                self.config["segmentation/n_max_ic_cluster"],
                num=self.config["segmentation/n_max_ic_cluster"],
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
        # make this result the NeXus default plot
        decorate_path_to_default_plot(
            h5w,
            f"/entry{self.config['entry_id']}/segmentation/ic_opt/result",
        )
        h5w.close()

    def run(self):
        """Run step 3 and 4 of the workflow."""
        self.perform_pca_and_write_results()
        self.perform_bics_minimization_and_write_results()

    # inspect version prior nexus-io feature branch was merged for generate_plots and plot3d
