import os
from compositionspace.ml_models import get_model
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
import h5py
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from tqdm.notebook import tqdm
from compositionspace.get_gitrepo_commit import get_repo_last_commit
from pyevtk.hl import pointsToVTK
# from pyevtk.hl import gridToVTK  # pointsToVTKAsTIN
import yaml
import pyvista as pv
from compositionspace.utils import EPSILON


class ProcessSegmentation():
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
        self.n_itypes = 0
        self.composition_matrix = None
        self.X_train = None

    def get_composition_matrix(self):
        self.composition_matrix = None
        with h5py.File(self.config["results_file_path"], "r") as h5r:
            src = f"/entry{self.config['entry_id']}/voxelization"
            total_weights = h5r[f"{src}/total"][:]
            self.n_itypes = 73  # TODO::fish from h5r

            self.composition_matrix = np.zeros([np.shape(total_weights)[0], self.n_itypes], np.float64)
            for ityp in np.arange(0, self.n_itypes):
                ityp_weights = h5r[f"{src}/ion{ityp}/weight"][:]
                if np.shape(ityp_weights) == np.shape(total_weights):
                    self.composition_matrix[:, ityp] = np.divide(ityp_weights, total_weights, where= total_weights >= EPSILON)
                    self.composition_matrix[np.where(self.composition_matrix[:, ityp] < EPSILON), ityp] = 0.
                    self.composition_matrix[np.isnan(self.composition_matrix[:, ityp]), ityp] = 0.
                else:
                    raise ValueError(f"Length of iontype-specific and total weight arrays for ityp {ityp} needs to be the same!")

    def perform_pca_and_write_results(self):
        self.get_composition_matrix()

        self.X_train = None
        self.X_train = self.composition_matrix
        PCAObj = PCA(n_components = self.n_itypes)
        PCATrans = PCAObj.fit_transform(self.X_train)
        PCACumsumArr = np.cumsum(PCAObj.explained_variance_ratio_)

        with h5py.File(self.config["results_file_path"], "a") as h5w:
            trg = f"/entry{self.config['entry_id']}/segmentation"
            grp = h5w.create_group(trg)
            grp.attrs["NX_class"] = "NXobject"  # TODO
            trg = f"/entry{self.config['entry_id']}/segmentation/pca"
            grp = h5w.create_group(trg)
            grp.attrs["NX_class"] = "NXprocess"
            dst = h5w.create_dataset(f"{trg}/sequence_index", data=np.uint64(2))
            trg = f"/entry{self.config['entry_id']}/segmentation/pca/result"
            grp = h5w.create_group(trg)
            grp.attrs["NX_class"] = "NXdata"
            grp.attrs["axes"] = "axis_pca_dimension"
            grp.attrs["axis_pca_dimension"] = np.uint64(0)
            grp.attrs["signal"] = "axis_explained_variance"
            # further attributes, to render it a proper NeXus NXdata object
            axis_dim = np.asarray(np.linspace(0, self.n_itypes - 1, num=self.n_itypes, endpoint=True), np.uint32)
            dst = h5w.create_dataset(f"{trg}/axis_pca_dimension", compression="gzip", compression_opts=1, data=axis_dim)
            dst.attrs["long_name"] = "Dimension"
            axis_expl_var = np.asarray(PCACumsumArr, np.float64)
            dst = h5w.create_dataset(f"{trg}/axis_explained_variance", compression="gzip", compression_opts=1, data=axis_expl_var)
            dst.attrs["long_name"] = "Explained variance"

    def perform_bics_minimization_and_write_results(self):
        self.get_composition_matrix()

        with h5py.File(self.config["results_file_path"], "a") as h5w:
            trg = f"/entry{self.config['entry_id']}/segmentation/ic_opt"  # information criterion optimization (minimization)
            grp = h5w.create_group(trg)
            grp.attrs["NX_class"] = "NXprocess"
            dst = h5w.create_dataset(f"{trg}/sequence_index", data=np.uint64(3))

        # gm_scores = []
        aics = []
        bics = []  # y_pred are stored directly into the HDF5 file
        n_clusters_queue = list(range(1, self.config["bics_clusters"] + 1))
        identifier = 1
        for n_cluster in n_clusters_queue:
            # why does the following result look entirely different by orders of magnitude if you change range to np.arange and drop the list creation?
            # floating point versus integer numbers, this needs to be checked !!!
            # again !!! even though now we are using list and range again the result appear random!!!???
            # run sequentially first to assure
            print(f"GaussianMixture ML analysis with n_cluster {int(n_cluster)}")
            gm = GaussianMixture(n_components=int(n_cluster), verbose=0)
            gm.fit(self.X_train)
            y_pred = gm.predict(self.composition_matrix)
            # gm_scores.append(homogeneity_score(y, y_pred))
            aics.append(gm.aic(self.composition_matrix))
            bics.append(gm.bic(self.composition_matrix))

            with h5py.File(self.config["results_file_path"], "a") as h5w:
                trg = f"/entry{self.config['entry_id']}/segmentation/ic_opt/cluster_analysis{identifier}"
                grp = h5w.create_group(trg)
                grp.attrs["NX_class"] = "NXprocess"
                dst = h5w.create_dataset(f"{trg}/n_cluster", data=np.uint32(n_cluster))
                dst = h5w.create_dataset(f"{trg}/y_pred", compression="gzip", compression_opts=1, data=np.asarray(y_pred, np.uint32))
            identifier += 1
        # all clusters processed TODO: take advantage of trivial parallelism here

        with h5py.File(self.config["results_file_path"], "a") as h5w:
            trg = f"/entry{self.config['entry_id']}/segmentation/ic_opt/summary"
            grp = h5w.create_group(trg)
            grp.attrs["NX_class"] = "NXdata"
            grp.attrs["axes"] = "axis_dimension"
            grp.attrs["axis_dimension"] = np.uint64(0)
            # grp.attrs["signal"] = "axis_aic"  # Akaike information criterion
            grp.attrs["signal"] = "axis_bic"  # Bayes information criterion
            grp.attrs["auxiliary_signals"] = ["axis_aic"]
            dst = h5w.create_dataset(f"{trg}/title", data="Information criterion minimization")

            # further attributes to render it a proper NeXus NXdata object
            axis_dim = np.asarray(np.linspace(1, self.config["bics_clusters"], num=self.config["bics_clusters"], endpoint=True), np.uint32)
            dst = h5w.create_dataset(f"{trg}/axis_dimension", compression="gzip", compression_opts=1, data=axis_dim)
            dst.attrs["long_name"] = "Number of cluster"
            # dst.attrs["units"] = "1"
            axis_aic = np.asarray(aics, np.float64)
            dst = h5w.create_dataset(f"{trg}/axis_aic", compression="gzip", compression_opts=1, data=axis_aic)
            # dst.attrs["long_name"] = "Akaike information criterion"
            # dst.attrs["units"] = ""  # is NX_DIMENSIONLESS
            axis_bic = np.asarray(bics, np.float64)
            dst = h5w.create_dataset(f"{trg}/axis_bic", compression="gzip", compression_opts=1, data=axis_bic)
            dst.attrs["long_name"] = "Information criterion value"  # "Bayes information criterion"
            # dst.attrs["units"] = ""  # is NX_DIMENSIONLESS

    def run(self):
        self.perform_pca_and_write_results()
        self.perform_bics_minimization_and_write_results()

    # inspect version prior nexus-io feature branch was merged for generate_plots and plot3d
