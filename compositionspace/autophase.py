import os
import yaml
import h5py
import numpy as np
import flatdict as fd
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import RandomForestClassifier
from compositionspace.get_gitrepo_commit import get_repo_last_commit
from compositionspace.utils import APT_UINT, get_composition_matrix


class ProcessAutomatedPhaseAssignment:
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

    def auto_phase_assign(self):
        self.composition_matrix, self.n_chem_classes = get_composition_matrix(
            self.config["results_file_path"], self.config["entry_id"]
        )
        X = self.composition_matrix

        gm = GaussianMixture(
            n_components=int(self.config["autophase/initial_guess"]), verbose=0
        )
        gm.fit(X)
        y_pred = gm.predict(X)

        """
        Ratios = pd.DataFrame(data=X.values, columns=Chem_list)
        # Replace this with your actual dataset loading code
        X_ = X.values
        y = y_pred
        """

        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X, y_pred)

        feature_importances = rf.feature_importances_
        del rf

        descending_importances = []
        sorted_indices = feature_importances.argsort()[::-1]
        # given that we have element0 but leave it unpopulated sorted_indices resolve element<<identifier>> !
        if self.verbose:
            print(f"sorted_indices {sorted_indices}")
            print(f"sorted_index, feature_importance[sorted_index]")
            for idx in sorted_indices:
                descending_importances.append(feature_importances[idx])
                print(f"{idx}, {feature_importances[idx]}")
        del feature_importances

        with h5py.File(self.config["results_file_path"], "a") as h5w:
            trg = f"/entry{self.config['entry_id']}/autophase"
            grp = h5w.create_group(trg)
            grp.attrs["NX_class"] = "NXprocess"
            dst = h5w.create_dataset(f"{trg}/sequence_index", data=np.uint32(2))
            trg = f"/entry{self.config['entry_id']}/autophase/result"
            grp = h5w.create_group(trg)
            grp.attrs["NX_class"] = "NXdata"
            grp.attrs["axes"] = "axis_feature_identifier"
            grp.attrs["axis_feature_identifier"] = np.uint32(0)
            grp.attrs["signal"] = "axis_feature_importance"
            # further attributes, to render it a proper NeXus NXdata object
            dst = h5w.create_dataset(
                f"{trg}/axis_feature_identifier",
                compression="gzip",
                compression_opts=1,
                data=np.asarray(sorted_indices, APT_UINT),
            )
            dst.attrs["long_name"] = "Element identifier"
            dst = h5w.create_dataset(
                f"{trg}/axis_feature_importance",
                compression="gzip",
                compression_opts=1,
                data=np.asarray(descending_importances, np.float64),
            )
            dst.attrs["long_name"] = "Relative feature importance"

        """
        # BIC analysis on modified compositions
        if modified_comp_analysis == True:

            #n_trunc_spec = 2
            X_modified = X.values[:, sorted_indices][:,0:n_trunc_spec]
            gm_scores=[]
            aics=[]
            bics=[]

            n_clusters=list(range(1,11))
            for n_cluster in tqdm(n_clusters):
                gm = GaussianMixture(n_components=n_cluster,verbose=0)
                gm.fit(X_modified)
                y_pred=gm.predict(X_modified)
                #gm_scores.append(homogeneity_score(y,y_pred))
                aics.append(gm.aic(X_modified))
                bics.append(gm.bic(X_modified))

            plt.plot(n_clusters, bics, "-o",label="BIC")

        return sorted_indices
        """

    def run(self):
        """Run step 2 of the workflow."""
        self.auto_phase_assign()
