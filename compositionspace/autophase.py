import os

import flatdict as fd
import h5py
import numpy as np
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.mixture import GaussianMixture

from compositionspace.get_gitrepo_commit import get_repo_last_commit
from compositionspace.utils import APT_UINT, PRNG_SEED, get_composition_matrix


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

    def automated_phase_assignment(self):
        """Run random forest to rank elements by feature importance to guide dimensional reduction of segmentation."""
        self.composition_matrix, self.n_chem_classes = get_composition_matrix(
            self.config["results_file_path"], self.config["entry_id"]
        )
        X = self.composition_matrix

        # https://stackoverflow.com/questions/76096345/gaussian-mixture-model-is-inconsistent-in-its-results-using-sklearn-library
        # Alaukik mind the above-mentioned discussion, here I just use your prototypic implementation which might not be ideal!
        gm = GaussianMixture(
            n_components=int(self.config["autophase/initial_guess"]),
            random_state=PRNG_SEED,
            verbose=0,
        )
        gm.fit(X)
        y_pred = gm.predict(X)

        # https://stackoverflow.com/questions/70535224/how-to-reproduce-a-randomforestclassifier-when-random-state-uses-randomstate
        # Alaukik mind the above-mentioned discussion, here I just use your prototypic implementation which might not be ideal!
        rf = RandomForestClassifier(n_estimators=100, random_state=PRNG_SEED, n_jobs=-1)
        rf.fit(X, y_pred)

        # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
        # Warning: impurity-based feature importances can be misleading for high cardinality features (many unique values).
        # See sklearn.inspection.permutation_importance as an alternative.
        # Alaukik mind the above-mentioned discussion, here I just use your prototypic implementation which might not be ideal!
        # currently Gini importance
        feature_importances = rf.feature_importances_
        del rf

        descending_importances = []
        sorted_indices = feature_importances.argsort()[::-1]
        # given that we have element0 but leave it unpopulated sorted_indices resolve element<<identifier>> !
        if self.verbose:
            print(f"sorted_indices {sorted_indices} in decreasing feature importance")
            print(f"sorted_index, feature_importance[sorted_index]")
            for idx in sorted_indices:
                descending_importances.append(feature_importances[idx])
                print(f"{idx}, {feature_importances[idx]}")
        del feature_importances

        h5w = h5py.File(self.config["results_file_path"], "a")
        trg = f"/entry{self.config['entry_id']}/autophase"
        grp = h5w.create_group(trg)
        grp.attrs["NX_class"] = "NXprocess"
        dst = h5w.create_dataset(f"{trg}/sequence_index", data=np.uint64(2))
        trg = f"/entry{self.config['entry_id']}/autophase/result"
        grp = h5w.create_group(trg)
        grp.attrs["NX_class"] = "NXdata"
        grp.attrs["axes"] = "axis_feature_identifier"
        grp.attrs["axis_feature_identifier_indices"] = np.uint64(0)
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
        h5w.close()

    def run(self):
        """Run step 2 of the workflow."""
        if self.config["autophase/use"]:
            self.automated_phase_assignment()
