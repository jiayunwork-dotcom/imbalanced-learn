"""
====================================================
Weight-conserving resampling in a training pipeline
====================================================

This example shows how the weight-conserving resampling keeps the books of
every original observation: duplicated and synthetic rows share the weight of
the observations they derive from, cleaning steps record what they drop, and
the pipeline hands the conserved weights to the final estimator.

"""

# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
# License: MIT

import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from imblearn.combine import SMOTEENN
from imblearn.pipeline import Pipeline

print(__doc__)

##############################################################################
# Toy data generation
##############################################################################

##############################################################################
# We generate an imbalanced dataset in which each observation carries a
# weight, e.g. the number of days a customer stayed subscribed.

X, y = make_classification(
    n_classes=2,
    class_sep=1.5,
    weights=[0.1, 0.9],
    n_informative=3,
    n_redundant=1,
    flip_y=0,
    n_features=10,
    n_clusters_per_class=1,
    n_samples=2_000,
    random_state=10,
)
rng = np.random.RandomState(42)
sample_weight = rng.rand(X.shape[0]) * 10

X_train, X_test, y_train, y_test, w_train, _ = train_test_split(
    X, y, sample_weight, test_size=0.3, stratify=y, random_state=0
)

##############################################################################
# Resampling without conservation inflates the mass
##############################################################################

##############################################################################
# Duplicating the minority class tenfold also multiplies its total weight by
# ten: the class balance of the training set changes as if the minority were
# ten times more prevalent.

from imblearn.over_sampling import SMOTE

X_res, y_res = SMOTE(random_state=0).fit_resample(X_train, y_train)
minority = np.argmin(np.bincount(y_train.astype(int)))
inflation = np.sum(y_res == minority) / np.sum(y_train == minority)
print(
    f"Rows before/after SMOTE: {X_train.shape[0]} / {X_res.shape[0]}\n"
    "Mass of the minority class before/after naive duplication: "
    f"{w_train[y_train == minority].sum():.1f} / "
    f"{w_train[y_train == minority].sum() * inflation:.1f}"
)

##############################################################################
# Weight-conserving resampling keeps the books
##############################################################################

##############################################################################
# With ``fit_resample_weighted``, the mass of each original observation is
# split over the rows it funds: its own kept row, and the synthetic samples
# generated from it. The total mass of the class is conserved.

result = SMOTEENN(random_state=0).fit_resample_weighted(
    X_train, y_train, sample_weight=w_train
)
result.check_conserved(w_train)
print(
    f"Mass before: {w_train.sum():.1f}\n"
    f"Mass after (train): {result.sample_weight.sum():.1f}\n"
    f"Mass dropped by ENN: {result.dropped_weight.sum():.1f}"
)

##############################################################################
# The same bookkeeping is available inside a pipeline
##############################################################################

##############################################################################
# With ``weight_resampling='conserve'``, ``fit(X, y, sample_weight=...)``
# carries the weights through the samplers and hands the weights of the
# final training rows to the final estimator. The reconciliation is stored
# in ``resampling_audit_``.

pipe = Pipeline(
    [("smoteenn", SMOTEENN(random_state=0)), ("clf", LogisticRegression())],
    weight_resampling="conserve",
)
pipe.fit(X_train, y_train, sample_weight=w_train)
audit = pipe.resampling_audit_
print(
    f"Mass handed to the classifier: {audit.sample_weight_out.sum():.1f}\n"
    f"Mass dropped along the way: {audit.dropped_weight.sum():.1f}\n"
    f"Books balance: {audit.is_conserved()}"
)

##############################################################################
# The audit is rooted at the original observations: for each of them, the
# mass it handed to the final training rows plus its dropped mass equals its
# input weight.

original_mass = audit.contributions.input_sums() + audit.dropped_weight
print(f"Max per-observation imbalance: {np.abs(original_mass - w_train).max():.2e}")
print(f"Test score: {pipe.score(X_test, y_test):.3f}")
