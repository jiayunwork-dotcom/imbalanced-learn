"""Demonstration of the exact per-class quotas of KMeansSMOTE and ADASYN.

Run with::

    python examples/over-sampling/demo_exact_quotas.py

The script builds small synthetic datasets that highlight the three failure
modes that were fixed:

1. KMeansSMOTE used to round the quota up cluster by cluster and over-generate
   (two eligible clusters sharing 3 rows -> 4 rows);
2. KMeansSMOTE used to crash with ``NaN`` when the class points of every
   eligible cluster coincided (zero sparsity);
3. ADASYN used to round every anchor down and report that no sample could be
   generated even though one row was requested and neighbours were available.

It then checks that both samplers follow the same integer plan for the same
ordered weights, requested total and integer seed, and that the per-source
quotas are actually honoured by the generated rows.
"""

import numpy as np
from sklearn.base import BaseEstimator

from imblearn.over_sampling import ADASYN, KMeansSMOTE
from imblearn.over_sampling._planning import allocate_integer_quotas

# ``imbalanced-learn`` imports require scikit-learn; matplotlib is not used here
# so that the script runs headless.


class SplitOnFirstFeature(BaseEstimator):
    """Deterministic clusterer used to keep the demonstration reproducible."""

    def __init__(self, n_clusters=2, threshold=5.0):
        self.n_clusters = n_clusters
        self.threshold = threshold

    def fit(self, X, y=None):
        return self

    def fit_predict(self, X, y=None):
        return self.predict(X)

    def predict(self, X):
        X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
        return (X[:, 0] >= self.threshold).astype(int)


def demo_two_clusters_three_rows():
    print("=" * 72)
    print("1. KMeansSMOTE: two eligible clusters, 3 rows to generate")
    left = np.array([[0.0, 0.0], [0.0, 1.0], [0.0, 2.0], [0.0, 3.0], [0.0, 4.0]])
    right = np.array([[10.0, 0.0], [10.0, 1.0], [10.0, 2.0], [10.0, 3.0], [10.0, 4.0]])
    majority = np.array([[1.0, -1.0], [11.0, -1.0]])
    X = np.vstack([left, right, majority])
    y = np.array([1] * 10 + [0] * 2)  # 10 class-1 rows, ask for 13

    sampler = KMeansSMOTE(
        kmeans_estimator=SplitOnFirstFeature(),
        random_state=0,
        k_neighbors=1,
        cluster_balance_threshold=0.0,
        sampling_strategy={1: 13},
    )
    X_res, y_res = sampler.fit_resample(X, y)
    print("  class counts after :", np.bincount(y_res).tolist())
    print("  per-cluster quotas :", sampler._source_quotas_[1].tolist())
    print("  expected sum       : 3 (old ceil-per-cluster rule gave 4)")


def demo_coincident_points():
    print("=" * 72)
    print("2. KMeansSMOTE: every eligible cluster is degenerate (zero sparsity)")
    X = np.array(
        [
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [100.0, 0.0],
            [100.0, 0.0],
            [100.0, 0.0],
            [100.0, 0.0],
            [100.0, 1.0],
            [100.0, 1.0],
        ]
    )
    y = np.array([1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0])
    sampler = KMeansSMOTE(
        kmeans_estimator=SplitOnFirstFeature(threshold=50.0),
        random_state=0,
        k_neighbors=1,
        cluster_balance_threshold=0.0,
        sampling_strategy={1: 10},
    )
    X_res, y_res = sampler.fit_resample(X, y)
    print("  class counts after :", np.bincount(y_res).tolist())
    print("  per-cluster quotas :", sampler._source_quotas_[1].tolist())
    print("  generated rows      :", X_res[X.shape[0] :].tolist())
    print("  (uniform rule: no NaN, coincident points reproduce their value)")


def demo_adasyn_single_slot():
    print("=" * 72)
    print("3. ADASYN: several equally hard anchors, only 1 row requested")
    X = np.array(
        [[0.0, 0.0], [10.0, 10.0], [20.0, 20.0]]
        + [
            [0.0, 1.0],
            [1.0, 0.0],
            [10.0, 11.0],
            [11.0, 10.0],
            [20.0, 21.0],
            [21.0, 20.0],
        ]
    )
    y = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0])
    sampler = ADASYN(n_neighbors=2, random_state=0, sampling_strategy={1: 4})
    X_res, y_res = sampler.fit_resample(X, y)
    print("  class counts after :", np.bincount(y_res).tolist())
    print("  per-anchor quotas  :", sampler._source_quotas_[1].tolist())
    print("  (old np.rint gave [0, 0, 0] and raised 'No samples ...')")


def demo_shared_plan():
    print("=" * 72)
    print("4. Shared allocation rule for both samplers")
    weights = np.array([1.0, 1.0, 1.0, 1.0])
    print("  equal weights, 2 rows to generate (fractional parts are tied):")
    for seed in (0, 1, 2, 3):
        quotas_km = KMeansSMOTE(random_state=seed)._allocate_source_quotas(weights, 2)
        quotas_ada = ADASYN(random_state=seed)._allocate_source_quotas(weights, 2)
        print(
            f"    seed={seed}: KMeansSMOTE={quotas_km.tolist()} "
            f"ADASYN={quotas_ada.tolist()}"
        )
    print("  -> same plan for both, and the winners change with the seed;")
    print("     a tie-break by position would always pick the first two.")
    weights = np.array([0.5, 1.0, 1.5, 2.0])
    ideal = 7 * weights / weights.sum()
    quotas = allocate_integer_quotas(weights, 7, random_state=0)
    print("  unequal weights     :", weights.tolist())
    print("  ideal shares        :", np.round(ideal, 3).tolist())
    print("  quotas              :", quotas.tolist())
    print("  |quota - ideal| < 1 :", bool(np.all(np.abs(quotas - ideal) < 1)))


def demo_multi_class():
    print("=" * 72)
    print("5. Multi-class: each targeted class gets its exact count")
    X = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [0.0, 2.0],
            [0.0, 3.0],
            [0.0, 4.0],
            [0.0, 5.0],
            [10.0, 0.0],
            [10.0, 1.0],
            [10.0, 2.0],
            [10.0, 3.0],
            [10.0, 4.0],
            [10.0, 5.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [1.0, 2.0],
            [11.0, 0.0],
            [11.0, 1.0],
            [11.0, 2.0],
            [50.0, 0.0],
            [52.0, 0.0],
            [54.0, 0.0],
        ]
    )
    y = np.array([0] * 6 + [1] * 6 + [2] * 9)
    sampler = ADASYN(n_neighbors=3, random_state=0, sampling_strategy={0: 10, 1: 10})
    X_res, y_res = sampler.fit_resample(X, y)
    print("  before:", np.bincount(y).tolist(), "after:", np.bincount(y_res).tolist())
    print("  (class 2 is not targeted and stays at 9)")


if __name__ == "__main__":
    demo_two_clusters_three_rows()
    demo_coincident_points()
    demo_adasyn_single_slot()
    demo_shared_plan()
    demo_multi_class()
