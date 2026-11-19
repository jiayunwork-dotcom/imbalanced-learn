"""Tests for the exact per-class quotas of KMeansSMOTE and ADASYN.

These tests pin the behaviour requested for the two samplers:

* the number of rows added to a class is exactly the number of additional
  rows parsed from ``sampling_strategy_`` (original rows are kept, in their
  original order);
* the integer plan over the eligible sources is the shared one, honoured by
  the generation itself rather than by trimming extra rows afterwards;
* the two samplers agree on the per-source plan for the same ordered weights,
  the same requested total and the same integer seed;
* ties and degenerate (zero-sparsity) sources are handled by a documented,
  reproducible rule;
* dense and CSR inputs give the same rows;
* a class with a zero quota is left untouched.
"""

import numpy as np
import pytest
from scipy import sparse
from sklearn.base import BaseEstimator
from sklearn.utils._testing import assert_allclose, assert_array_equal

from imblearn.over_sampling import ADASYN, KMeansSMOTE
from imblearn.over_sampling._planning import allocate_integer_quotas
from imblearn.pipeline import Pipeline


class ThresholdClusterer(BaseEstimator):
    """Cluster rows by position on the first feature.

    Deterministic and independent of scikit-learn's k-means implementation so
    that the tests exercise the quota logic rather than cluster randomness.
    """

    def __init__(self, n_clusters=2, threshold=50.0):
        self.n_clusters = n_clusters
        self.threshold = threshold

    def fit(self, X, y=None):
        return self

    def fit_predict(self, X, y=None):
        return self.predict(X)

    def predict(self, X):
        X = X.toarray() if sparse.issparse(X) else np.asarray(X)
        return (X[:, 0] >= self.threshold).astype(int)


class ThreeClusterer(BaseEstimator):
    """Three deterministic clusters delimited on the first feature."""

    def __init__(self, n_clusters=3, boundaries=(5.0, 15.0)):
        self.n_clusters = n_clusters
        self.boundaries = boundaries

    def fit(self, X, y=None):
        return self

    def fit_predict(self, X, y=None):
        return self.predict(X)

    def predict(self, X):
        X = X.toarray() if sparse.issparse(X) else np.asarray(X)
        return np.digitize(X[:, 0], self.boundaries)


def _two_separated_clusters():
    """Two eligible clusters, each with a handful of class-1 rows."""
    left = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0], [0.0, 2.0]])
    right = np.array([[10.0, 0.0], [10.0, 1.0], [11.0, 0.0], [11.0, 1.0], [10.0, 2.0]])
    majority = np.array([[0.0, -1.0], [1.0, -1.0], [10.0, -1.0], [11.0, -1.0]])
    X = np.vstack([left, right, majority])
    y = np.array([1] * 5 + [1] * 5 + [0] * 4)
    return X, y


def _coincident_clusters():
    """Class-1 rows coincide inside each cluster: zero intra-cluster sparsity."""
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
    return X, y


def _mixed_sparsity_clusters():
    """One spread cluster and one coincident cluster, both eligible."""
    X = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [0.0, 2.0],
            [0.0, 3.0],
            [0.0, 4.0],
            [0.0, 5.0],
            [1.0, 0.0],
            [2.0, 1.0],
            [3.0, 2.0],
            [4.0, 3.0],
            [5.0, 4.0],
            [6.0, 5.0],
            [100.0, 0.0],
            [100.0, 0.0],
            [100.0, 0.0],
            [100.0, 0.0],
            [100.0, 1.0],
            [101.0, 1.0],
            [102.0, 1.0],
            [103.0, 1.0],
        ]
    )
    y = np.array([1] * 6 + [0] * 6 + [1] * 4 + [0] * 4)
    return X, y


def _fit_kmeans_smote(X, y, n_samples, **kwargs):
    params = dict(
        kmeans_estimator=ThresholdClusterer(threshold=kwargs.pop("threshold", 50.0)),
        random_state=0,
        k_neighbors=1,
        cluster_balance_threshold=0.0,
        sampling_strategy={1: n_samples},
    )
    params.update(kwargs)
    sampler = KMeansSMOTE(**params)
    return sampler.fit_resample(X, y)


def _fit_adasyn(X, y, n_samples, **kwargs):
    params = dict(random_state=0, n_neighbors=2, sampling_strategy={1: n_samples})
    params.update(kwargs)
    sampler = ADASYN(**params)
    return sampler.fit_resample(X, y)


@pytest.mark.parametrize(
    "threshold, data_factory, original_count, requested",
    [
        (5.0, _two_separated_clusters, 10, 13),
        (50.0, _coincident_clusters, 8, 10),
        (50.0, _mixed_sparsity_clusters, 10, 12),
    ],
)
def test_kmeans_smote_exact_class_count(
    threshold, data_factory, original_count, requested
):
    X, y = data_factory()
    X_res, y_res = _fit_kmeans_smote(X, y, requested, threshold=threshold)
    counts = np.bincount(y_res)
    assert counts[1] == requested
    # the majority class is untouched
    assert counts[0] == np.bincount(y)[0]
    # original rows are kept first, in their original order
    assert_array_equal(X_res[: X.shape[0]], X)
    assert_array_equal(y_res[: y.shape[0]], y)
    assert X_res.shape[0] == X.shape[0] + (requested - original_count)


@pytest.mark.parametrize(
    "threshold, data_factory, original_count",
    [(5.0, _two_separated_clusters, 10), (50.0, _mixed_sparsity_clusters, 10)],
)
def test_kmeans_smote_regression_would_overshoot(
    threshold, data_factory, original_count
):
    """The old ceil-per-cluster rule generated more rows than requested.

    With two clusters sharing 3 rows, ``ceil(1.5)`` twice produces 4 rows.
    The exact plan must generate exactly 3.
    """
    X, y = data_factory()
    requested = original_count + 3
    X_res, y_res = _fit_kmeans_smote(X, y, requested, threshold=threshold)
    assert np.bincount(y_res)[1] == requested


def test_kmeans_smote_quotas_sum_to_requested_total():
    X, y = _two_separated_clusters()
    sampler = KMeansSMOTE(
        kmeans_estimator=ThresholdClusterer(threshold=5.0),
        random_state=0,
        k_neighbors=1,
        cluster_balance_threshold=0.0,
        sampling_strategy={1: 13},
    )
    sampler.fit_resample(X, y)
    quotas = sampler._source_quotas_[1]
    assert quotas.sum() == 3
    assert quotas.tolist() == allocate_integer_quotas([1.0, 1.0], 3, 0).tolist()


def test_kmeans_smote_generation_honours_quotas():
    """Each new row is an interpolation inside one eligible cluster.

    The clusterer splits the data at ``x == 50``; each generated row must lie
    in the bounding box of the cluster that was allocated it.
    """
    X, y = _two_separated_clusters()
    X_res, y_res = _fit_kmeans_smote(X, y, 13, threshold=5.0)
    new_rows = X_res[X.shape[0] :]
    assert new_rows.shape[0] == 3
    per_cluster = [int(np.sum(new_rows[:, 0] < 5)), int(np.sum(new_rows[:, 0] >= 5))]
    assert sum(per_cluster) == 3
    # disjoint bounding boxes: each point stays inside its own cluster
    left = new_rows[new_rows[:, 0] < 5]
    right = new_rows[new_rows[:, 0] >= 5]
    if left.size:
        assert left[:, 0].min() >= 0.0 and left[:, 0].max() <= 1.0
        assert left[:, 1].min() >= 0.0 and left[:, 1].max() <= 2.0
    if right.size:
        assert right[:, 0].min() >= 10.0 and right[:, 0].max() <= 11.0
        assert right[:, 1].min() >= 0.0 and right[:, 1].max() <= 2.0


def test_kmeans_smote_zero_sparsity_fallback_is_documented_uniform():
    """All eligible clusters degenerate: quota is split uniformly.

    No NaN is produced and the coincident points can regenerate their common
    value.
    """
    X, y = _coincident_clusters()
    sampler = KMeansSMOTE(
        kmeans_estimator=ThresholdClusterer(threshold=50.0),
        random_state=0,
        k_neighbors=1,
        cluster_balance_threshold=0.0,
        sampling_strategy={1: 10},
    )
    X_res, y_res = sampler.fit_resample(X, y)
    assert np.bincount(y_res)[1] == 10
    # both eligible clusters receive a share of the two requested rows
    assert sampler._source_quotas_[1].sum() == 2
    # interpolation between coincident points yields their common value
    new_rows = X_res[X.shape[0] :]
    assert_allclose(new_rows[:, 0], [0.0, 100.0], atol=1e-12)


def test_kmeans_smote_zero_sparsity_cluster_receives_nothing():
    """A degenerate cluster alongside a spread one gets a zero quota."""
    X, y = _mixed_sparsity_clusters()
    sampler = KMeansSMOTE(
        kmeans_estimator=ThresholdClusterer(threshold=50.0),
        random_state=0,
        k_neighbors=1,
        cluster_balance_threshold=0.0,
        sampling_strategy={1: 12},
    )
    X_res, y_res = sampler.fit_resample(X, y)
    assert np.bincount(y_res)[1] == 12
    quotas = sampler._source_quotas_[1]
    assert quotas[1] == 0
    assert quotas.sum() == 2
    # all generated rows come from the spread (left) cluster
    assert np.all(X_res[X.shape[0] :] < 50)


def test_kmeans_smote_no_eligible_cluster_still_fails():
    X, y = _two_separated_clusters()
    sampler = KMeansSMOTE(
        kmeans_estimator=ThresholdClusterer(threshold=50.0),
        random_state=0,
        cluster_balance_threshold=10.0,
    )
    with pytest.raises(RuntimeError, match="No clusters found"):
        sampler.fit_resample(X, y)


def test_kmeans_smote_zero_quota_class_untouched():
    X, y = _two_separated_clusters()
    # class 0 already has 4 rows; request the same amount -> zero new rows
    sampler = KMeansSMOTE(
        kmeans_estimator=ThresholdClusterer(threshold=50.0),
        random_state=0,
        k_neighbors=1,
        cluster_balance_threshold=0.0,
        sampling_strategy={0: 4, 1: 11},
    )
    X_res, y_res = sampler.fit_resample(X, y)
    assert np.bincount(y_res).tolist() == [4, 11]
    # the plan for class 0 is either absent or empty: no construction ran
    assert sampler._source_quotas_.get(0, np.zeros(1)).sum() == 0


def test_kmeans_smote_dense_and_csr_agree():
    X, y = _mixed_sparsity_clusters()
    X_res_dense, y_res_dense = _fit_kmeans_smote(X, y, 12)
    X_res_sparse, y_res_sparse = _fit_kmeans_smote(sparse.csr_matrix(X), y, 12)
    assert sparse.issparse(X_res_sparse)
    assert X_res_sparse.format == "csr"
    assert_allclose(X_res_sparse.toarray(), X_res_dense)
    assert_array_equal(y_res_sparse, y_res_dense)


def test_kmeans_smote_repeated_fit_reproducible():
    X, y = _two_separated_clusters()
    first = _fit_kmeans_smote(X, y, 13)
    second = _fit_kmeans_smote(X, y, 13)
    assert_allclose(first[0], second[0])
    assert_array_equal(first[1], second[1])


@pytest.mark.parametrize("n_neighbors", [2, 3])
def test_adasyn_exact_class_count(n_neighbors):
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
    sampler = ADASYN(n_neighbors=n_neighbors, random_state=0, sampling_strategy={1: 6})
    X_res, y_res = sampler.fit_resample(X, y)
    assert np.bincount(y_res).tolist() == [6, 6]
    assert_array_equal(X_res[: X.shape[0]], X)
    assert_array_equal(y_res[: y.shape[0]], y)


def test_adasyn_one_slot_among_equally_hard_anchors():
    """Several equally difficult anchors but only one row to generate.

    The old ``np.rint`` allocation could round every anchor down to zero and
    then raise "No samples will be generated".  Exactly one row is generated,
    by exactly one anchor, chosen reproducibly.
    """
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
    assert np.bincount(y_res).tolist() == [6, 4]
    quotas = sampler._source_quotas_[1]
    assert quotas.sum() == 1
    assert np.count_nonzero(quotas) == 1
    new_row = X_res[X.shape[0] :]
    assert new_row.shape == (1, 2)
    # the new row lies strictly on the segment between the allocated anchor
    # (a class-1 row) and another class-1 row, so it is inside the smallest
    # axis-aligned box of the three anchors
    anchors = X[:3]
    assert np.all(new_row >= anchors.min(axis=0) - 1e-12)
    assert np.all(new_row <= anchors.max(axis=0) + 1e-12)


def test_adasyn_one_slot_same_anchor_for_repeated_fit():
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
    quotas = []
    for _ in range(3):
        sampler = ADASYN(n_neighbors=2, random_state=0, sampling_strategy={1: 4})
        sampler.fit_resample(X, y)
        quotas.append(sampler._source_quotas_[1].tolist())
    assert quotas[0] == quotas[1] == quotas[2]


def test_adasyn_quotas_from_shared_rule():
    """Per-class quotas equal the shared largest-remainder plan.

    The difficulty ratios are recomputed independently from the sampler's
    neighbour definition, then the generation is checked to match the plan.
    """
    from sklearn.neighbors import NearestNeighbors

    from imblearn.over_sampling._planning import neighbors_excluding_self

    X = np.array(
        [[0.0, 0.0], [5.0, 5.0], [10.0, 10.0], [15.0, 15.0]]
        + [
            [0.0, 1.0],
            [1.0, 0.0],
            [0.0, -1.0],
            [5.0, 6.0],
            [10.0, 11.0],
            [11.0, 10.0],
            [15.0, 16.0],
            [16.0, 15.0],
            [15.0, 14.0],
        ]
    )
    y = np.array([1, 1, 1, 1] + [0] * 9)
    n_requested = 10
    sampler = ADASYN(n_neighbors=4, random_state=0, sampling_strategy={1: n_requested})
    X_res, y_res = sampler.fit_resample(X, y)

    indices = np.flatnonzero(y == 1)
    nn = NearestNeighbors(n_neighbors=4).fit(X)
    raw = nn.kneighbors(X[indices], return_distance=False)
    neighbors = neighbors_excluding_self(raw, indices, 3)
    ratio = np.mean(y[neighbors] != 1, axis=1)
    expected = allocate_integer_quotas(ratio, 6, random_state=0)
    assert_array_equal(sampler._source_quotas_[1], expected)
    assert np.bincount(y_res)[1] == n_requested


def test_adasyn_zero_ratio_anchors_get_no_quota():
    """An anchor whose neighbourhood is pure class 1 receives nothing."""
    # anchors 0 and 1 are surrounded only by class 1, anchor 2 sees class 0
    X = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [0.5, 0.0],
            [2.0, 1.0],
            [2.5, 0.5],
            [3.0, 0.0],
        ]
    )
    y = np.array([1, 1, 1, 1, 0, 0, 0])
    sampler = ADASYN(n_neighbors=3, random_state=0, sampling_strategy={1: 6})
    X_res, y_res = sampler.fit_resample(X, y)
    quotas = sampler._source_quotas_[1]
    assert quotas[0] == 0
    assert quotas[1] == 0
    assert quotas[2] == 2
    assert np.bincount(y_res)[1] == 6


def test_adasyn_no_minority_neighbour_still_fails():
    X = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [50.0, 50.0], [51.0, 51.0]])
    y = np.array([1, 1, 1, 0, 0])
    with pytest.raises(RuntimeError, match="Not any neigbours"):
        ADASYN(n_neighbors=2, random_state=0, sampling_strategy={1: 4}).fit_resample(
            X, y
        )


def test_adasyn_single_sample_class_is_reported():
    X = np.array([[0.0, 0.0], [5.0, 5.0], [6.0, 6.0]])
    y = np.array([1, 0, 0])
    with pytest.raises(ValueError, match="single sample"):
        ADASYN(n_neighbors=2, random_state=0).fit_resample(X, y)


def test_adasyn_single_neighbor_parameter():
    """``n_neighbors=1`` still yields at least one neighbour besides the
    anchor, so the difficulty ratio is defined."""
    X = np.array(
        [
            [0.0, 0.0],
            [10.0, 10.0],
            [20.0, 20.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [10.0, 11.0],
            [11.0, 10.0],
        ]
    )
    y = np.array([1, 1, 1, 0, 0, 0, 0])
    sampler = ADASYN(n_neighbors=1, random_state=0, sampling_strategy={1: 5})
    X_res, y_res = sampler.fit_resample(X, y)
    assert np.bincount(y_res).tolist() == [4, 5]


def test_adasyn_identical_rows_are_neighbours():
    """A row with identical features is a valid neighbour; only the anchor's
    own position is excluded."""
    # Two class-1 rows share the feature vector (0, 0); a third class-1 row is
    # at (10, 10).  All class-1 rows must be usable as neighbours: were they
    # de-duplicated by feature value, the two rows at (0, 0) would see each
    # other as "self" and the anchor could no longer interpolate to (0, 0).
    X = np.array(
        [[0.0, 0.0], [0.0, 0.0], [10.0, 10.0], [0.0, 1.0], [1.0, 0.0], [10.0, 11.0]]
    )
    y = np.array([1, 1, 1, 0, 0, 0])
    sampler = ADASYN(n_neighbors=2, random_state=2, sampling_strategy={1: 6})
    X_res, y_res = sampler.fit_resample(X, y)
    assert np.bincount(y_res).tolist() == [3, 6]
    quotas = sampler._source_quotas_[1]
    assert quotas.sum() == 3
    new_rows = X_res[X.shape[0] :]
    # at least one generated row is exactly (0, 0): it results from
    # interpolating one of the two identical rows with the other, which is
    # only possible if identical rows count as distinct neighbours
    assert np.any(np.all(np.isclose(new_rows, [0.0, 0.0]), axis=1))


def test_zero_new_rows_leaves_data_untouched():
    """Asking for the current class count returns the data unchanged."""
    X, y = _two_separated_clusters()
    for sampler in (
        ADASYN(n_neighbors=2, random_state=0, sampling_strategy={1: 10}),
        KMeansSMOTE(
            kmeans_estimator=ThresholdClusterer(threshold=5.0),
            random_state=0,
            k_neighbors=1,
            cluster_balance_threshold=0.0,
            sampling_strategy={1: 10},
        ),
    ):
        X_res, y_res = sampler.fit_resample(X, y)
        assert_array_equal(X_res, X)
        assert_array_equal(y_res, y)


def test_six_rows_to_nine_adds_exactly_three():
    """The motivating example: 6 rows in a class, the dict asks for 9."""
    X = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [0.0, 2.0],
            [0.0, 3.0],
            [0.0, 4.0],
            [0.0, 5.0],
            [0.6, 0.0],
            [0.6, 1.0],
            [0.6, 2.0],
            [0.6, 3.0],
            [0.6, 4.0],
            [0.6, 5.0],
            [1.2, 0.0],
            [1.2, 1.0],
            [1.2, 2.0],
            [1.2, 3.0],
        ]
    )
    y = np.array([1] * 6 + [0] * 10)
    sampler = ADASYN(n_neighbors=3, random_state=0, sampling_strategy={1: 9})
    X_res, y_res = sampler.fit_resample(X, y)
    assert np.bincount(y_res).tolist() == [10, 9]
    assert_array_equal(X_res[: X.shape[0]], X)
    assert_array_equal(y_res[: y.shape[0]], y)
    assert X_res.shape[0] == 19


def test_adasyn_dense_and_csr_agree():
    rng = np.random.RandomState(0)
    X = rng.randn(40, 3)
    y = np.array([0] * 30 + [1] * 10)
    dense = ADASYN(n_neighbors=3, random_state=0, sampling_strategy={1: 14})
    X_dense, y_dense = dense.fit_resample(X, y)

    sparse_sampler = ADASYN(n_neighbors=3, random_state=0, sampling_strategy={1: 14})
    X_sparse, y_sparse = sparse_sampler.fit_resample(sparse.csr_matrix(X), y)
    assert sparse.issparse(X_sparse)
    assert X_sparse.format == "csr"
    assert_allclose(X_sparse.toarray(), X_dense)
    assert_array_equal(y_sparse, y_dense)


def test_adasyn_repeated_fit_reproducible():
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

    def run():
        sampler = ADASYN(n_neighbors=2, random_state=7, sampling_strategy={1: 6})
        return sampler.fit_resample(X, y)

    first = run()
    second = run()
    assert_allclose(first[0], second[0])
    assert_array_equal(first[1], second[1])


def _three_equal_clusters():
    """Three clusters with identical geometry: equal sparsity weights."""
    base = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0], [0.0, 2.0]])
    majority = np.array([[0.0, -1.0], [1.0, -1.0]])
    X = np.vstack(
        [np.vstack([base + off, majority + off]) for off in (0.0, 10.0, 20.0)]
    )
    y = np.array([1] * 5 + [0] * 2 + [1] * 5 + [0] * 2 + [1] * 5 + [0] * 2)
    return X, y


def _three_equal_anchors():
    """Three anchors with the same difficulty ratio (1 of 2 neighbours)."""
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
    return X, y


def test_both_samplers_share_the_same_per_source_plan():
    """Same ordered weights, same total, same integer seed -> same quotas.

    Both samplers are run for real on data where their own weighting rule
    yields three *equal* positive weights.  With 4 rows to generate the shared
    largest-remainder plan is ``[1, 1, 2]`` for both, and each sampler must
    generate exactly that many rows from its sources.
    """
    X_km, y_km = _three_equal_clusters()
    km = KMeansSMOTE(
        kmeans_estimator=ThreeClusterer(),
        random_state=0,
        k_neighbors=1,
        cluster_balance_threshold=0.0,
        sampling_strategy={1: 19},
    )
    X_km_res, y_km_res = km.fit_resample(X_km, y_km)

    X_ada, y_ada = _three_equal_anchors()
    ada = ADASYN(n_neighbors=2, random_state=0, sampling_strategy={1: 7})
    X_ada_res, y_ada_res = ada.fit_resample(X_ada, y_ada)

    expected = allocate_integer_quotas([1.0, 1.0, 1.0], 4, random_state=0)
    assert expected.tolist() == [1, 1, 2]
    assert_array_equal(km._source_quotas_[1], expected)
    assert_array_equal(ada._source_quotas_[1], expected)
    assert np.bincount(y_km_res)[1] == 19
    assert np.bincount(y_ada_res)[1] == 7

    # the generated rows really are distributed as planned: KMeansSMOTE puts
    # each row inside its own cluster's bounding box
    new_km = X_km_res[X_km.shape[0] :]
    per_cluster = np.bincount(np.digitize(new_km[:, 0], (5.0, 15.0)), minlength=3)
    assert per_cluster.tolist() == expected.tolist()


def test_kmeans_smote_unequal_weights_are_respected():
    """A denser cluster gets fewer rows than a sparser one."""
    # left: spread class-1 points; right: tight class-1 points.
    left = np.array([[0.0, 0.0], [0.0, 1.0], [0.0, 2.0], [0.0, 3.0], [0.0, 4.0]])
    right = np.array([[10.0, 0.0], [10.0, 0.2], [10.0, 0.4], [10.0, 0.6], [10.0, 0.8]])
    majority = np.array([[1.0, 0.0], [11.0, 0.0], [1.0, 4.0], [11.0, 0.8]])
    X = np.vstack([left, right, majority])
    y = np.array([1] * 5 + [1] * 5 + [0] * 4)
    sampler = KMeansSMOTE(
        kmeans_estimator=ThresholdClusterer(threshold=5.0),
        random_state=0,
        k_neighbors=1,
        cluster_balance_threshold=0.0,
        sampling_strategy={1: 16},
    )
    X_res, y_res = sampler.fit_resample(X, y)
    quotas = sampler._source_quotas_[1]
    sparsities = [
        sampler._find_cluster_sparsity(left),
        sampler._find_cluster_sparsity(right),
    ]
    assert sparsities[0] > sparsities[1]
    # the exact plan lies within one row of each ideal share
    ideal = 6 * np.array(sparsities) / np.sum(sparsities)
    assert np.all(np.abs(quotas - ideal) < 1)
    assert quotas[0] > quotas[1]
    assert np.bincount(y_res)[1] == 16
    new_rows = X_res[X.shape[0] :]
    assert np.sum(new_rows[:, 0] < 5) == quotas[0]
    assert np.sum(new_rows[:, 0] >= 5) == quotas[1]


def test_kmeans_smote_positive_and_zero_share_mixed():
    """Positive weights and zero-weight sources coexist."""
    X, y = _mixed_sparsity_clusters()
    sampler = KMeansSMOTE(
        kmeans_estimator=ThresholdClusterer(threshold=50.0),
        random_state=0,
        k_neighbors=1,
        cluster_balance_threshold=0.0,
        sampling_strategy={1: 13},
    )
    X_res, y_res = sampler.fit_resample(X, y)
    quotas = sampler._source_quotas_[1]
    assert quotas[1] == 0
    assert quotas[0] == 3
    assert np.bincount(y_res)[1] == 13


def test_kmeans_smote_interpolates_between_same_class_rows():
    """Both endpoints of every generated row belong to the class being
    sampled and to the same eligible cluster."""
    X, y = _two_separated_clusters()
    sampler = KMeansSMOTE(
        kmeans_estimator=ThresholdClusterer(threshold=5.0),
        random_state=0,
        k_neighbors=1,
        cluster_balance_threshold=0.0,
        sampling_strategy={1: 14},
    )
    X_res, y_res = sampler.fit_resample(X, y)
    new_rows = X_res[X.shape[0] :]
    assert len(new_rows) == 4
    left_class = X[y == 1][X[y == 1][:, 0] < 5]
    right_class = X[y == 1][X[y == 1][:, 0] >= 5]
    for row in new_rows:
        if row[0] < 5:
            assert _on_some_segment(row, left_class)
        else:
            assert _on_some_segment(row, right_class)


def _on_some_segment(point, class_points, tol=1e-9):
    """Return True if point lies on a segment joining two class rows."""
    for i in range(len(class_points)):
        for j in range(i + 1, len(class_points)):
            a, b = class_points[i], class_points[j]
            ab = b - a
            denom = ab @ ab
            if denom == 0:
                continue
            t = np.clip((point - a) @ ab / denom, 0.0, 1.0)
            closest = a + t * ab
            if np.linalg.norm(point - closest) <= tol:
                return True
    return False


def test_adasyn_unequal_weights_are_respected():
    """The plan follows ADASYN's anchor difficulty ratios.

    Anchor 0 has 3 of 4 neighbours outside class 1, anchors at rows 4, 5 have
    1 of 4, and the remaining anchors see only class 1 (zero ratio).
    """
    X = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [0.2, 0.0],
            [0.3, 0.0],
            [1.0, 0.0],
            [1.5, 0.0],
            [100.0, 0.0],
            [100.1, 0.0],
            [101.0, 0.0],
            [102.0, 0.0],
            [103.0, 0.0],
            [200.0, 0.0],
            [201.0, 0.0],
            [202.0, 0.0],
            [203.0, 0.0],
            [204.0, 0.0],
        ]
    )
    y = np.array([1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1])
    sampler = ADASYN(n_neighbors=4, random_state=0, sampling_strategy={1: 20})
    X_res, y_res = sampler.fit_resample(X, y)
    quotas = sampler._source_quotas_[1]
    ratio = np.array([0.75, 0.75, 0.75, 0.25, 0.25, 0.25, 0.25, 0, 0, 0, 0, 0])
    expected = allocate_integer_quotas(ratio, 8, random_state=0)
    assert_array_equal(quotas, expected)
    # the hardest anchors are allocated strictly more than the 1/4 ones, and
    # zero-ratio anchors never receive a row
    assert quotas[0] > quotas[3]
    assert np.all(quotas[7:] == 0)
    assert np.bincount(y_res)[1] == 20


def test_adasyn_multi_class_targets_each_class_exactly():
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
    counts = np.bincount(y_res)
    assert counts.tolist() == [10, 10, 9]
    assert X_res.shape[0] == X.shape[0] + 4 + 4
    # class 2 is untouched and the original rows come first, in order
    assert np.array_equal(X_res[: X.shape[0]], X)
    assert np.array_equal(y_res[: X.shape[0]], y)


def test_kmeans_smote_multi_class_targets_each_class_exactly():
    rng = np.random.RandomState(0)
    X = np.vstack(
        [
            rng.randn(8, 2) * 0.5,
            rng.randn(8, 2) * 0.5 + 5.0,
            rng.randn(24, 2) * 0.5 + 12.0,
        ]
    )
    y = np.array([0] * 8 + [1] * 8 + [2] * 24)
    sampler = KMeansSMOTE(
        kmeans_estimator=ThreeClusterer(boundaries=(2.5, 8.5)),
        random_state=0,
        k_neighbors=2,
        cluster_balance_threshold=0.0,
        sampling_strategy={0: 10, 1: 12},
    )
    X_res, y_res = sampler.fit_resample(X, y)
    counts = np.bincount(y_res)
    assert counts.tolist() == [10, 12, 24]
    assert X_res.shape[0] == X.shape[0] + 2 + 4


def test_kmeans_smote_zero_quota_skips_construction(monkeypatch):
    """No rows are built for a class whose quota is zero."""
    X, y = _two_separated_clusters()
    sampler = KMeansSMOTE(
        kmeans_estimator=ThresholdClusterer(threshold=50.0),
        random_state=0,
        k_neighbors=1,
        cluster_balance_threshold=0.0,
        sampling_strategy={0: 4},
    )
    calls = []
    original = KMeansSMOTE._make_samples

    def spy(self, *args, **kwargs):
        calls.append(args[5])
        return original(self, *args, **kwargs)

    monkeypatch.setattr(KMeansSMOTE, "_make_samples", spy)
    X_res, y_res = sampler.fit_resample(X, y)
    assert calls == []
    assert np.array_equal(y_res, y)


def test_shared_plan_is_identical_for_both_samplers():
    """Both samplers expose the shared plan entry point."""
    weights = np.array([0.5, 1.0, 1.5, 2.0])
    total = 7
    seed = 3

    expected = allocate_integer_quotas(weights, total, random_state=seed)
    km = KMeansSMOTE(
        kmeans_estimator=ThresholdClusterer(threshold=50.0), random_state=seed
    )
    assert_array_equal(km._allocate_source_quotas(weights, total), expected)
    ada = ADASYN(random_state=seed)
    assert_array_equal(ada._allocate_source_quotas(weights, total), expected)


def test_allocate_entry_point_is_used_by_both_samplers():
    """Both samplers route their plan through the shared entry point.

    Spying on the shared helper shows that each sampler passes its own
    ordered weights and the class total through exactly one code path.
    """
    from imblearn.over_sampling import base as over_sampling_base

    calls = []
    original = over_sampling_base.allocate_integer_quotas

    def spy(weights, n_samples, random_state=None):
        calls.append((np.asarray(weights).copy(), int(n_samples)))
        return original(weights, n_samples, random_state)

    over_sampling_base.allocate_integer_quotas = spy
    try:
        X, y = _two_separated_clusters()
        KMeansSMOTE(
            kmeans_estimator=ThresholdClusterer(threshold=5.0),
            random_state=0,
            k_neighbors=1,
            cluster_balance_threshold=0.0,
            sampling_strategy={1: 13},
        ).fit_resample(X, y)

        X_ada, y_ada = _three_equal_anchors()
        ADASYN(n_neighbors=2, random_state=0, sampling_strategy={1: 7}).fit_resample(
            X_ada, y_ada
        )
    finally:
        over_sampling_base.allocate_integer_quotas = original

    weights_seen = [weights for weights, _ in calls]
    totals_seen = [total for _, total in calls]
    assert len(calls) == 2
    assert totals_seen == [3, 4]
    # two clusters with the same geometry have the same sparsity weight
    assert_allclose(weights_seen[0], [weights_seen[0][0]] * 2)
    # three identical anchors have the same difficulty ratio
    assert_allclose(weights_seen[1], [weights_seen[1][0]] * 3)
    assert np.all(weights_seen[1] > 0)


def test_pipeline_final_estimator_sees_exact_class_counts():
    class RecordingClassifier(BaseEstimator):
        def fit(self, X, y):
            self.class_counts_ = np.bincount(y)
            return self

        def predict(self, X):
            return np.zeros(X.shape[0], dtype=int)

    X, y = _two_separated_clusters()
    pipeline = Pipeline(
        [
            (
                "sampler",
                KMeansSMOTE(
                    kmeans_estimator=ThresholdClusterer(threshold=5.0),
                    random_state=0,
                    k_neighbors=1,
                    cluster_balance_threshold=0.0,
                    sampling_strategy={1: 13},
                ),
            ),
            ("classifier", RecordingClassifier()),
        ]
    )
    pipeline.fit(X, y)
    assert pipeline.named_steps["classifier"].class_counts_.tolist() == [4, 13]

    X_ada = np.array(
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
    y_ada = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0])
    pipeline = Pipeline(
        [
            (
                "sampler",
                ADASYN(n_neighbors=2, random_state=0, sampling_strategy={1: 6}),
            ),
            ("classifier", RecordingClassifier()),
        ]
    )
    pipeline.fit(X_ada, y_ada)
    assert pipeline.named_steps["classifier"].class_counts_.tolist() == [6, 6]


def test_pipeline_with_estimator_checks_real_class_counts():
    """Final estimator validates the counts it actually receives."""
    X, y = _two_separated_clusters()

    class CountCheckingClassifier(BaseEstimator):
        def __init__(self, expected_counts=None):
            self.expected_counts = expected_counts

        def fit(self, X, y):
            counts = np.bincount(y)
            if self.expected_counts is not None:
                assert counts.tolist() == list(self.expected_counts)
            self.seen_counts_ = counts
            self.n_features_ = X.shape[1]
            return self

        def predict(self, X):
            return np.zeros(X.shape[0], dtype=int)

    pipeline = Pipeline(
        [
            (
                "sampler",
                KMeansSMOTE(
                    kmeans_estimator=ThresholdClusterer(threshold=5.0),
                    random_state=0,
                    k_neighbors=1,
                    cluster_balance_threshold=0.0,
                    sampling_strategy={1: 13},
                ),
            ),
            ("classifier", CountCheckingClassifier(expected_counts=(4, 13))),
        ]
    )
    pipeline.fit(X, y)
    assert pipeline.named_steps["classifier"].seen_counts_.tolist() == [4, 13]
    assert pipeline.named_steps["classifier"].n_features_ == 2
