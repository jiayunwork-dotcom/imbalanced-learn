"""Test the module random under sampler."""
# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
#          Christos Aridas
# License: MIT

from collections import Counter
from datetime import datetime

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.utils._testing import assert_array_equal

from imblearn.under_sampling import RandomUnderSampler

RND_SEED = 0
X = np.array(
    [
        [0.04352327, -0.20515826],
        [0.92923648, 0.76103773],
        [0.20792588, 1.49407907],
        [0.47104475, 0.44386323],
        [0.22950086, 0.33367433],
        [0.15490546, 0.3130677],
        [0.09125309, -0.85409574],
        [0.12372842, 0.6536186],
        [0.13347175, 0.12167502],
        [0.094035, -2.55298982],
    ]
)
Y = np.array([1, 0, 1, 0, 1, 1, 1, 1, 0, 1])


@pytest.mark.parametrize("as_frame", [True, False], ids=["dataframe", "array"])
def test_rus_fit_resample(as_frame):
    if as_frame:
        pd = pytest.importorskip("pandas")
        X_ = pd.DataFrame(X)
    else:
        X_ = X
    rus = RandomUnderSampler(random_state=RND_SEED, replacement=True)
    X_resampled, y_resampled = rus.fit_resample(X_, Y)

    X_gt = np.array(
        [
            [0.92923648, 0.76103773],
            [0.47104475, 0.44386323],
            [0.13347175, 0.12167502],
            [0.09125309, -0.85409574],
            [0.12372842, 0.6536186],
            [0.04352327, -0.20515826],
        ]
    )
    y_gt = np.array([0, 0, 0, 1, 1, 1])

    if as_frame:
        assert hasattr(X_resampled, "loc")
        X_resampled = X_resampled.to_numpy()

    assert_array_equal(X_resampled, X_gt)
    assert_array_equal(y_resampled, y_gt)


def test_rus_fit_resample_half():
    sampling_strategy = {0: 3, 1: 6}
    rus = RandomUnderSampler(
        sampling_strategy=sampling_strategy,
        random_state=RND_SEED,
        replacement=True,
    )
    X_resampled, y_resampled = rus.fit_resample(X, Y)

    X_gt = np.array(
        [
            [0.92923648, 0.76103773],
            [0.47104475, 0.44386323],
            [0.92923648, 0.76103773],
            [0.15490546, 0.3130677],
            [0.15490546, 0.3130677],
            [0.15490546, 0.3130677],
            [0.20792588, 1.49407907],
            [0.15490546, 0.3130677],
            [0.12372842, 0.6536186],
        ]
    )
    y_gt = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1])
    assert_array_equal(X_resampled, X_gt)
    assert_array_equal(y_resampled, y_gt)


def test_multiclass_fit_resample():
    y = Y.copy()
    y[5] = 2
    y[6] = 2
    rus = RandomUnderSampler(random_state=RND_SEED)
    X_resampled, y_resampled = rus.fit_resample(X, y)
    count_y_res = Counter(y_resampled)
    assert count_y_res[0] == 2
    assert count_y_res[1] == 2
    assert count_y_res[2] == 2


def test_random_under_sampling_heterogeneous_data():
    X_hetero = np.array(
        [["xxx", 1, 1.0], ["yyy", 2, 2.0], ["zzz", 3, 3.0]], dtype=object
    )
    y = np.array([0, 0, 1])
    rus = RandomUnderSampler(random_state=RND_SEED)
    X_res, y_res = rus.fit_resample(X_hetero, y)

    assert X_res.shape[0] == 2
    assert y_res.shape[0] == 2
    assert X_res.dtype == object


def test_random_under_sampling_nan_inf():
    # check that we can undersample even with missing or infinite data
    # regression tests for #605
    rng = np.random.RandomState(42)
    n_not_finite = X.shape[0] // 3
    row_indices = rng.choice(np.arange(X.shape[0]), size=n_not_finite)
    col_indices = rng.randint(0, X.shape[1], size=n_not_finite)
    not_finite_values = rng.choice([np.nan, np.inf], size=n_not_finite)

    X_ = X.copy()
    X_[row_indices, col_indices] = not_finite_values

    rus = RandomUnderSampler(random_state=0)
    X_res, y_res = rus.fit_resample(X_, Y)

    assert y_res.shape == (6,)
    assert X_res.shape == (6, 2)
    assert np.any(~np.isfinite(X_res))


@pytest.mark.parametrize(
    "sampling_strategy", ["auto", "majority", "not minority", "not majority", "all"]
)
def test_random_under_sampler_strings(sampling_strategy):
    """Check that we support all supposed strings as `sampling_strategy` in
    a sampler inheriting from `BaseUnderSampler`."""

    X, y = make_classification(
        n_samples=100,
        n_clusters_per_class=1,
        n_classes=3,
        weights=[0.1, 0.3, 0.6],
        random_state=0,
    )
    RandomUnderSampler(sampling_strategy=sampling_strategy).fit_resample(X, y)


def test_random_under_sampling_datetime():
    """Check that we don't convert input data and only sample from it."""
    pd = pytest.importorskip("pandas")
    X = pd.DataFrame({"label": [0, 0, 0, 1], "td": [datetime.now()] * 4})
    y = X["label"]
    rus = RandomUnderSampler(random_state=0)
    X_res, y_res = rus.fit_resample(X, y)

    pd.testing.assert_series_equal(X_res.dtypes, X.dtypes)
    pd.testing.assert_index_equal(X_res.index, y_res.index)
    assert_array_equal(y_res.to_numpy(), np.array([0, 1]))


def test_random_under_sampler_full_nat():
    """Check that we can return timedelta columns full of NaT.

    Non-regression test for:
    https://github.com/scikit-learn-contrib/imbalanced-learn/issues/1055
    """
    pd = pytest.importorskip("pandas")

    X = pd.DataFrame(
        {
            "col_str": ["abc", "def", "xyz"],
            "col_timedelta": pd.to_timedelta([np.nan, np.nan, np.nan]),
        }
    )
    y = np.array([0, 0, 1])

    X_res, y_res = RandomUnderSampler().fit_resample(X, y)
    assert X_res.shape == (2, 2)
    assert y_res.shape == (2,)

    assert X_res["col_timedelta"].dtype == "timedelta64[ns]"


# ---------------------------------------------------------------------------
# Group-aware under-sampling
# ---------------------------------------------------------------------------


def _grouped_dataset():
    """Three classes over single-label groups of equal size (2 rows).

    Class 0 has 3 groups, class 1 has 2 groups and class 2 has 4 groups.  The
    class sizes are 6, 4 and 8 so the balanced target is 4 (the size of the
    smallest class).
    """
    memberships = {0: 3, 1: 2, 2: 4}
    X_rows = []
    y_rows = []
    groups_rows = []
    group = 0
    for label, n_groups in memberships.items():
        for _ in range(n_groups):
            for _ in range(2):
                X_rows.append([group, label])
                y_rows.append(label)
                groups_rows.append(group)
            group += 1
    return (
        np.asarray(X_rows, dtype=float),
        np.asarray(y_rows),
        np.asarray(groups_rows),
    )


def test_rus_groups_keeps_whole_groups_and_balances():
    X, y, groups = _grouped_dataset()
    rus = RandomUnderSampler(random_state=0)
    X_res, y_res = rus.fit_resample(X, y, groups=groups)

    assert Counter(y_res) == {0: 4, 1: 4, 2: 4}
    assert rus.n_groups_ == 9
    assert len(rus.group_ids_) == 6
    assert set(rus.group_ids_).issubset(set(groups))
    # ``sample_indices_`` are exactly the rows of the selected groups.
    expected = np.flatnonzero(np.isin(groups, rus.group_ids_))
    assert_array_equal(rus.sample_indices_, expected)
    assert_array_equal(X_res, X[expected])
    assert_array_equal(y_res, y[expected])
    # Each row appears at most once and every class is still represented.
    assert len(np.unique(rus.sample_indices_)) == len(rus.sample_indices_)
    assert set(np.unique(y_res)) == {0, 1, 2}
    # At least one group is held out.
    assert set(rus.group_ids_) != set(groups)


def test_rus_groups_mixed_labels_holds_out_group_and_covers_classes():
    # Group 0: 4 rows of class 0; group 1: 2 rows of class 1; group 2: one row
    # of each class; group 3: 2 rows of each class.  A group mixes labels, so
    # removing it moves both classes at once.
    y = np.array([0] * 4 + [1] * 2 + [0, 1] + [0, 0, 1, 1])
    groups = np.repeat(np.arange(4), [4, 2, 2, 4])
    X = np.arange(y.size * 2, dtype=float).reshape(-1, 2)

    rus = RandomUnderSampler(random_state=0)
    X_res, y_res = rus.fit_resample(X, y, groups=groups)

    expected = np.flatnonzero(np.isin(groups, rus.group_ids_))
    assert_array_equal(rus.sample_indices_, expected)
    assert_array_equal(X_res, X[expected])
    # Every class is represented and at least one group is left out.
    assert set(y_res) == {0, 1}
    assert len(rus.group_ids_) < 4
    # The "target" is the size of the smallest original class, i.e. 6 rows of
    # class 1, so keeping the two pure groups cannot reach the target and the
    # deviation is visible in the actual class counts.
    assert Counter(y_res) != {0: 6, 1: 6}
    assert Counter(y_res)[0] >= 1 and Counter(y_res)[1] >= 1


def test_rus_groups_string_identifiers():
    X, y, groups = _grouped_dataset()
    labels = np.array([f"device-{g}" for g in groups])
    rus = RandomUnderSampler(random_state=0)
    _, y_res = rus.fit_resample(X, y, groups=labels)
    assert set(rus.group_ids_).issubset(set(labels))
    assert Counter(y_res) == {0: 4, 1: 4, 2: 4}


def test_rus_groups_csr(tmp_path):
    sp = pytest.importorskip("scipy.sparse")
    X, y, groups = _grouped_dataset()
    sampler_dense = RandomUnderSampler(random_state=0)
    _, y_dense = sampler_dense.fit_resample(X, y, groups=groups)

    sampler_sparse = RandomUnderSampler(random_state=0)
    X_sparse, y_sparse = sampler_sparse.fit_resample(
        sp.csr_matrix(X), y, groups=groups
    )

    assert sp.issparse(X_sparse)
    assert_array_equal(
        X_sparse.toarray(), X[sampler_sparse.sample_indices_]
    )
    assert_array_equal(y_sparse, y_dense)
    assert_array_equal(
        sampler_sparse.sample_indices_, sampler_dense.sample_indices_
    )


def test_rus_groups_reproducible():
    X, y, groups = _grouped_dataset()
    first = RandomUnderSampler(random_state=42).fit_resample(X, y, groups=groups)
    second = RandomUnderSampler(random_state=42).fit_resample(X, y, groups=groups)
    assert_array_equal(first[0], second[0])
    assert_array_equal(first[1], second[1])


def test_rus_groups_does_not_mutate_unrelated_behaviour():
    X, y, groups = _grouped_dataset()
    # The legacy path is unchanged when ``groups`` is omitted or None.
    legacy = RandomUnderSampler(random_state=0)
    X_legacy, y_legacy = legacy.fit_resample(X, y)
    explicit_none = RandomUnderSampler(random_state=0)
    X_none, y_none = explicit_none.fit_resample(X, y, groups=None)
    assert_array_equal(X_legacy, X_none)
    assert_array_equal(y_legacy, y_none)
    assert_array_equal(legacy.sample_indices_, explicit_none.sample_indices_)
    assert not hasattr(legacy, "groups_")


def test_rus_groups_clears_previous_group_attributes():
    X, y, groups = _grouped_dataset()
    rus = RandomUnderSampler(random_state=0)
    rus.fit_resample(X, y, groups=groups)
    assert hasattr(rus, "groups_")
    X_legacy, y_legacy = rus.fit_resample(X, y)
    for attribute in ("groups_", "group_ids_", "group_codes_", "n_groups_"):
        assert not hasattr(rus, attribute)
    expected = RandomUnderSampler(random_state=0).fit_resample(X, y)
    assert_array_equal(X_legacy, expected[0])
    assert_array_equal(y_legacy, expected[1])


@pytest.mark.parametrize(
    "groups, match",
    [
        (np.zeros(5, dtype=int), "same number of rows"),
        (np.zeros((18, 2), dtype=int), "one-dimensional"),
        (np.array([None] * 18), "NoneType"),
        (np.array([1.5] * 18), "type float"),
        (np.array([""] * 18), "empty string"),
        (np.array([True] * 18), "boolean"),
    ],
)
def test_rus_groups_invalid(groups, match):
    X, y, _ = _grouped_dataset()
    with pytest.raises(ValueError, match=match):
        RandomUnderSampler().fit_resample(X, y, groups=groups)


@pytest.mark.parametrize(
    "params",
    [{"sampling_strategy": "all"}, {"replacement": True}],
)
def test_rus_groups_unsupported_parameters(params):
    X, y, groups = _grouped_dataset()
    with pytest.raises(ValueError):
        RandomUnderSampler(**params).fit_resample(X, y, groups=groups)
