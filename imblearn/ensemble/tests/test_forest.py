import numpy as np
import pytest
from sklearn.base import clone
from sklearn.datasets import make_classification
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils._testing import assert_allclose, assert_array_equal
from sklearn.utils.fixes import parse_version

from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.utils._sklearn_compat import sklearn_version


@pytest.fixture
def imbalanced_dataset():
    return make_classification(
        n_samples=10000,
        n_features=2,
        n_informative=2,
        n_redundant=0,
        n_repeated=0,
        n_classes=3,
        n_clusters_per_class=1,
        weights=[0.01, 0.05, 0.94],
        class_sep=0.8,
        random_state=0,
    )


def test_balanced_random_forest_error_warning_warm_start(imbalanced_dataset):
    brf = BalancedRandomForestClassifier(
        n_estimators=5, sampling_strategy="all", replacement=True, bootstrap=False
    )
    brf.fit(*imbalanced_dataset)

    with pytest.raises(ValueError, match="must be larger or equal to"):
        brf.set_params(warm_start=True, n_estimators=2)
        brf.fit(*imbalanced_dataset)

    brf.set_params(n_estimators=10)
    brf.fit(*imbalanced_dataset)

    with pytest.warns(UserWarning, match="Warm-start fitting without"):
        brf.fit(*imbalanced_dataset)


def test_balanced_random_forest(imbalanced_dataset):
    n_estimators = 10
    brf = BalancedRandomForestClassifier(
        n_estimators=n_estimators,
        random_state=0,
        sampling_strategy="all",
        replacement=True,
        bootstrap=False,
    )
    brf.fit(*imbalanced_dataset)

    assert len(brf.samplers_) == n_estimators
    assert len(brf.estimators_) == n_estimators
    assert len(brf.pipelines_) == n_estimators
    assert len(brf.feature_importances_) == imbalanced_dataset[0].shape[1]


def test_balanced_random_forest_attributes(imbalanced_dataset):
    X, y = imbalanced_dataset
    n_estimators = 10
    brf = BalancedRandomForestClassifier(
        n_estimators=n_estimators,
        random_state=0,
        sampling_strategy="all",
        replacement=True,
        bootstrap=False,
    )
    brf.fit(X, y)

    for idx in range(n_estimators):
        X_res, y_res = brf.samplers_[idx].fit_resample(X, y)
        X_res_2, y_res_2 = (
            brf.pipelines_[idx].named_steps["randomundersampler"].fit_resample(X, y)
        )
        assert_allclose(X_res, X_res_2)
        assert_array_equal(y_res, y_res_2)

        y_pred = brf.estimators_[idx].fit(X_res, y_res).predict(X)
        y_pred_2 = brf.pipelines_[idx].fit(X, y).predict(X)
        assert_array_equal(y_pred, y_pred_2)

        y_pred = brf.estimators_[idx].fit(X_res, y_res).predict_proba(X)
        y_pred_2 = brf.pipelines_[idx].fit(X, y).predict_proba(X)
        assert_array_equal(y_pred, y_pred_2)


def test_balanced_random_forest_sample_weight(imbalanced_dataset):
    rng = np.random.RandomState(42)
    X, y = imbalanced_dataset
    sample_weight = rng.rand(y.shape[0])
    brf = BalancedRandomForestClassifier(
        n_estimators=5,
        random_state=0,
        sampling_strategy="all",
        replacement=True,
        bootstrap=False,
    )
    brf.fit(X, y, sample_weight)


@pytest.mark.filterwarnings("ignore:Some inputs do not have OOB scores")
def test_balanced_random_forest_oob(imbalanced_dataset):
    X, y = imbalanced_dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=42, stratify=y
    )
    est = BalancedRandomForestClassifier(
        oob_score=True,
        random_state=0,
        n_estimators=1000,
        min_samples_leaf=2,
        sampling_strategy="all",
        replacement=True,
        bootstrap=True,
    )

    est.fit(X_train, y_train)
    test_score = est.score(X_test, y_test)

    assert abs(test_score - est.oob_score_) < 0.1

    # Check warning if not enough estimators
    est = BalancedRandomForestClassifier(
        oob_score=True,
        random_state=0,
        n_estimators=1,
        bootstrap=True,
        sampling_strategy="all",
        replacement=True,
    )
    with pytest.warns(UserWarning) and np.errstate(divide="ignore", invalid="ignore"):
        est.fit(X, y)


def test_balanced_random_forest_grid_search(imbalanced_dataset):
    brf = BalancedRandomForestClassifier(
        sampling_strategy="all", replacement=True, bootstrap=False
    )
    grid = GridSearchCV(brf, {"n_estimators": (1, 2), "max_depth": (1, 2)}, cv=3)
    grid.fit(*imbalanced_dataset)


def test_little_tree_with_small_max_samples():
    rng = np.random.RandomState(1)

    X = rng.randn(10000, 2)
    y = rng.randn(10000) > 0

    # First fit with no restriction on max samples
    est1 = BalancedRandomForestClassifier(
        n_estimators=1,
        random_state=rng,
        max_samples=None,
        sampling_strategy="all",
        replacement=True,
        bootstrap=True,
    )

    # Second fit with max samples restricted to just 2
    est2 = BalancedRandomForestClassifier(
        n_estimators=1,
        random_state=rng,
        max_samples=2,
        sampling_strategy="all",
        replacement=True,
        bootstrap=True,
    )

    est1.fit(X, y)
    est2.fit(X, y)

    tree1 = est1.estimators_[0].tree_
    tree2 = est2.estimators_[0].tree_

    msg = "Tree without `max_samples` restriction should have more nodes"
    assert tree1.node_count > tree2.node_count, msg


def test_balanced_random_forest_pruning(imbalanced_dataset):
    brf = BalancedRandomForestClassifier(
        sampling_strategy="all", replacement=True, bootstrap=False
    )
    brf.fit(*imbalanced_dataset)
    n_nodes_no_pruning = brf.estimators_[0].tree_.node_count

    brf_pruned = BalancedRandomForestClassifier(
        ccp_alpha=0.015, sampling_strategy="all", replacement=True, bootstrap=False
    )
    brf_pruned.fit(*imbalanced_dataset)
    n_nodes_pruning = brf_pruned.estimators_[0].tree_.node_count

    assert n_nodes_no_pruning > n_nodes_pruning


@pytest.mark.parametrize("ratio", [0.5, 0.1])
@pytest.mark.filterwarnings("ignore:Some inputs do not have OOB scores")
def test_balanced_random_forest_oob_binomial(ratio):
    # Regression test for #655: check that the oob score is closed to 0.5
    # a binomial experiment.
    rng = np.random.RandomState(42)
    n_samples = 1000
    X = np.arange(n_samples).reshape(-1, 1)
    y = rng.binomial(1, ratio, size=n_samples)

    erf = BalancedRandomForestClassifier(
        oob_score=True,
        random_state=42,
        sampling_strategy="not minority",
        replacement=False,
        bootstrap=True,
    )
    erf.fit(X, y)
    assert np.abs(erf.oob_score_ - 0.5) < 0.1


@pytest.mark.skipif(
    parse_version(sklearn_version.base_version) < parse_version("1.4"),
    reason="scikit-learn should be >= 1.4",
)
def test_missing_values_is_resilient():
    """Check that forest can deal with missing values and has decent performance."""

    rng = np.random.RandomState(0)
    n_samples, n_features = 1000, 10
    X, y = make_classification(
        n_samples=n_samples, n_features=n_features, random_state=rng
    )

    # Create dataset with missing values
    X_missing = X.copy()
    X_missing[rng.choice([False, True], size=X.shape, p=[0.95, 0.05])] = np.nan
    assert np.isnan(X_missing).any()

    X_missing_train, X_missing_test, y_train, y_test = train_test_split(
        X_missing, y, random_state=0
    )

    # Train forest with missing values
    forest_with_missing = BalancedRandomForestClassifier(
        sampling_strategy="all",
        replacement=True,
        bootstrap=False,
        random_state=rng,
        n_estimators=50,
    )
    forest_with_missing.fit(X_missing_train, y_train)
    score_with_missing = forest_with_missing.score(X_missing_test, y_test)

    # Train forest without missing values
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    forest = BalancedRandomForestClassifier(
        sampling_strategy="all",
        replacement=True,
        bootstrap=False,
        random_state=rng,
        n_estimators=50,
    )
    forest.fit(X_train, y_train)
    score_without_missing = forest.score(X_test, y_test)

    # Score is still 80 percent of the forest's score that had no missing values
    assert score_with_missing >= 0.80 * score_without_missing


@pytest.mark.skipif(
    parse_version(sklearn_version.base_version) < parse_version("1.4"),
    reason="scikit-learn should be >= 1.4",
)
def test_missing_value_is_predictive():
    """Check that the forest learns when missing values are only present for
    a predictive feature."""
    rng = np.random.RandomState(0)
    n_samples = 300

    X_non_predictive = rng.standard_normal(size=(n_samples, 10))
    y = rng.randint(0, high=2, size=n_samples)

    # Create a predictive feature using `y` and with some noise
    X_random_mask = rng.choice([False, True], size=n_samples, p=[0.95, 0.05])
    y_mask = y.astype(bool)
    y_mask[X_random_mask] = ~y_mask[X_random_mask]

    predictive_feature = rng.standard_normal(size=n_samples)
    predictive_feature[y_mask] = np.nan
    assert np.isnan(predictive_feature).any()

    X_predictive = X_non_predictive.copy()
    X_predictive[:, 5] = predictive_feature

    (
        X_predictive_train,
        X_predictive_test,
        X_non_predictive_train,
        X_non_predictive_test,
        y_train,
        y_test,
    ) = train_test_split(X_predictive, X_non_predictive, y, random_state=0)
    forest_predictive = BalancedRandomForestClassifier(
        sampling_strategy="all", replacement=True, bootstrap=False, random_state=0
    ).fit(X_predictive_train, y_train)
    forest_non_predictive = BalancedRandomForestClassifier(
        sampling_strategy="all", replacement=True, bootstrap=False, random_state=0
    ).fit(X_non_predictive_train, y_train)

    predictive_test_score = forest_predictive.score(X_predictive_test, y_test)

    assert predictive_test_score >= 0.75
    assert predictive_test_score >= forest_non_predictive.score(
        X_non_predictive_test, y_test
    )


# ---------------------------------------------------------------------------
# Group-aware sampling
# ---------------------------------------------------------------------------


def _grouped_forest_dataset(n_groups_per_class=(3, 2, 4), rows_per_group=2):
    """Single-label groups of equal size with imbalanced class counts."""
    X_rows, y_rows, groups_rows = [], [], []
    group = 0
    for label, n_groups in enumerate(n_groups_per_class):
        for _ in range(n_groups):
            for _ in range(rows_per_group):
                X_rows.append(
                    [
                        group + 0.5 * label,
                        float(label),
                        float(group % 3),
                    ]
                )
                y_rows.append(label)
                groups_rows.append(group)
            group += 1
    return (
        np.asarray(X_rows, dtype=float),
        np.asarray(y_rows),
        np.asarray(groups_rows),
    )


def _fitted_group_forest(X, y, groups, **params):
    defaults = dict(
        n_estimators=7,
        random_state=0,
        sampling_strategy="auto",
        replacement=False,
        bootstrap=False,
        max_samples=None,
        warm_start=False,
    )
    defaults.update({k: v for k, v in params.items() if k != "sample_weight"})
    forest = BalancedRandomForestClassifier(**defaults)
    forest.fit(X, y, sample_weight=params.get("sample_weight"), groups=groups)
    return forest


def test_grouped_forest_samples_rows_of_selected_groups():
    X, y, groups = _grouped_forest_dataset()
    forest = _fitted_group_forest(X, y, groups)

    assert forest.n_groups_ == 9
    assert_array_equal(forest.groups_, groups)
    assert len(forest.estimators_) == len(forest.samplers_) == 7

    for sampler in forest.samplers_:
        selected = sampler.group_ids_
        expected = np.flatnonzero(np.isin(groups, selected))
        assert_array_equal(sampler.sample_indices_, expected)
        assert set(np.unique(y[expected])) == {0, 1, 2}
        # Each row appears at most once and at least one group is held out.
        assert sampler.sample_indices_.size == len(
            np.unique(sampler.sample_indices_)
        )
        assert len(selected) < forest.n_groups_


def test_grouped_forest_class_counts_balanced():
    X, y, groups = _grouped_forest_dataset()
    forest = _fitted_group_forest(X, y, groups)
    for sampler in forest.samplers_:
        counts = np.bincount(y[sampler.sample_indices_], minlength=3)
        # Smallest original class has 2 groups * 2 rows = 4 rows, so with
        # equal-size single-label groups an exact balance is reachable.
        assert_array_equal(counts, np.array([4, 4, 4]))


def test_grouped_forest_trees_select_different_groups():
    X, y, groups = _grouped_forest_dataset(n_groups_per_class=(4, 4, 4))
    forest = _fitted_group_forest(X, y, groups, n_estimators=20)
    selections = {tuple(sorted(s.group_codes_)) for s in forest.samplers_}
    assert len(selections) > 1


def test_grouped_forest_sample_weight_aligned():
    X, y, groups = _grouped_forest_dataset()
    weights = np.linspace(0.1, 3.0, X.shape[0])

    forest = _fitted_group_forest(X, y, groups, n_estimators=1)
    sampler = forest.samplers_[0]
    tree = forest.estimators_[0]
    rows = sampler.sample_indices_
    reference = DecisionTreeClassifier(**{
        key: value
        for key, value in tree.get_params().items()
        if key != "random_state"
    }, random_state=tree.random_state)
    reference.fit(X[rows], y[rows], sample_weight=weights[rows])
    assert_array_equal(tree.predict(X), reference.predict(X))

    # The root node of the tree must have accumulated the weights of the
    # sampled rows only, in the sampler's order.
    weighted = _fitted_group_forest(
        X, y, groups, n_estimators=1, sample_weight=weights
    )
    weighted_rows = weighted.samplers_[0].sample_indices_
    root_weight = weighted.estimators_[0].tree_.weighted_n_node_samples[0]
    assert_allclose(root_weight, weights[weighted_rows].sum())
    assert_allclose(weighted.estimators_[0].tree_.n_node_samples[0], len(rows))


def test_grouped_forest_oob_counts_and_probabilities_from_scratch():
    X, y, groups = _grouped_forest_dataset(n_groups_per_class=(3, 2, 4))
    n_estimators = 8
    forest = _fitted_group_forest(
        X, y, groups, n_estimators=n_estimators, oob_score=True
    )

    n_samples = X.shape[0]
    n_classes = len(forest.classes_)
    votes = np.zeros((n_samples, n_classes))
    counts = np.zeros(n_samples, dtype=int)
    for sampler, estimator in zip(forest.samplers_, forest.estimators_):
        eligible = ~np.isin(groups, sampler.group_ids_)
        counts += eligible
        votes[eligible] += estimator.predict_proba(X[eligible])
    assert_array_equal(forest.oob_counts_, counts)
    # Each tree saw every class (the grouped sampler keeps every class
    # represented), so the columns follow ``classes_``.
    for estimator in forest.estimators_:
        assert_array_equal(estimator.classes_, forest.classes_)

    expected = np.full_like(votes, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        expected[counts > 0] = votes[counts > 0] / counts[counts > 0, np.newaxis]
    assert_allclose(forest.oob_decision_function_, expected, equal_nan=True)

    eligible_rows = forest.oob_counts_ > 0
    predicted = np.argmax(forest.oob_decision_function_[eligible_rows], axis=1)
    expected_score = np.mean(forest.classes_[predicted] == y[eligible_rows])
    assert_allclose(forest.oob_score_, expected_score)


def test_grouped_forest_oob_partial_zero_votes():
    # Two groups per class of 2 rows each: the tree keeps one group per class
    # and leaves the other two groups out.
    X, y, groups = _grouped_forest_dataset(
        n_groups_per_class=(2, 2), rows_per_group=2
    )

    forest = _fitted_group_forest(X, y, groups, n_estimators=1, oob_score=True)
    trained_rows = np.flatnonzero(
        np.isin(groups, forest.samplers_[0].group_ids_)
    )
    held_out_rows = np.flatnonzero(forest.oob_counts_ == 0)
    # The rows of the groups the tree trained on get no vote.
    assert_array_equal(held_out_rows, trained_rows)
    assert np.isnan(forest.oob_decision_function_[held_out_rows]).all()
    assert (forest.oob_counts_[held_out_rows] == 0).all()
    assert np.isfinite(
        forest.oob_decision_function_[forest.oob_counts_ > 0]
    ).all()
    assert np.isfinite(forest.oob_score_)


def test_grouped_forest_oob_all_zero_votes_is_nan():
    X = np.arange(8, dtype=float).reshape(4, 2)
    y = np.array([0, 0, 1, 1])
    # A single group cannot be held out: no row is out of bag.
    forest = _fitted_group_forest(
        X, y, np.zeros(4, dtype=int), n_estimators=1, oob_score=True
    )
    assert (forest.oob_counts_ == 0).all()
    assert np.isnan(forest.oob_decision_function_).all()
    assert np.isnan(forest.oob_score_)


def test_grouped_forest_reproducible_across_fits_and_n_jobs():
    X, y, groups = _grouped_forest_dataset(n_groups_per_class=(4, 4, 4))
    forests = [
        _fitted_group_forest(
            X, y, groups, n_estimators=9, oob_score=True, n_jobs=n_jobs
        )
        for n_jobs in (1, 1, 2)
    ]
    # Re-fitting the same instance gives the same result as a fresh fit.
    reused = _fitted_group_forest(
        X, y, groups, n_estimators=9, oob_score=True, n_jobs=1
    )
    reused.fit(X, y, groups=groups)
    forests.append(reused)

    reference = forests[0]
    for forest in forests[1:]:
        assert_array_equal(
            [s.sample_indices_ for s in reference.samplers_],
            [s.sample_indices_ for s in forest.samplers_],
        )
        assert_array_equal(reference.oob_counts_, forest.oob_counts_)
        assert_allclose(
            reference.oob_decision_function_,
            forest.oob_decision_function_,
            equal_nan=True,
        )
        assert_array_equal(reference.predict(X), forest.predict(X))
        assert_allclose(reference.predict_proba(X), forest.predict_proba(X))


def test_grouped_forest_csr_matches_dense():
    sp = pytest.importorskip("scipy.sparse")
    X, y, groups = _grouped_forest_dataset()
    dense = _fitted_group_forest(X, y, groups, oob_score=True)
    sparse = _fitted_group_forest(sp.csr_matrix(X), y, groups, oob_score=True)

    assert_array_equal(
        [s.sample_indices_ for s in dense.samplers_],
        [s.sample_indices_ for s in sparse.samplers_],
    )
    assert_allclose(dense.oob_decision_function_, sparse.oob_decision_function_)
    assert_array_equal(dense.predict(X), sparse.predict(X))
    assert_allclose(dense.predict_proba(X), sparse.predict_proba(X))


def test_grouped_forest_refit_clears_group_state_and_restores_legacy():
    X, y, groups = _grouped_forest_dataset()
    forest = _fitted_group_forest(X, y, groups, oob_score=True)
    assert hasattr(forest, "groups_")
    assert hasattr(forest, "oob_counts_")

    forest.set_params(sampling_strategy="all", replacement=True, oob_score=False)
    forest.fit(X, y)
    for attribute in ("groups_", "_groups_codes_", "n_groups_", "oob_counts_"):
        assert not hasattr(forest, attribute)

    legacy = BalancedRandomForestClassifier(
        **{p: getattr(forest, p) for p in forest.get_params(deep=False)}
    )
    legacy.fit(X, y)
    assert_array_equal(forest.predict(X), legacy.predict(X))
    assert_array_equal(forest.predict_proba(X), legacy.predict_proba(X))


@pytest.mark.parametrize(
    "params, match",
    [
        ({"bootstrap": True}, "bootstrap=False"),
        ({"max_samples": 3}, "max_samples=None"),
        ({"warm_start": True}, "warm_start=False"),
        ({"sampling_strategy": "all"}, "sampling_strategy='auto'"),
        ({"replacement": True}, "replacement=True"),
    ],
)
def test_grouped_forest_unsupported_parameters_raise_before_building(params, match):
    X, y, groups = _grouped_forest_dataset()
    forest_params = dict(
        n_estimators=3,
        random_state=0,
        oob_score=True,
        sampling_strategy="auto",
        replacement=False,
        bootstrap=False,
    )
    forest_params.update(params)
    forest = BalancedRandomForestClassifier(**forest_params)
    with pytest.raises(ValueError, match=match):
        forest.fit(X, y, groups=groups)
    assert not hasattr(forest, "estimators_")


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
def test_grouped_forest_invalid_groups(groups, match):
    X, y, _ = _grouped_forest_dataset()
    forest = BalancedRandomForestClassifier(
        n_estimators=3,
        sampling_strategy="auto",
        replacement=False,
        bootstrap=False,
    )
    with pytest.raises(ValueError, match=match):
        forest.fit(X, y, groups=groups)
    assert not hasattr(forest, "estimators_")


def test_grouped_forest_mixed_group_approximate_balance():
    # Group 0 is pure class 0 (4 rows), group 1 pure class 1 (2 rows), group 2
    # mixes both classes and group 3 has two rows of each class.
    y = np.array([0] * 4 + [1] * 2 + [0, 1] + [0, 0, 1, 1])
    groups = np.repeat(np.arange(4), [4, 2, 2, 4])
    X = np.column_stack([np.arange(y.size, dtype=float), groups.astype(float)])
    forest = _fitted_group_forest(X, y, groups, n_estimators=5, oob_score=True)

    for sampler in forest.samplers_:
        assert set(np.unique(y[sampler.sample_indices_])) == {0, 1}
        expected = np.flatnonzero(np.isin(groups, sampler.group_ids_))
        assert_array_equal(sampler.sample_indices_, expected)
        assert len(sampler.group_ids_) < 4
    assert np.isfinite(forest.oob_score_)


def test_grouped_forest_legacy_path_unchanged():
    X, y, _ = _grouped_forest_dataset()
    params = dict(
        n_estimators=5,
        random_state=0,
        sampling_strategy="all",
        replacement=True,
        bootstrap=False,
    )
    omitted = BalancedRandomForestClassifier(**params).fit(X, y)
    explicit_none = BalancedRandomForestClassifier(**params).fit(X, y, groups=None)
    assert_array_equal(
        [s.sample_indices_ for s in omitted.samplers_],
        [s.sample_indices_ for s in explicit_none.samplers_],
    )
    assert_array_equal(omitted.predict(X), explicit_none.predict(X))
    assert not hasattr(omitted, "groups_")


def test_grouped_forest_oob_uses_classes_column_order():
    X, y_int, groups = _grouped_forest_dataset()
    names = {0: "minority", 1: "middle", 2: "majority"}
    y = np.array([names[label] for label in y_int])
    forest = _fitted_group_forest(X, y, groups, n_estimators=6, oob_score=True)
    # ``classes_`` is sorted, so its order differs from the class codes.
    assert list(forest.classes_) == ["majority", "middle", "minority"]

    votes = np.zeros((X.shape[0], len(forest.classes_)))
    counts = np.zeros(X.shape[0], dtype=int)
    for sampler, estimator in zip(forest.samplers_, forest.estimators_):
        eligible = ~np.isin(groups, sampler.group_ids_)
        counts += eligible
        votes[eligible] += estimator.predict_proba(X[eligible])
    expected = np.full_like(votes, np.nan)
    mask = counts > 0
    expected[mask] = votes[mask] / counts[mask, np.newaxis]
    assert_allclose(forest.oob_decision_function_, expected, equal_nan=True)


def test_grouped_forest_oob_probabilities_sum_to_one():
    X, y, groups = _grouped_forest_dataset(n_groups_per_class=(3, 2, 4))
    forest = _fitted_group_forest(X, y, groups, n_estimators=10, oob_score=True)
    voted = forest.oob_counts_ > 0
    assert_allclose(forest.oob_decision_function_[voted].sum(axis=1), 1.0)
    # Every tree sees every class, hence all rows of a selected group are
    # trained on and cannot be voted on by that tree.
    for sampler in forest.samplers_:
        trained_rows = np.flatnonzero(np.isin(groups, sampler.group_ids_))
        assert trained_rows.size == sampler.sample_indices_.size


def test_grouped_forest_clone_and_pipeline_unchanged():
    X, y, groups = _grouped_forest_dataset()
    forest = _fitted_group_forest(X, y, groups, n_estimators=3)
    cloned = clone(forest)
    cloned.fit(X, y, groups=groups)
    assert_array_equal(forest.predict(X), cloned.predict(X))
    # The fitted pipeline of a tree exposes the usual predict API.
    assert_array_equal(
        forest.pipelines_[0].predict(X), forest.estimators_[0].predict(X)
    )
    assert forest.predict_proba(X).shape == (X.shape[0], len(forest.classes_))
