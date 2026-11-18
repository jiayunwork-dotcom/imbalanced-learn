"""Test the weight-conserving ``fit_resample_weighted`` of the samplers."""

# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
# License: MIT

import numpy as np
import pytest
from scipy import sparse
from sklearn.datasets import make_classification

from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.utils import compose_weighted_results

RTOL = 1e-10
ATOL = 1e-12


def _as_dense(X):
    return X.toarray() if sparse.issparse(X) else np.asarray(X)


def _make_data(n_samples=1000, weights=(0.1, 0.9), n_classes=2, random_state=10):
    if n_classes == 2:
        return make_classification(
            n_classes=2,
            class_sep=2,
            weights=list(weights),
            n_informative=3,
            n_redundant=1,
            flip_y=0,
            n_features=20,
            n_clusters_per_class=1,
            n_samples=n_samples,
            random_state=random_state,
        )
    return make_classification(
        n_classes=n_classes,
        class_sep=2,
        n_informative=5,
        n_redundant=1,
        flip_y=0,
        n_features=20,
        n_clusters_per_class=1,
        n_samples=n_samples,
        random_state=random_state,
    )


def _twin(sampler):
    """Build a fresh estimator with the same parameters."""
    return sampler.__class__(**sampler.get_params())


def _assert_same_resampling(sampler, X, y, sample_weight):
    """The weighted path must not change the resampled X/y."""
    result = sampler.fit_resample_weighted(X, y, sample_weight)
    X_expected, y_expected = _twin(sampler).fit_resample(X, y)
    if sparse.issparse(X_expected):
        assert sparse.issparse(result.X)
        assert result.X.format == X_expected.format
        assert (result.X != X_expected).nnz == 0
    else:
        np.testing.assert_allclose(_as_dense(result.X), _as_dense(X_expected))
    np.testing.assert_array_equal(np.asarray(result.y), np.asarray(y_expected))
    return result


SAMPLER_FACTORIES = [
    pytest.param(lambda: RandomOverSampler(random_state=42), id="RandomOverSampler"),
    pytest.param(lambda: SMOTE(random_state=42), id="SMOTE"),
    pytest.param(lambda: SMOTEENN(random_state=42), id="SMOTEENN"),
    pytest.param(lambda: EditedNearestNeighbours(), id="EditedNearestNeighbours"),
]


@pytest.mark.parametrize("make_sampler", SAMPLER_FACTORIES)
class TestWeightedSamplersCommonContract:
    def test_conservation(self, make_sampler):
        X, y = _make_data()
        rng = np.random.RandomState(0)
        sample_weight = rng.rand(X.shape[0]) * 2
        result = make_sampler().fit_resample_weighted(X, y, sample_weight)
        result.check_conserved(sample_weight, rtol=RTOL, atol=ATOL)
        assert result.is_conserved(sample_weight, rtol=RTOL, atol=ATOL)

    def test_conservation_zero_weights(self, make_sampler):
        X, y = _make_data()
        rng = np.random.RandomState(1)
        sample_weight = rng.rand(X.shape[0])
        # sprinkle zero weights, including on the minority class
        sample_weight[::7] = 0.0
        sample_weight[y == np.argmin(np.bincount(y))][:3] = 0.0
        result = make_sampler().fit_resample_weighted(X, y, sample_weight)
        result.check_conserved(sample_weight, rtol=RTOL, atol=ATOL)
        # contributions only hold non-zero records
        assert np.all(result.contributions.mass > 0)

    def test_same_resampling_as_unweighted(self, make_sampler):
        X, y = _make_data()
        sample_weight = np.random.RandomState(2).rand(X.shape[0])
        _assert_same_resampling(make_sampler(), X, y, sample_weight)

    def test_same_resampling_as_unweighted_csr(self, make_sampler):
        X, y = _make_data()
        X = sparse.csr_matrix(X)
        sample_weight = np.random.RandomState(3).rand(X.shape[0])
        result = _assert_same_resampling(make_sampler(), X, y, sample_weight)
        assert sparse.issparse(result.X)

    def test_multiclass_conservation(self, make_sampler):
        X, y = _make_data(n_classes=3)
        rng = np.random.RandomState(4)
        sample_weight = rng.rand(X.shape[0])
        result = make_sampler().fit_resample_weighted(X, y, sample_weight)
        result.check_conserved(sample_weight, rtol=RTOL, atol=ATOL)

    def test_no_cross_class_funding(self, make_sampler):
        X, y = _make_data(n_classes=3)
        sample_weight = np.random.RandomState(5).rand(X.shape[0]) + 0.1
        result = make_sampler().fit_resample_weighted(X, y, sample_weight)
        y_in = np.asarray(y)[result.contributions.input_index]
        y_out = np.asarray(result.y)[result.contributions.output_index]
        np.testing.assert_array_equal(y_in, y_out)

    def test_positive_weight_of_kept_and_synthetic_rows(self, make_sampler):
        # with positive weights everywhere, every kept original row and every
        # non-degenerate synthetic row must get a positive weight
        X, y = _make_data()
        sample_weight = np.random.RandomState(6).rand(X.shape[0]) + 0.5
        sampler = make_sampler()
        result = sampler.fit_resample_weighted(X, y, sample_weight)
        n_inputs = X.shape[0]
        if isinstance(sampler, EditedNearestNeighbours):
            kept = sampler.sample_indices_
            np.testing.assert_allclose(
                result.sample_weight, sample_weight[kept], rtol=RTOL, atol=ATOL
            )
            assert np.all(result.sample_weight > 0)
        else:
            # kept original rows keep a positive share
            assert np.all(result.sample_weight[:n_inputs] > 0)
            # synthetic rows funded by positive endpoints are positive; the
            # random data of this test has no degenerate pair
            assert np.all(result.sample_weight[n_inputs:] > 0)

    def test_dropped_weight_only_for_cleaning(self, make_sampler):
        X, y = _make_data()
        sample_weight = np.random.RandomState(7).rand(X.shape[0])
        sampler = make_sampler()
        result = sampler.fit_resample_weighted(X, y, sample_weight)
        if isinstance(sampler, (RandomOverSampler, SMOTE)):
            # pure over-samplers never drop mass
            np.testing.assert_array_equal(result.dropped_weight, np.zeros(X.shape[0]))
        assert result.dropped_weight.shape == (X.shape[0],)
        assert np.all(result.dropped_weight >= 0)

    def test_caller_arrays_not_modified(self, make_sampler):
        X, y = _make_data()
        sample_weight = np.random.RandomState(8).rand(X.shape[0])
        X_copy, y_copy, w_copy = X.copy(), y.copy(), sample_weight.copy()
        make_sampler().fit_resample_weighted(X, y, sample_weight)
        np.testing.assert_array_equal(X, X_copy)
        np.testing.assert_array_equal(y, y_copy)
        np.testing.assert_array_equal(sample_weight, w_copy)

    @pytest.mark.parametrize(
        "sample_weight",
        [
            np.ones(5),  # wrong length
            np.full(1000, np.nan),  # non-finite
            -np.ones(1000),  # negative
            np.zeros(1000),  # zero total mass
            np.ones((1000, 1)),  # not 1-D
        ],
        ids=["length", "nan", "negative", "zero-sum", "2d"],
    )
    def test_invalid_sample_weight(self, make_sampler, sample_weight):
        X, y = _make_data()
        sampler = make_sampler()
        with pytest.raises(ValueError):
            sampler.fit_resample_weighted(X, y, sample_weight)
        # the sampler must not have been fitted
        assert not hasattr(sampler, "sampling_strategy_")

    def test_fit_resample_still_returns_pair(self, make_sampler):
        X, y = _make_data()
        output = make_sampler().fit_resample(X, y)
        assert isinstance(output, tuple) and len(output) == 2


class TestRandomOverSamplerWeighted:
    def test_copies_funded_by_their_source_only(self):
        X, y = _make_data()
        rng = np.random.RandomState(9)
        sample_weight = rng.rand(X.shape[0]) + 0.1
        ros = RandomOverSampler(random_state=42)
        result = ros.fit_resample_weighted(X, y, sample_weight)
        X_dense = _as_dense(result.X)
        sources = ros.sample_indices_
        n_copies = np.bincount(sources, minlength=X.shape[0])
        for input_idx, output_idx, mass in result.contributions:
            # a copy is funded by the row it duplicates
            assert sources[output_idx] == input_idx
            np.testing.assert_allclose(X_dense[output_idx], X[input_idx])
            np.testing.assert_allclose(
                mass,
                sample_weight[input_idx] / n_copies[input_idx],
                rtol=RTOL,
                atol=ATOL,
            )

    def test_duplicate_rows_do_not_share_weight(self):
        # two rows with identical features but different weights are two
        # independent sources: copies of one never borrow from the other
        X = np.array([[0.0], [0.0], [1.0], [2.0], [3.0], [4.0]])
        y = np.array([0, 0, 1, 1, 1, 1])
        sample_weight = np.array([1.0, 3.0, 1.0, 1.0, 1.0, 1.0])
        ros = RandomOverSampler(random_state=0)
        result = ros.fit_resample_weighted(X, y, sample_weight)
        result.check_conserved(sample_weight)
        sources = ros.sample_indices_
        n_copies = np.bincount(sources, minlength=X.shape[0])
        for input_idx, output_idx, mass in result.contributions:
            assert sources[output_idx] == input_idx
            np.testing.assert_allclose(
                mass, sample_weight[input_idx] / n_copies[input_idx]
            )
        # the two identical rows kept distinct books
        assert 0 in sources and 1 in sources or n_copies[0] != n_copies[1]

    def test_shrinkage_not_supported(self):
        X, y = _make_data()
        sample_weight = np.ones(X.shape[0])
        ros = RandomOverSampler(random_state=0, shrinkage=0.5)
        with pytest.raises(ValueError, match="shrinkage"):
            ros.fit_resample_weighted(X, y, sample_weight)


class TestSMOTEWeighted:
    def test_synthetic_funded_by_actual_endpoints(self):
        X, y = _make_data()
        rng = np.random.RandomState(10)
        sample_weight = rng.rand(X.shape[0]) + 0.1
        smote = SMOTE(random_state=42)
        result = smote.fit_resample_weighted(X, y, sample_weight)
        n_inputs = X.shape[0]
        X_dense = _as_dense(result.X)
        X_in = _as_dense(X)

        # per-source shares: each input splits its weight over the outputs it
        # funds (its kept row plus one share per synthetic it generated)
        endpoint_count = np.zeros(n_inputs)
        for input_idx, output_idx, _ in result.contributions:
            if output_idx >= n_inputs:
                endpoint_count[input_idx] += 1
        shares = sample_weight / (1 + endpoint_count)

        synthetic_records = {}
        for input_idx, output_idx, mass in result.contributions:
            if output_idx >= n_inputs:
                synthetic_records.setdefault(output_idx, []).append((input_idx, mass))
            else:
                # kept original rows are funded by themselves only
                assert input_idx == output_idx
                np.testing.assert_allclose(
                    mass, shares[input_idx], rtol=RTOL, atol=ATOL
                )

        for output_idx, funders in synthetic_records.items():
            # a synthetic sample is funded by exactly its two endpoints
            assert len(funders) == 2
            (i, mass_i), (j, mass_j) = funders
            assert i != j
            np.testing.assert_allclose(mass_i, shares[i], rtol=RTOL, atol=ATOL)
            np.testing.assert_allclose(mass_j, shares[j], rtol=RTOL, atol=ATOL)
            # the endpoints are the points that actually generated the
            # sample: it lies on the segment between them
            segment = X_in[j] - X_in[i]
            point = X_dense[output_idx] - X_in[i]
            denom = segment @ segment
            assert denom > 0  # non-degenerate pair in this dataset
            step = (point @ segment) / denom
            assert -1e-6 <= step <= 1 + 1e-6
            np.testing.assert_allclose(point, step * segment, rtol=1e-6, atol=1e-8)

    def test_synthetic_positive_mass_with_positive_endpoints(self):
        X, y = _make_data()
        sample_weight = np.random.RandomState(11).rand(X.shape[0]) + 0.1
        smote = SMOTE(random_state=0)
        result = smote.fit_resample_weighted(X, y, sample_weight)
        n_inputs = X.shape[0]
        assert np.all(result.sample_weight[n_inputs:] > 0)

    def test_zero_weight_endpoints_give_zero_mass_synthetic(self):
        # a synthetic sample generated by two zero-weight endpoints gets a
        # zero weight; the books still balance
        X, y = _make_data()
        sample_weight = np.ones(X.shape[0])
        minority = np.flatnonzero(y == np.argmin(np.bincount(y)))
        sample_weight[minority[:5]] = 0.0
        smote = SMOTE(random_state=0)
        result = smote.fit_resample_weighted(X, y, sample_weight)
        result.check_conserved(sample_weight)
        # zero-weight sources fund nothing positive
        flow = result.contributions.to_scipy_sparse()
        assert flow[minority[:5]].nnz == 0


class TestEditedNearestNeighboursWeighted:
    def test_dropped_mass_is_not_redistributed(self):
        X, y = _make_data()
        rng = np.random.RandomState(12)
        sample_weight = rng.rand(X.shape[0]) + 0.1
        enn = EditedNearestNeighbours()
        result = enn.fit_resample_weighted(X, y, sample_weight)
        kept = enn.sample_indices_
        dropped = np.setdiff1d(np.arange(X.shape[0]), kept)
        # kept rows keep exactly their own weight: no renormalization
        np.testing.assert_allclose(
            result.sample_weight, sample_weight[kept], rtol=RTOL, atol=ATOL
        )
        # dropped rows hand their whole weight to dropped_weight
        np.testing.assert_allclose(
            result.dropped_weight[dropped],
            sample_weight[dropped],
            rtol=RTOL,
            atol=ATOL,
        )
        np.testing.assert_array_equal(result.dropped_weight[kept], 0.0)
        # kept rows are funded by themselves only
        for input_idx, output_idx, mass in result.contributions:
            assert kept[output_idx] == input_idx
            assert mass == sample_weight[input_idx]


class TestSMOTEENNWeighted:
    @pytest.mark.parametrize("to_sparse", [False, True], ids=["dense", "csr"])
    def test_equivalent_to_explicit_smote_then_enn(self, to_sparse):
        X, y = _make_data()
        if to_sparse:
            X = sparse.csr_matrix(X)
        rng = np.random.RandomState(13)
        sample_weight = rng.rand(X.shape[0]) + 0.1

        combo = SMOTEENN(random_state=42)
        result_combo = combo.fit_resample_weighted(X, y, sample_weight)

        first = SMOTE(random_state=42).fit_resample_weighted(X, y, sample_weight)
        second = EditedNearestNeighbours(sampling_strategy="all").fit_resample_weighted(
            first.X, first.y, first.sample_weight
        )
        result_manual = compose_weighted_results(first, second)

        # same data
        np.testing.assert_allclose(
            _as_dense(result_combo.X), _as_dense(result_manual.X)
        )
        np.testing.assert_array_equal(result_combo.y, result_manual.y)
        # same books
        np.testing.assert_allclose(
            result_combo.sample_weight,
            result_manual.sample_weight,
            rtol=RTOL,
            atol=ATOL,
        )
        np.testing.assert_allclose(
            result_combo.dropped_weight,
            result_manual.dropped_weight,
            rtol=RTOL,
            atol=ATOL,
        )
        flow_combo = result_combo.contributions.to_scipy_sparse()
        flow_manual = result_manual.contributions.to_scipy_sparse()
        assert (flow_combo != flow_manual).nnz == 0
        result_combo.check_conserved(sample_weight, rtol=RTOL, atol=ATOL)

    def test_equivalent_with_custom_sub_samplers(self):
        # the equivalence also holds with user-provided sub-samplers
        X, y = _make_data()
        sample_weight = np.random.RandomState(16).rand(X.shape[0]) + 0.1
        smote = SMOTE(random_state=7, k_neighbors=3)
        enn = EditedNearestNeighbours(sampling_strategy="all", n_neighbors=2)
        combo = SMOTEENN(random_state=7, smote=smote, enn=enn)
        result_combo = combo.fit_resample_weighted(X, y, sample_weight)

        first = SMOTE(random_state=7, k_neighbors=3).fit_resample_weighted(
            X, y, sample_weight
        )
        second = EditedNearestNeighbours(
            sampling_strategy="all", n_neighbors=2
        ).fit_resample_weighted(first.X, first.y, first.sample_weight)
        result_manual = compose_weighted_results(first, second)

        np.testing.assert_allclose(
            _as_dense(result_combo.X), _as_dense(result_manual.X)
        )
        np.testing.assert_allclose(
            result_combo.sample_weight,
            result_manual.sample_weight,
            rtol=RTOL,
            atol=ATOL,
        )
        flow_combo = result_combo.contributions.to_scipy_sparse()
        flow_manual = result_manual.contributions.to_scipy_sparse()
        assert (flow_combo != flow_manual).nnz == 0


def test_chained_smote_then_enn_conservation():
    # the composition of two weighted resamplings conserves the mass
    X, y = _make_data()
    sample_weight = np.random.RandomState(14).rand(X.shape[0]) + 0.1
    first = SMOTE(random_state=0).fit_resample_weighted(X, y, sample_weight)
    second = EditedNearestNeighbours(sampling_strategy="all").fit_resample_weighted(
        first.X, first.y, first.sample_weight
    )
    combined = compose_weighted_results(first, second)
    combined.check_conserved(sample_weight, rtol=RTOL, atol=ATOL)
    # the composed transfers are rooted at the original rows
    assert combined.contributions.n_inputs == X.shape[0]
    assert combined.contributions.n_outputs == second.X.shape[0]


def test_pandas_dataframe_with_duplicated_index():
    pd = pytest.importorskip("pandas")
    X, y = _make_data(n_samples=200)
    X = pd.DataFrame(X)
    # duplicated indices must not merge distinct sources
    X.index = [i // 2 for i in range(len(X))]
    y = pd.Series(y)
    sample_weight = np.random.RandomState(15).rand(X.shape[0]) + 0.1

    ros = RandomOverSampler(random_state=0)
    result = ros.fit_resample_weighted(X, y, sample_weight)
    result.check_conserved(sample_weight)
    assert isinstance(result.X, pd.DataFrame)
    # copies are funded by their own positional source, not by an
    # identically-indexed row
    sources = ros.sample_indices_
    n_copies = np.bincount(sources, minlength=X.shape[0])
    for input_idx, output_idx, mass in result.contributions:
        assert sources[output_idx] == input_idx
        np.testing.assert_allclose(mass, sample_weight[input_idx] / n_copies[input_idx])


def test_unsupported_samplers_raise():
    from imblearn.over_sampling import SMOTEN, SMOTENC, BorderlineSMOTE
    from imblearn.under_sampling import RandomUnderSampler

    X, y = _make_data()
    sample_weight = np.ones(X.shape[0])
    # samplers without the method at all refuse with AttributeError
    for sampler in [
        RandomUnderSampler(random_state=0),
        BorderlineSMOTE(random_state=0),
    ]:
        with pytest.raises(AttributeError):
            sampler.fit_resample_weighted(X, y, sample_weight)
    # subclasses of the supported samplers inherit the method but explicitly
    # refuse since their resampling logic differs
    X_cat = np.column_stack([X[:, 0].round(), X[:, 1].round(), X[:, 2:]])
    for sampler in [
        SMOTEN(random_state=0),
        SMOTENC(random_state=0, categorical_features=[0, 1]),
    ]:
        with pytest.raises(NotImplementedError, match="fit_resample_weighted"):
            sampler.fit_resample_weighted(X_cat, y, sample_weight)


def test_identity_resampling_keeps_weights_intact():
    # a sampler with nothing to resample returns the weights unchanged
    X, y = _make_data()
    sample_weight = np.random.RandomState(17).rand(X.shape[0]) + 0.1
    ros = RandomOverSampler(sampling_strategy={})
    result = ros.fit_resample_weighted(X, y, sample_weight)
    np.testing.assert_allclose(result.sample_weight, sample_weight)
    np.testing.assert_array_equal(result.dropped_weight, np.zeros(X.shape[0]))
    result.check_conserved(sample_weight)


def test_weighted_result_picklable():
    # the bookkeeping must be picklable to be cached by joblib.Memory
    import pickle

    X, y = _make_data(n_samples=200)
    sample_weight = np.random.RandomState(18).rand(X.shape[0]) + 0.1
    result = SMOTE(random_state=0).fit_resample_weighted(X, y, sample_weight)
    round_trip = pickle.loads(pickle.dumps(result))
    np.testing.assert_allclose(result.sample_weight, round_trip.sample_weight)
    assert (
        result.contributions.to_scipy_sparse()
        != round_trip.contributions.to_scipy_sparse()
    ).nnz == 0
