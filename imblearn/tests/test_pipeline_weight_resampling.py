"""Test the weight-conserving ``weight_resampling='conserve'`` pipeline mode."""

# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
# License: MIT

import shutil
import tempfile

import numpy as np
import pytest
from scipy import sparse
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.pipeline import Pipeline, make_pipeline
from imblearn.under_sampling import EditedNearestNeighbours, RandomUnderSampler

RTOL = 1e-10
ATOL = 1e-12


def _make_data(n_samples=1000, random_state=10):
    return make_classification(
        n_classes=2,
        class_sep=2,
        weights=[0.1, 0.9],
        n_informative=3,
        n_redundant=1,
        flip_y=0,
        n_features=20,
        n_clusters_per_class=1,
        n_samples=n_samples,
        random_state=random_state,
    )


class RecordingClassifier(ClassifierMixin, BaseEstimator):
    """A classifier recording what its fit and score methods receive."""

    def fit(self, X, y, sample_weight=None, scalar_param=None):
        self.fit_sample_weight_ = (
            None if sample_weight is None else np.asarray(sample_weight).copy()
        )
        self.fit_n_rows_ = X.shape[0]
        self.scalar_param_ = scalar_param
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        return np.zeros(X.shape[0], dtype=int)

    def score(self, X, y, sample_weight=None):
        self.score_sample_weight_ = sample_weight
        self.score_n_rows_ = X.shape[0]
        return 0.5


class NoWeightEstimator(BaseEstimator):
    def fit(self, X, y):
        return self


def _conserve_pipeline(sampler=None, final=None, **kwargs):
    sampler = SMOTE(random_state=42) if sampler is None else sampler
    final = RecordingClassifier() if final is None else final
    return Pipeline(
        [("sampler", sampler), ("clf", final)],
        weight_resampling="conserve",
        **kwargs,
    )


def test_conserve_fit_hands_aligned_weights_to_final_estimator():
    X, y = _make_data()
    sample_weight = np.random.RandomState(0).rand(X.shape[0]) + 0.1
    pipe = _conserve_pipeline()
    pipe.fit(X, y, sample_weight=sample_weight)

    audit = pipe.resampling_audit_
    audit.check_conserved(rtol=RTOL, atol=ATOL)
    clf = pipe.named_steps["clf"]
    # the final estimator received the weights of the final training rows
    assert clf.fit_sample_weight_ is not None
    np.testing.assert_allclose(
        clf.fit_sample_weight_, audit.sample_weight_out, rtol=RTOL, atol=ATOL
    )
    assert clf.fit_n_rows_ == audit.contributions.n_outputs
    # the audit is rooted at the rows passed to fit
    assert audit.contributions.n_inputs == X.shape[0]
    np.testing.assert_allclose(audit.sample_weight_in, sample_weight)
    # total mass is conserved
    assert np.allclose(
        audit.sample_weight_out.sum() + audit.dropped_weight.sum(),
        sample_weight.sum(),
        rtol=RTOL,
        atol=ATOL,
    )


def test_conserve_fit_uniform_weights_by_default():
    X, y = _make_data()
    pipe = _conserve_pipeline()
    pipe.fit(X, y)
    audit = pipe.resampling_audit_
    np.testing.assert_allclose(audit.sample_weight_in, np.ones(X.shape[0]))
    audit.check_conserved()


def test_conserve_with_transformer_interleaved():
    # a row-preserving transformer can be interleaved with the samplers
    X, y = _make_data()
    sample_weight = np.random.RandomState(1).rand(X.shape[0]) + 0.1
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("smote", SMOTE(random_state=42)),
            ("clf", RecordingClassifier()),
        ],
        weight_resampling="conserve",
    )
    pipe.fit(X, y, sample_weight=sample_weight)
    pipe.resampling_audit_.check_conserved(rtol=RTOL, atol=ATOL)

    # passthrough steps are skipped transparently: same books as without them
    pipe_passthrough = Pipeline(
        [
            ("noop", "passthrough"),
            ("smote", SMOTE(random_state=42)),
            ("clf", RecordingClassifier()),
        ],
        weight_resampling="conserve",
    )
    pipe_passthrough.fit(X, y, sample_weight=sample_weight)
    pipe_passthrough.resampling_audit_.check_conserved(rtol=RTOL, atol=ATOL)
    pipe_bare = Pipeline(
        [("smote", SMOTE(random_state=42)), ("clf", RecordingClassifier())],
        weight_resampling="conserve",
    )
    pipe_bare.fit(X, y, sample_weight=sample_weight)
    np.testing.assert_allclose(
        pipe_passthrough.resampling_audit_.sample_weight_out,
        pipe_bare.resampling_audit_.sample_weight_out,
    )


def test_conserve_several_samplers_chained():
    # the audit is rooted at the original rows, not at intermediate rows
    X, y = _make_data()
    sample_weight = np.random.RandomState(2).rand(X.shape[0]) + 0.1
    pipe = Pipeline(
        [
            ("ros", RandomOverSampler(random_state=0)),
            ("smote", SMOTE(random_state=0)),
            ("enn", EditedNearestNeighbours(sampling_strategy="all")),
            ("clf", RecordingClassifier()),
        ],
        weight_resampling="conserve",
    )
    pipe.fit(X, y, sample_weight=sample_weight)
    audit = pipe.resampling_audit_
    audit.check_conserved(rtol=RTOL, atol=ATOL)
    assert audit.contributions.n_inputs == X.shape[0]
    assert audit.dropped_weight.shape == (X.shape[0],)
    # ENN dropped some mass
    assert audit.dropped_weight.sum() > 0

    # the audit matches the manual composition of the three stages
    from imblearn.utils import compose_weighted_results

    first = RandomOverSampler(random_state=0).fit_resample_weighted(X, y, sample_weight)
    second = SMOTE(random_state=0).fit_resample_weighted(
        first.X, first.y, first.sample_weight
    )
    third = EditedNearestNeighbours(sampling_strategy="all").fit_resample_weighted(
        second.X, second.y, second.sample_weight
    )
    manual = compose_weighted_results(compose_weighted_results(first, second), third)
    np.testing.assert_allclose(
        audit.sample_weight_out, manual.sample_weight, rtol=RTOL, atol=ATOL
    )
    np.testing.assert_allclose(
        audit.dropped_weight, manual.dropped_weight, rtol=RTOL, atol=ATOL
    )
    flow_pipe = audit.contributions.to_scipy_sparse()
    flow_manual = manual.contributions.to_scipy_sparse()
    assert (flow_pipe != flow_manual).nnz == 0


def test_conserve_smoteenn_matches_explicit_steps():
    X, y = _make_data()
    sample_weight = np.random.RandomState(3).rand(X.shape[0]) + 0.1
    pipe_combo = Pipeline(
        [("combo", SMOTEENN(random_state=42)), ("clf", RecordingClassifier())],
        weight_resampling="conserve",
    )
    pipe_combo.fit(X, y, sample_weight=sample_weight)

    pipe_steps = Pipeline(
        [
            ("smote", SMOTE(random_state=42)),
            ("enn", EditedNearestNeighbours(sampling_strategy="all")),
            ("clf", RecordingClassifier()),
        ],
        weight_resampling="conserve",
    )
    pipe_steps.fit(X, y, sample_weight=sample_weight)

    audit_combo = pipe_combo.resampling_audit_
    audit_steps = pipe_steps.resampling_audit_
    np.testing.assert_allclose(
        audit_combo.sample_weight_out,
        audit_steps.sample_weight_out,
        rtol=RTOL,
        atol=ATOL,
    )
    np.testing.assert_allclose(
        audit_combo.dropped_weight,
        audit_steps.dropped_weight,
        rtol=RTOL,
        atol=ATOL,
    )
    flow_combo = audit_combo.contributions.to_scipy_sparse()
    flow_steps = audit_steps.contributions.to_scipy_sparse()
    assert (flow_combo != flow_steps).nnz == 0


def test_conserve_csr_input_stays_sparse():
    X, y = _make_data()
    X = sparse.csr_matrix(X)
    sample_weight = np.random.RandomState(4).rand(X.shape[0]) + 0.1
    pipe = _conserve_pipeline()
    pipe.fit(X, y, sample_weight=sample_weight)
    pipe.resampling_audit_.check_conserved(rtol=RTOL, atol=ATOL)


def test_conserve_repeated_fit_rebuilds_audit():
    X, y = _make_data()
    sample_weight = np.random.RandomState(5).rand(X.shape[0]) + 0.1
    pipe = _conserve_pipeline()
    pipe.fit(X, y, sample_weight=sample_weight)
    first_out = pipe.resampling_audit_.sample_weight_out.copy()
    # refitting with other weights rebuilds the books from scratch
    sample_weight_2 = sample_weight * 4
    pipe.fit(X, y, sample_weight=sample_weight_2)
    np.testing.assert_allclose(pipe.resampling_audit_.sample_weight_in, sample_weight_2)
    np.testing.assert_allclose(
        pipe.resampling_audit_.sample_weight_out,
        first_out * 4,
        rtol=RTOL,
        atol=ATOL,
    )
    pipe.resampling_audit_.check_conserved(rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize(
    "sample_weight",
    [
        np.ones(5),  # wrong length
        np.full(1000, np.nan),  # non-finite
        -np.ones(1000),  # negative
        np.zeros(1000),  # zero total mass
        np.ones((1000, 1)),  # not 1-D
        "not-an-array",  # not numeric
    ],
    ids=["length", "nan", "negative", "zero-sum", "2d", "string"],
)
def test_conserve_invalid_sample_weight_never_fits_final_estimator(sample_weight):
    X, y = _make_data()
    pipe = _conserve_pipeline()
    with pytest.raises(ValueError):
        pipe.fit(X, y, sample_weight=sample_weight)
    # neither the final estimator nor the sampler was fitted
    assert not hasattr(pipe.named_steps["clf"], "fit_sample_weight_")
    assert not hasattr(pipe.named_steps["sampler"], "sampling_strategy_")
    assert not hasattr(pipe, "resampling_audit_")


def test_conserve_rejects_unsupported_sampler():
    X, y = _make_data()
    sample_weight = np.ones(X.shape[0])
    pipe = _conserve_pipeline(sampler=RandomUnderSampler(random_state=0))
    with pytest.raises(TypeError, match="fit_resample_weighted"):
        pipe.fit(X, y, sample_weight=sample_weight)
    assert not hasattr(pipe.named_steps["clf"], "fit_sample_weight_")


def test_conserve_rejects_final_estimator_without_sample_weight():
    X, y = _make_data()
    sample_weight = np.ones(X.shape[0])
    pipe = _conserve_pipeline(final=NoWeightEstimator())
    with pytest.raises(TypeError, match="sample_weight"):
        pipe.fit(X, y, sample_weight=sample_weight)


def test_conserve_rejects_conflicting_sample_weight_routing():
    X, y = _make_data()
    sample_weight = np.ones(X.shape[0])
    pipe = _conserve_pipeline()
    with pytest.raises(ValueError, match="managed by the weight-conserving"):
        pipe.fit(X, y, sample_weight=sample_weight, clf__sample_weight=sample_weight)


def test_conserve_scalar_fit_params_still_routed():
    X, y = _make_data()
    sample_weight = np.random.RandomState(6).rand(X.shape[0]) + 0.1
    pipe = _conserve_pipeline()
    pipe.fit(X, y, sample_weight=sample_weight, clf__scalar_param=7)
    assert pipe.named_steps["clf"].scalar_param_ == 7


def test_conserve_metadata_routing_enabled():
    sklearn = pytest.importorskip("sklearn")
    X, y = _make_data()
    sample_weight = np.random.RandomState(7).rand(X.shape[0]) + 0.1
    with sklearn.config_context(enable_metadata_routing=True):
        clf = RecordingClassifier().set_fit_request(scalar_param=True)
        clf.set_score_request(sample_weight=True)
        pipe = Pipeline(
            [("smote", SMOTE(random_state=42)), ("clf", clf)],
            weight_resampling="conserve",
        )
        pipe.fit(X, y, sample_weight=sample_weight, scalar_param=3)
        assert pipe.named_steps["clf"].scalar_param_ == 3
        np.testing.assert_allclose(
            pipe.named_steps["clf"].fit_sample_weight_,
            pipe.resampling_audit_.sample_weight_out,
        )
        pipe.resampling_audit_.check_conserved(rtol=RTOL, atol=ATOL)
        # score weights refer to the test rows, no resampling happens
        pipe.score(X[:10], y[:10], sample_weight=sample_weight[:10])
        assert len(pipe.named_steps["clf"].score_sample_weight_) == 10


def test_predict_and_score_do_not_resample():
    X, y = _make_data()
    sample_weight = np.random.RandomState(8).rand(X.shape[0]) + 0.1
    pipe = _conserve_pipeline()
    pipe.fit(X, y, sample_weight=sample_weight)
    X_test, y_test = _make_data(n_samples=100, random_state=77)
    weight_test = np.random.RandomState(9).rand(100) + 0.1
    y_pred = pipe.predict(X_test)
    assert y_pred.shape == (100,)
    pipe.score(X_test, y_test, sample_weight=weight_test)
    clf = pipe.named_steps["clf"]
    # score received the weights of the test rows, unchanged
    np.testing.assert_allclose(clf.score_sample_weight_, weight_test)
    assert clf.score_n_rows_ == 100


def test_conserve_with_memory_cache():
    X, y = _make_data()
    sample_weight = np.random.RandomState(10).rand(X.shape[0]) + 0.1
    cachedir = tempfile.mkdtemp()
    try:
        kwargs = {"memory": cachedir}
        pipe = _conserve_pipeline(**kwargs)
        pipe.fit(X, y, sample_weight=sample_weight)
        cold_out = pipe.resampling_audit_.sample_weight_out.copy()
        cold_flow = pipe.resampling_audit_.contributions.to_scipy_sparse().copy()

        # a fresh pipeline hits the warm cache and gets the same books
        pipe_warm = _conserve_pipeline(**kwargs)
        pipe_warm.fit(X, y, sample_weight=sample_weight)
        np.testing.assert_allclose(
            cold_out, pipe_warm.resampling_audit_.sample_weight_out
        )
        warm_flow = pipe_warm.resampling_audit_.contributions.to_scipy_sparse()
        assert (cold_flow != warm_flow).nnz == 0

        # the same data with different weights must not hit the old entry
        sample_weight_2 = sample_weight * 2 + 0.3
        pipe_warm.fit(X, y, sample_weight=sample_weight_2)
        assert not np.allclose(pipe_warm.resampling_audit_.sample_weight_out, cold_out)
        pipe_warm.resampling_audit_.check_conserved(rtol=RTOL, atol=ATOL)

        # and recomputing without cache gives the same result as the cache
        pipe_cold = _conserve_pipeline()
        pipe_cold.fit(X, y, sample_weight=sample_weight_2)
        np.testing.assert_allclose(
            pipe_warm.resampling_audit_.sample_weight_out,
            pipe_cold.resampling_audit_.sample_weight_out,
            rtol=RTOL,
            atol=ATOL,
        )
    finally:
        shutil.rmtree(cachedir)


def test_conserve_clone():
    pipe = _conserve_pipeline()
    cloned = clone(pipe)
    assert cloned.weight_resampling == "conserve"
    assert cloned.named_steps["sampler"].random_state == 42


def test_make_pipeline_forwards_weight_resampling():
    pipe = make_pipeline(
        SMOTE(random_state=0), LogisticRegression(), weight_resampling="conserve"
    )
    assert pipe.weight_resampling == "conserve"


def test_default_pipeline_unchanged_by_weight_resampling_param():
    # without weight_resampling='conserve', a bare sample_weight keeps the
    # historical behaviour: it is rejected
    X, y = _make_data()
    pipe = Pipeline([("smote", SMOTE(random_state=42)), ("clf", RecordingClassifier())])
    with pytest.raises(ValueError, match="does not accept the sample_weight"):
        pipe.fit(X, y, sample_weight=np.ones(X.shape[0]))
    # and no audit is left behind
    assert not hasattr(pipe, "resampling_audit_")


def test_conserve_invalid_weight_resampling_value():
    pipe = Pipeline(
        [("smote", SMOTE(random_state=42)), ("clf", RecordingClassifier())],
        weight_resampling="inflate",
    )
    X, y = _make_data()
    with pytest.raises(Exception, match="weight_resampling"):
        pipe.fit(X, y, sample_weight=np.ones(X.shape[0]))


def test_conserve_end_to_end_training():
    # a small but realistic training chain: scale, resample with weight
    # conservation, train a logistic regression and evaluate it
    X, y = _make_data(n_samples=2000)
    sample_weight = np.random.RandomState(11).rand(X.shape[0]) + 0.1
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("smote", SMOTE(random_state=0)),
            ("clf", LogisticRegression(max_iter=500)),
        ],
        weight_resampling="conserve",
    )
    pipe.fit(X, y, sample_weight=sample_weight)
    pipe.resampling_audit_.check_conserved(rtol=RTOL, atol=ATOL)
    # the conserved weights change the training compared to no weights
    pipe_plain = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("smote", SMOTE(random_state=0)),
            ("clf", LogisticRegression(max_iter=500)),
        ]
    )
    pipe_plain.fit(X, y)
    assert not np.allclose(
        pipe.named_steps["clf"].coef_, pipe_plain.named_steps["clf"].coef_
    )
    # the model is still usable as usual
    assert pipe.score(X, y) > 0.9
