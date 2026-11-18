"""Test the weight-conservation bookkeeping primitives."""

# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
# License: MIT

import numpy as np
import pytest
from scipy import sparse

from imblearn.utils._weight_accounting import (
    ResamplingAudit,
    WeightContributions,
    WeightedResamplingResult,
    compose_weighted_results,
    validate_sample_weight,
)


def test_validate_sample_weight_valid():
    # check that a private float64 copy is returned and the caller array is
    # not modified
    weight = np.array([0.5, 1.0, 2.0])
    validated = validate_sample_weight(weight, 3)
    assert validated.dtype == np.float64
    assert_array_equal = np.testing.assert_array_equal
    assert_array_equal(validated, weight)
    validated[0] = 100.0
    assert weight[0] == 0.5

    # lists and integer arrays are accepted
    assert_array_equal(validate_sample_weight([1, 2, 3], 3), [1.0, 2.0, 3.0])
    # a single strictly positive entry is enough
    assert_array_equal(validate_sample_weight([0.0, 0.0, 1.0], 3), [0.0, 0.0, 1.0])


@pytest.mark.parametrize(
    "sample_weight, err_msg",
    [
        (None, "cannot be None"),
        ([[1.0, 2.0], [3.0, 4.0]], "one-dimensional"),
        (np.ones((4, 1)), "one-dimensional"),
        ([1.0, 2.0], "as many entries"),
        ([1.0, np.nan, 1.0], "finite"),
        ([1.0, np.inf, 1.0], "finite"),
        ([1.0, -0.5, 1.0], "negative"),
        ([0.0, 0.0, 0.0], "positive total mass"),
        (["a", "b", "c"], "finite"),
        (np.array([1 + 2j, 3 + 4j, 5 + 6j]), "finite"),
    ],
)
def test_validate_sample_weight_invalid(sample_weight, err_msg):
    with pytest.raises(ValueError, match=err_msg):
        validate_sample_weight(sample_weight, 3)


def test_weight_contributions_records_only_non_zero():
    contributions = WeightContributions(
        [0, 1, 2, 2],
        [0, 0, 1, 2],
        [0.0, 1.5, 0.0, 2.0],
        n_inputs=3,
        n_outputs=3,
    )
    # zero-mass records are dropped
    assert len(contributions) == 2
    assert set(contributions) == {(1, 0, 1.5), (2, 2, 2.0)}
    assert contributions.shape == (3, 3)


def test_weight_contributions_sums():
    contributions = WeightContributions(
        [0, 0, 1, 2],
        [0, 1, 1, 1],
        [1.0, 2.0, 3.0, 4.0],
        n_inputs=3,
        n_outputs=2,
    )
    np.testing.assert_allclose(contributions.input_sums(), [3.0, 3.0, 4.0])
    np.testing.assert_allclose(contributions.output_sums(), [1.0, 9.0])


def test_weight_contributions_validation():
    with pytest.raises(ValueError, match="same length"):
        WeightContributions([0], [0, 1], [1.0], n_inputs=1, n_outputs=2)
    with pytest.raises(ValueError, match="out-of-bound"):
        WeightContributions([5], [0], [1.0], n_inputs=1, n_outputs=2)
    with pytest.raises(ValueError, match="out-of-bound"):
        WeightContributions([0], [-1], [1.0], n_inputs=1, n_outputs=2)
    with pytest.raises(ValueError, match="non-negative"):
        WeightContributions([0], [0], [-1.0], n_inputs=1, n_outputs=2)


def test_weight_contributions_to_scipy_sparse():
    contributions = WeightContributions(
        [0, 0, 1], [1, 1, 0], [1.0, 2.0, 3.0], n_inputs=2, n_outputs=2
    )
    flow = contributions.to_scipy_sparse()
    assert sparse.issparse(flow) and flow.format == "csr"
    # duplicate transfers are summed
    assert flow[0, 1] == 3.0
    assert flow[1, 0] == 3.0
    # round trip
    round_trip = WeightContributions.from_scipy_sparse(flow)
    assert (round_trip.to_scipy_sparse() != flow).nnz == 0


def test_weight_contributions_compose():
    # first stage: row 0 is duplicated, row 1 is kept
    first = WeightContributions(
        [0, 0, 1], [0, 1, 2], [0.5, 0.5, 2.0], n_inputs=2, n_outputs=3
    )
    # second stage: intermediate rows 0 and 2 are kept, row 1 is dropped
    second = WeightContributions([0, 2], [0, 1], [0.5, 2.0], n_inputs=3, n_outputs=2)
    composed = first.compose(second)
    # the original row 0 funds the final row 0 with 0.5, row 1 funds row 1
    assert composed.shape == (2, 2)
    np.testing.assert_allclose(composed.output_sums(), [0.5, 2.0])
    np.testing.assert_allclose(composed.input_sums(), [0.5, 2.0])
    # the dropped mass of the intermediate row 1 (0.5) is charged to row 0
    dropped_next = np.array([0.0, 0.5, 0.0])
    np.testing.assert_allclose(first.lift_dropped(dropped_next), [0.5, 0.0])


def test_weight_contributions_compose_errors():
    first = WeightContributions([0], [0], [1.0], n_inputs=1, n_outputs=1)
    with pytest.raises(TypeError, match="WeightContributions"):
        first.compose("not-a-contribution")
    other = WeightContributions([0, 1], [0, 1], [1.0, 1.0], n_inputs=2, n_outputs=2)
    with pytest.raises(ValueError, match="Cannot compose"):
        first.compose(other)


def test_weight_contributions_compose_zero_weight_rows():
    # a zero-mass intermediate row must not break the composition (0/0 case)
    first = WeightContributions([0], [0], [1.0], n_inputs=2, n_outputs=2)
    second = WeightContributions([0], [0], [1.0], n_inputs=2, n_outputs=2)
    composed = first.compose(second)
    np.testing.assert_allclose(composed.output_sums(), [1.0, 0.0])
    dropped = first.lift_dropped(np.array([0.0, 0.0]))
    np.testing.assert_allclose(dropped, [0.0, 0.0])


def _make_result(sample_weight_in, flows_and_drops, X_blocks):
    """Chain contributions and build the final weighted result."""
    contributions, dropped = flows_and_drops[0]
    for flow_next, dropped_next in flows_and_drops[1:]:
        dropped = dropped + contributions.lift_dropped(dropped_next)
        contributions = contributions.compose(flow_next)
    return WeightedResamplingResult(
        X=X_blocks,
        y=None,
        sample_weight=contributions.output_sums(),
        contributions=contributions,
        dropped_weight=dropped,
    )


def test_weighted_resampling_result_conservation():
    sample_weight_in = np.array([1.0, 2.0, 3.0])
    # stage 1: duplicate row 0, keep rows 1 and 2
    flow1 = WeightContributions(
        [0, 0, 1, 2],
        [0, 1, 2, 3],
        [0.5, 0.5, 2.0, 3.0],
        n_inputs=3,
        n_outputs=4,
    )
    drop1 = np.zeros(3)
    # stage 2: drop intermediate row 2 (funded by input row 1)
    flow2 = WeightContributions(
        [0, 1, 3], [0, 1, 2], [0.5, 0.5, 3.0], n_inputs=4, n_outputs=3
    )
    drop2 = np.array([0.0, 0.0, 2.0, 0.0])
    result = _make_result(
        sample_weight_in, [(flow1, drop1), (flow2, drop2)], X_blocks=np.zeros((3, 1))
    )
    assert result.is_conserved(sample_weight_in)
    result.check_conserved(sample_weight_in)
    np.testing.assert_allclose(result.dropped_weight, [0.0, 2.0, 0.0])
    np.testing.assert_allclose(result.sample_weight, [0.5, 0.5, 3.0])


def test_weighted_resampling_result_check_conserved_raises():
    sample_weight_in = np.array([1.0, 2.0])
    contributions = WeightContributions(
        [0, 1], [0, 1], [1.0, 1.0], n_inputs=2, n_outputs=2
    )
    result = WeightedResamplingResult(
        X=np.zeros((2, 1)),
        y=None,
        sample_weight=np.array([1.0, 1.0]),
        contributions=contributions,
        dropped_weight=np.zeros(2),
    )
    assert not result.is_conserved(sample_weight_in)
    with pytest.raises(AssertionError, match="input row 1"):
        result.check_conserved(sample_weight_in)


def test_compose_weighted_results():
    first = WeightedResamplingResult(
        X=np.zeros((3, 1)),
        y=None,
        sample_weight=np.array([0.5, 0.5, 2.0]),
        contributions=WeightContributions(
            [0, 0, 1], [0, 1, 2], [0.5, 0.5, 2.0], n_inputs=2, n_outputs=3
        ),
        dropped_weight=np.zeros(2),
    )
    second = WeightedResamplingResult(
        X=np.zeros((2, 1)),
        y=None,
        sample_weight=np.array([0.5, 2.0]),
        contributions=WeightContributions(
            [0, 2], [0, 1], [0.5, 2.0], n_inputs=3, n_outputs=2
        ),
        dropped_weight=np.array([0.0, 0.5, 0.0]),
    )
    combined = compose_weighted_results(first, second)
    np.testing.assert_allclose(combined.sample_weight, [0.5, 2.0])
    np.testing.assert_allclose(combined.dropped_weight, [0.5, 0.0])
    np.testing.assert_allclose(combined.contributions.input_sums(), [0.5, 2.0])
    with pytest.raises(TypeError, match="WeightedResamplingResult"):
        compose_weighted_results(first, "not-a-result")


def test_resampling_audit_conservation():
    audit = ResamplingAudit(
        sample_weight_in=np.array([1.0, 2.0]),
        sample_weight_out=np.array([0.5, 0.5, 2.0]),
        contributions=WeightContributions(
            [0, 0, 1], [0, 1, 2], [0.5, 0.5, 2.0], n_inputs=2, n_outputs=3
        ),
        dropped_weight=np.zeros(2),
    )
    assert audit.is_conserved()
    audit.check_conserved()
    # corrupt the audit and check that the verification fails
    bad_audit = ResamplingAudit(
        sample_weight_in=np.array([1.0, 2.0]),
        sample_weight_out=np.array([0.5, 0.5, 1.0]),
        contributions=WeightContributions(
            [0, 0, 1], [0, 1, 2], [0.5, 0.5, 2.0], n_inputs=2, n_outputs=3
        ),
        dropped_weight=np.zeros(2),
    )
    assert not bad_audit.is_conserved()
    with pytest.raises(AssertionError, match="row 2 received"):
        bad_audit.check_conserved()
