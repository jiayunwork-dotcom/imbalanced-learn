"""Tests for the shared integer allocation of over-sampling quotas."""

import numpy as np
import pytest

from imblearn.over_sampling._planning import (
    allocate_integer_quotas,
    neighbors_excluding_self,
)


@pytest.mark.parametrize(
    "weights, n_samples",
    [
        ([1.0, 1.0], 3),
        ([1.0, 2.0, 3.0], 7),
        ([0.1, 0.2, 0.3, 0.4], 5),
        ([1.0, 1.0, 1.0], 1),
        ([0.0, 1.0, 0.0, 1.0], 2),
        ([5.0], 4),
        ([3.0, 0.0], 0),
    ],
)
def test_allocate_integer_quotas_sums_exactly(weights, n_samples):
    quotas = allocate_integer_quotas(weights, n_samples, random_state=0)
    weights = np.asarray(weights, dtype=np.float64)
    assert quotas.sum() == n_samples
    assert np.all(quotas >= 0)
    assert np.all(quotas[weights == 0] == 0)
    # every positive weight gets its ideal share to within one row
    ideal = n_samples * weights / weights.sum()
    assert np.all(np.abs(quotas - ideal) < 1)


def test_allocate_integer_quotas_zero_weights_never_rounded_up():
    # 1 row to generate and 10 sources sharing equal weight: 9 sources get
    # nothing but the zero-weight sources must never be the lucky one.
    weights = np.array([0.0, 0.0, 1.0, 1.0, 1.0])
    quotas = allocate_integer_quotas(weights, 1, random_state=0)
    assert quotas.sum() == 1
    assert quotas[0] == quotas[1] == 0
    assert quotas[2:].sum() == 1


def test_allocate_integer_quotas_largest_remainder():
    # weights sum to 4, 5 rows requested: ideal = 2.5, 1.25, 1.25 -> floors
    # 2, 1, 1 and one row remains; it goes to the largest remainder (0.5).
    quotas = allocate_integer_quotas([2.0, 1.0, 1.0], 5, random_state=0)
    assert quotas.tolist() == [3, 1, 1]
    # weights sum to 10, 3 rows: ideal = 1.8, 0.9, 0.3 -> floors 1, 0, 0,
    # remainders 0.8, 0.9, 0.3; the first remaining row goes to 0.9 and the
    # second to 0.8.
    quotas = allocate_integer_quotas([6.0, 3.0, 1.0], 3, random_state=0)
    assert quotas.tolist() == [2, 1, 0]


def test_allocate_integer_quotas_ties_are_seeded():
    weights = [1.0, 1.0, 1.0, 1.0]
    first = allocate_integer_quotas(weights, 2, random_state=0)
    second = allocate_integer_quotas(weights, 2, random_state=0)
    assert first.tolist() == second.tolist()

    seen = {
        tuple(allocate_integer_quotas(weights, 2, random_state=seed).tolist())
        for seed in range(20)
    }
    # a tie-break by position would always pick sources 0 and 1
    assert len(seen) > 1
    assert any(quota[0] == 0 for quota in seen)


def test_allocate_integer_quotas_tie_does_not_favour_position():
    # With two equally weighted sources and one row, the winner changes with
    # the seed: position is not used as an implicit tie-break.
    winners = {
        int(np.argmax(allocate_integer_quotas([1.0, 1.0], 1, random_state=seed)))
        for seed in range(20)
    }
    assert winners == {0, 1}


def test_allocate_integer_quotas_memory_independent_of_total():
    # A source-by-row table would be 100_000 x 100_000 here; the plan only
    # stores one entry per source.
    weights = np.full(100_000, 1.0)
    quotas = allocate_integer_quotas(weights, 100_000, random_state=0)
    assert quotas.shape == (100_000,)
    assert quotas.sum() == 100_000


@pytest.mark.parametrize(
    "weights, n_samples",
    [
        ([1.0, -0.1], 2),
        ([0.0, 0.0], 2),
        ([np.inf, 1.0], 2),
        ([1.0, 2.0], -1),
        ([[1.0, 2.0]], 2),
        ([], 2),
    ],
)
def test_allocate_integer_quotas_invalid(weights, n_samples):
    with pytest.raises(ValueError):
        allocate_integer_quotas(weights, n_samples, random_state=0)


def test_neighbors_excluding_self_keeps_tied_rows():
    # rows 0 and 1 are identical: the search may return either of them first,
    # but position 1 is another row and must remain a valid neighbour.
    neighbor_indices = np.array([[1, 0, 2, 3]])
    selected = neighbors_excluding_self(neighbor_indices, [0], 2)
    assert selected.tolist() == [[1, 2]]


def test_neighbors_excluding_self_removes_self_anywhere():
    neighbor_indices = np.array([[2, 1, 0, 3]])
    selected = neighbors_excluding_self(neighbor_indices, [0], 3)
    assert selected.tolist() == [[2, 1, 3]]
